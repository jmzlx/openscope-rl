"""
Decision Transformer for OpenScope ATC.

This module implements the Decision Transformer architecture for offline RL,
treating reinforcement learning as a sequence modeling problem.

Key features:
- Predicts actions from (return-to-go, state, action) sequences
- No value functions or TD learning - pure supervised learning
- Condition on desired return to control behavior at test time
- Works with mixed-quality offline data

Reference: "Decision Transformer: Reinforcement Learning via Sequence Modeling"
"""

import logging
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from .encoders import ATCTransformerEncoder
from .config import EncoderConfig

logger = logging.getLogger(__name__)


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for OpenScope ATC.

    The model processes sequences of (return-to-go, state, action) tuples and
    predicts the next action to take. At test time, conditioning on high returns
    produces better policies.

    Architecture:
        1. Embed returns-to-go, states, and actions into tokens
        2. Interleave tokens: [RTG_1, S_1, A_1, RTG_2, S_2, A_2, ...]
        3. Process with causal transformer (GPT-2 style)
        4. Predict actions from state tokens

    Example:
        >>> model = DecisionTransformer(
        ...     state_dim=14,
        ...     act_dim=65,  # Total action dimensions
        ...     max_aircraft=20,
        ...     hidden_size=256,
        ...     max_ep_len=1000,
        ...     context_len=20
        ... )
        >>> # Training
        >>> returns = torch.randn(8, 20, 1)  # batch, seq_len, 1
        >>> states = torch.randn(8, 20, 20, 14)  # batch, seq_len, max_aircraft, features
        >>> masks = torch.ones(8, 20, 20, dtype=torch.bool)
        >>> actions = torch.randint(0, 65, (8, 20))
        >>> timesteps = torch.arange(20).unsqueeze(0).expand(8, -1)
        >>> action_preds = model(returns, states, masks, actions, timesteps)
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_aircraft: int,
        hidden_size: int = 256,
        max_ep_len: int = 1000,
        context_len: int = 20,
        n_layer: int = 6,
        n_head: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize Decision Transformer.

        Args:
            state_dim: Dimension of aircraft state features (14)
            act_dim: Total action dimension (aircraft_id + command_type + params)
            max_aircraft: Maximum number of aircraft (20)
            hidden_size: Hidden dimension for transformer
            max_ep_len: Maximum episode length for positional embeddings
            context_len: Context window size (number of timesteps)
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_aircraft = max_aircraft
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len
        self.context_len = context_len

        # Token embeddings
        # Return-to-go: scalar -> hidden_size
        self.embed_return = nn.Linear(1, hidden_size)

        # State: use ATCTransformerEncoder to process aircraft sequences
        encoder_config = EncoderConfig(
            input_dim=state_dim,
            hidden_dim=hidden_size,
            num_heads=4,  # Fewer heads for state encoder
            num_layers=2,  # Fewer layers for state encoder
            dropout=dropout,
        )
        self.embed_state = ATCTransformerEncoder(encoder_config)

        # Pool aircraft states to single vector (mean pooling with masking)
        self.state_pooling = nn.Linear(hidden_size, hidden_size)

        # Action: discrete action index -> hidden_size
        self.embed_action = nn.Embedding(act_dim, hidden_size)

        # Positional embedding for timesteps
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        # Layer normalization
        self.embed_ln = nn.LayerNorm(hidden_size)

        # GPT-2 style causal transformer
        config = GPT2Config(
            vocab_size=1,  # Not used, we provide embeddings directly
            n_positions=3 * context_len,  # 3 tokens per timestep (RTG, state, action)
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * hidden_size,
            activation_function=activation,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2Model(config)

        # Action prediction head
        self.predict_action = nn.Linear(hidden_size, act_dim)

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"Initialized DecisionTransformer with {self.get_num_params():,} parameters")

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
        aircraft_masks: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            returns: Return-to-go tensor (batch_size, seq_len, 1)
            states: State tensor (batch_size, seq_len, max_aircraft, state_dim)
            aircraft_masks: Aircraft validity mask (batch_size, seq_len, max_aircraft)
            actions: Action indices (batch_size, seq_len)
            timesteps: Timestep indices (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Action logits (batch_size, seq_len, act_dim)
        """
        batch_size, seq_len = returns.shape[0], returns.shape[1]

        # Validate inputs
        assert states.shape[0] == batch_size
        assert states.shape[1] == seq_len
        assert actions.shape == (batch_size, seq_len)
        assert timesteps.shape == (batch_size, seq_len)

        # Embed returns
        # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_size)
        return_embeddings = self.embed_return(returns)

        # Embed states
        # Process each timestep's state separately
        state_embeddings = []
        for t in range(seq_len):
            # (batch_size, max_aircraft, state_dim)
            state_t = states[:, t, :, :]
            mask_t = aircraft_masks[:, t, :]

            # Encode with transformer
            encoded = self.embed_state(state_t, mask_t)  # (batch_size, max_aircraft, hidden_size)

            # Mean pooling over aircraft (respecting mask)
            mask_expanded = mask_t.unsqueeze(-1).float()  # (batch_size, max_aircraft, 1)
            pooled = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
            pooled = self.state_pooling(pooled)  # (batch_size, hidden_size)

            state_embeddings.append(pooled)

        # Stack state embeddings
        # (batch_size, seq_len, hidden_size)
        state_embeddings = torch.stack(state_embeddings, dim=1)

        # Embed actions
        # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        action_embeddings = self.embed_action(actions)

        # Embed timesteps
        # (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        time_embeddings = self.embed_timestep(timesteps)

        # Add timestep embeddings to all tokens
        return_embeddings = return_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # Interleave tokens: [RTG_1, S_1, A_1, RTG_2, S_2, A_2, ...]
        # Shape: (batch_size, 3 * seq_len, hidden_size)
        stacked_inputs = torch.stack(
            [return_embeddings, state_embeddings, action_embeddings], dim=2
        ).reshape(batch_size, 3 * seq_len, self.hidden_size)

        # Apply layer norm
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Create attention mask if needed
        if attention_mask is not None:
            # Expand attention mask to cover all tokens
            attention_mask = torch.repeat_interleave(attention_mask, 3, dim=1)

        # Pass through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask,
        )
        x = transformer_outputs.last_hidden_state

        # Extract state tokens (predict action from state)
        # State tokens are at positions 1, 4, 7, ...
        state_tokens = x[:, 1::3, :]  # (batch_size, seq_len, hidden_size)

        # Predict actions
        action_preds = self.predict_action(state_tokens)  # (batch_size, seq_len, act_dim)

        return action_preds

    def get_action(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
        aircraft_masks: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor]:
        """
        Get action for inference (single timestep prediction).

        Args:
            returns: Return-to-go context (batch_size, context_len, 1)
            states: State context (batch_size, context_len, max_aircraft, state_dim)
            aircraft_masks: Aircraft mask context (batch_size, context_len, max_aircraft)
            actions: Action context (batch_size, context_len)
            timesteps: Timestep context (batch_size, context_len)
            temperature: Sampling temperature (higher = more random)
            deterministic: If True, take argmax; if False, sample

        Returns:
            Tuple of (action index, action logits)
        """
        # Get action predictions for the last timestep
        with torch.no_grad():
            action_preds = self.forward(
                returns, states, aircraft_masks, actions, timesteps
            )

            # Take last timestep prediction
            logits = action_preds[:, -1, :]  # (batch_size, act_dim)

            # Apply temperature
            logits = logits / temperature

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return action.item() if action.shape[0] == 1 else action, logits


class MultiDiscreteDecisionTransformer(DecisionTransformer):
    """
    Decision Transformer variant that handles multi-discrete action spaces.

    Instead of predicting a single action index, this model predicts each
    action component separately: aircraft_id, command_type, altitude, heading, speed.

    This is more natural for OpenScope where we have a structured action space.
    """

    def __init__(
        self,
        state_dim: int,
        max_aircraft: int,
        action_dims: Dict[str, int],
        hidden_size: int = 256,
        max_ep_len: int = 1000,
        context_len: int = 20,
        n_layer: int = 6,
        n_head: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize multi-discrete Decision Transformer.

        Args:
            state_dim: Dimension of aircraft state features
            max_aircraft: Maximum number of aircraft
            action_dims: Dictionary mapping action component names to dimensions
                Example: {"aircraft_id": 21, "command_type": 5, "altitude": 18, ...}
            hidden_size: Hidden dimension
            max_ep_len: Maximum episode length
            context_len: Context window size
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            dropout: Dropout rate
            activation: Activation function
        """
        # Calculate total action dimension for embedding
        self.action_dims = action_dims
        self.action_keys = list(action_dims.keys())
        total_act_dim = sum(action_dims.values())

        super().__init__(
            state_dim=state_dim,
            act_dim=total_act_dim,
            max_aircraft=max_aircraft,
            hidden_size=hidden_size,
            max_ep_len=max_ep_len,
            context_len=context_len,
            n_layer=n_layer,
            n_head=n_head,
            dropout=dropout,
            activation=activation,
        )

        # Replace single action prediction head with multiple heads
        self.predict_action = nn.ModuleDict({
            key: nn.Linear(hidden_size, dim)
            for key, dim in action_dims.items()
        })

    def forward(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
        aircraft_masks: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-discrete actions.

        Args:
            returns: Return-to-go tensor (batch_size, seq_len, 1)
            states: State tensor (batch_size, seq_len, max_aircraft, state_dim)
            aircraft_masks: Aircraft validity mask (batch_size, seq_len, max_aircraft)
            actions: Dictionary of action tensors, each (batch_size, seq_len)
            timesteps: Timestep indices (batch_size, seq_len)
            attention_mask: Optional attention mask (batch_size, seq_len)

        Returns:
            Dictionary of action logits for each component
        """
        batch_size, seq_len = returns.shape[0], returns.shape[1]

        # Create combined action embeddings from all components
        action_embeds = []
        for key in self.action_keys:
            action_embeds.append(self.embed_action(actions[key]))

        # Average all action embeddings
        combined_action_embed = torch.stack(action_embeds, dim=0).mean(dim=0)

        # Embed returns and states (same as before)
        return_embeddings = self.embed_return(returns)

        state_embeddings = []
        for t in range(seq_len):
            state_t = states[:, t, :, :]
            mask_t = aircraft_masks[:, t, :]
            encoded = self.embed_state(state_t, mask_t)
            mask_expanded = mask_t.unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-6)
            pooled = self.state_pooling(pooled)
            state_embeddings.append(pooled)

        state_embeddings = torch.stack(state_embeddings, dim=1)

        # Embed timesteps
        time_embeddings = self.embed_timestep(timesteps)

        # Add timestep embeddings
        return_embeddings = return_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        combined_action_embed = combined_action_embed + time_embeddings

        # Interleave tokens
        stacked_inputs = torch.stack(
            [return_embeddings, state_embeddings, combined_action_embed], dim=2
        ).reshape(batch_size, 3 * seq_len, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        if attention_mask is not None:
            attention_mask = torch.repeat_interleave(attention_mask, 3, dim=1)

        # Transformer forward
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=attention_mask,
        )
        x = transformer_outputs.last_hidden_state

        # Extract state tokens
        state_tokens = x[:, 1::3, :]

        # Predict each action component separately
        action_preds = {
            key: head(state_tokens)
            for key, head in self.predict_action.items()
        }

        return action_preds

    def get_action(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
        aircraft_masks: torch.Tensor,
        actions: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
        """
        Get action for inference.

        Returns:
            Tuple of (action dict, logits dict)
        """
        with torch.no_grad():
            action_preds = self.forward(
                returns, states, aircraft_masks, actions, timesteps
            )

            # Take last timestep prediction for each component
            selected_actions = {}
            all_logits = {}

            for key, logits in action_preds.items():
                logits_last = logits[:, -1, :] / temperature
                all_logits[key] = logits_last

                if deterministic:
                    action = logits_last.argmax(dim=-1)
                else:
                    probs = F.softmax(logits_last, dim=-1)
                    action = torch.multinomial(probs, num_samples=1).squeeze(-1)

                selected_actions[key] = action.item() if action.shape[0] == 1 else action

        return selected_actions, all_logits
