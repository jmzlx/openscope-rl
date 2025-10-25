# Decision Transformer - Implementation Guide

**Worktree**: `.trees/05-decision-transformer/`
**Branch**: `experiment/05-decision-transformer`
**Priority**: ⭐⭐ HIGH (Transformer-based, sample-efficient!)

## Objective

Implement Decision Transformer (DT) for offline RL. Learn from fixed dataset of episodes without online environment interaction. DT treats RL as sequence modeling: given (return-to-go, state, action) sequences, predict next action.

## Why Decision Transformer?

**Key advantages over PPO:**
1. **Offline learning**: Train on pre-collected data (no expensive environment interaction)
2. **Return conditioning**: Control behavior by specifying desired return at test time
3. **Sample efficient**: Learn from mixed-quality data (random, suboptimal, expert)
4. **Simple**: No value functions, no bootstrapping, just supervised learning!

**Perfect for OpenScope because:**
- Can learn from your POC episode recordings!
- Transformer architecture (you already have this!)
- Works with sparse rewards and long horizons

## Implementation Steps

### Step 1: Implement Decision Transformer Model (3-4 hours)

Create `models/decision_transformer.py`:

```python
"""Decision Transformer for OpenScope ATC."""

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Model, GPT2Config

class DecisionTransformer(nn.Module):
    """
    Decision Transformer for OpenScope.

    Treats RL as conditional sequence modeling.
    Input: (return-to-go, state, action) * K timesteps
    Output: Next action prediction
    """

    def __init__(
        self,
        state_dim=14,  # Aircraft feature dim
        act_dim=5,  # Number of action components
        hidden_size=256,
        max_ep_len=1000,
        max_aircraft=10,
        context_len=20,  # Number of timesteps to condition on
        n_layer=6,
        n_head=8,
        dropout=0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len
        self.max_aircraft = max_aircraft
        self.context_len = context_len

        # Token embeddings
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)

        # State embedding (use existing transformer encoder from main repo!)
        from models.encoders import ATCTransformerEncoder, EncoderConfig
        encoder_config = EncoderConfig(
            input_dim=state_dim,
            hidden_dim=hidden_size,
            num_layers=2,
            num_heads=4,
        )
        self.embed_state = ATCTransformerEncoder(encoder_config)

        # Action embedding (multi-discrete actions)
        self.embed_action = nn.ModuleDict({
            'aircraft_id': nn.Embedding(max_aircraft + 1, hidden_size // 5),
            'command_type': nn.Embedding(5, hidden_size // 5),
            'altitude': nn.Embedding(18, hidden_size // 5),
            'heading': nn.Embedding(13, hidden_size // 5),
            'speed': nn.Embedding(8, hidden_size // 5),
        })
        self.action_proj = nn.Linear(hidden_size, hidden_size)

        # Causal Transformer (GPT-2 architecture)
        gpt_config = GPT2Config(
            vocab_size=1,  # Not used
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=3 * context_len,  # (R, s, a) * context_len
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2Model(gpt_config)

        # Action prediction heads (multi-discrete)
        self.predict_action = nn.ModuleDict({
            'aircraft_id': nn.Linear(hidden_size, max_aircraft + 1),
            'command_type': nn.Linear(hidden_size, 5),
            'altitude': nn.Linear(hidden_size, 18),
            'heading': nn.Linear(hidden_size, 13),
            'speed': nn.Linear(hidden_size, 8),
        })

        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_size)

    def forward(self, returns_to_go, states, actions, timesteps):
        """
        Forward pass for training.

        Args:
            returns_to_go: (batch, seq_len, 1) - Returns-to-go at each timestep
            states: (batch, seq_len, max_aircraft, state_dim) - States
            actions: (batch, seq_len, 5) - Actions (multi-discrete indices)
            timesteps: (batch, seq_len) - Timestep indices

        Returns:
            Action logits for each component
        """
        batch_size, seq_len = returns_to_go.shape[0], returns_to_go.shape[1]

        # Embed each modality
        time_embeddings = self.embed_timestep(timesteps)  # (B, T, H)

        # Returns-to-go embedding
        returns_embeddings = self.embed_return(returns_to_go)  # (B, T, H)

        # State embedding (process each timestep's aircraft)
        state_embeddings = []
        for t in range(seq_len):
            # Get aircraft mask (which aircraft are valid)
            aircraft_mask = torch.any(states[:, t] != 0, dim=-1)  # (B, max_aircraft)
            # Encode aircraft states
            encoded = self.embed_state(states[:, t], aircraft_mask)  # (B, max_aircraft, H)
            # Pool across aircraft
            pooled = encoded.mean(dim=1)  # (B, H)
            state_embeddings.append(pooled)
        state_embeddings = torch.stack(state_embeddings, dim=1)  # (B, T, H)

        # Action embedding
        action_embeddings = []
        for t in range(seq_len):
            # Embed each action component
            embedded = torch.cat([
                self.embed_action['aircraft_id'](actions[:, t, 0]),
                self.embed_action['command_type'](actions[:, t, 1]),
                self.embed_action['altitude'](actions[:, t, 2]),
                self.embed_action['heading'](actions[:, t, 3]),
                self.embed_action['speed'](actions[:, t, 4]),
            ], dim=-1)  # (B, H)
            action_embeddings.append(embedded)
        action_embeddings = torch.stack(action_embeddings, dim=1)  # (B, T, H)
        action_embeddings = self.action_proj(action_embeddings)

        # Add time embeddings to all
        returns_embeddings = returns_embeddings + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # Stack tokens: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Shape: (B, 3*T, H)
        stacked_inputs = torch.stack(
            [returns_embeddings, state_embeddings, action_embeddings], dim=2
        ).reshape(batch_size, 3 * seq_len, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Transformer forward pass
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        hidden_states = transformer_outputs.last_hidden_state

        # Extract action predictions (from state tokens)
        # State tokens are at positions 1, 4, 7, ... (every 3rd starting from 1)
        action_hidden = hidden_states[:, 1::3, :]  # (B, T, H)

        # Predict actions
        action_preds = {
            'aircraft_id': self.predict_action['aircraft_id'](action_hidden),
            'command_type': self.predict_action['command_type'](action_hidden),
            'altitude': self.predict_action['altitude'](action_hidden),
            'heading': self.predict_action['heading'](action_hidden),
            'speed': self.predict_action['speed'](action_hidden),
        }

        return action_preds

    @torch.no_grad()
    def get_action(self, returns_to_go, states, actions, timesteps):
        """
        Get action at inference time (last timestep only).

        Args:
            Same as forward, but only predicts action for last timestep

        Returns:
            Dictionary of action indices
        """
        # Ensure correct shapes
        if returns_to_go.dim() == 2:
            returns_to_go = returns_to_go.unsqueeze(0)
        if states.dim() == 3:
            states = states.unsqueeze(0)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(0)

        # Forward pass
        action_preds = self(returns_to_go, states, actions, timesteps)

        # Sample from last timestep
        action = {
            key: torch.argmax(logits[:, -1], dim=-1)
            for key, logits in action_preds.items()
        }

        return action
```

### Step 2: Collect Offline Dataset (2-3 hours)

Create `data/offline_dataset.py`:

```python
"""Collect offline dataset for Decision Transformer training."""

import numpy as np
import pickle
from tqdm import tqdm

from environment import PlaywrightEnv, create_default_config


def collect_episode(env, policy='random'):
    """Collect single episode."""
    states, actions, rewards = [], [], []

    obs, info = env.reset()
    done = False

    while not done:
        # Get action (random or heuristic)
        if policy == 'random':
            action = env.action_space.sample()
        elif policy == 'heuristic':
            action = heuristic_policy(obs, info)  # Simple rule-based

        states.append(obs)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        done = terminated or truncated

    # Compute returns-to-go
    returns_to_go = []
    rtg = 0
    for r in reversed(rewards):
        rtg += r
        returns_to_go.insert(0, rtg)

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'returns_to_go': returns_to_go,
        'episode_return': sum(rewards),
    }


def heuristic_policy(obs, info):
    """Simple rule-based policy for better data quality."""
    # TODO: Implement simple heuristics
    # - Maintain separation
    # - Guide to exits
    # - Issue ILS when near runway
    pass


def collect_dataset(num_episodes=1000, save_path='offline_dataset.pkl'):
    """Collect full dataset."""
    config = create_default_config(headless=True)
    env = PlaywrightEnv(**config.__dict__)

    dataset = []

    for i in tqdm(range(num_episodes), desc="Collecting episodes"):
        # Mix of random and heuristic policies
        policy = 'random' if i % 2 == 0 else 'heuristic'
        episode = collect_episode(env, policy)
        dataset.append(episode)

    env.close()

    # Save dataset
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Collected {num_episodes} episodes")
    print(f"Average return: {np.mean([ep['episode_return'] for ep in dataset]):.2f}")
    print(f"Saved to {save_path}")

    return dataset
```

### Step 3: Training Script (2-3 hours)

Create `training/dt_trainer.py`:

```python
"""Train Decision Transformer on offline data."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb
import pickle
import numpy as np

from models.decision_transformer import DecisionTransformer


class OfflineDataset(Dataset):
    """Dataset for Decision Transformer training."""

    def __init__(self, episodes, context_len=20):
        self.episodes = episodes
        self.context_len = context_len

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode = self.episodes[idx]

        # Get full episode
        states = np.array([s['aircraft'] for s in episode['states']])
        actions = np.array([[a[k] for k in ['aircraft_id', 'command_type',
                                            'altitude', 'heading', 'speed']]
                           for a in episode['actions']])
        returns_to_go = np.array(episode['returns_to_go']).reshape(-1, 1)
        timesteps = np.arange(len(states))

        # Random crop to context_len
        if len(states) > self.context_len:
            start = np.random.randint(0, len(states) - self.context_len)
            states = states[start:start + self.context_len]
            actions = actions[start:start + self.context_len]
            returns_to_go = returns_to_go[start:start + self.context_len]
            timesteps = timesteps[start:start + self.context_len]
        else:
            # Pad if too short
            pad_len = self.context_len - len(states)
            states = np.pad(states, ((0, pad_len), (0, 0), (0, 0)))
            actions = np.pad(actions, ((0, pad_len), (0, 0)))
            returns_to_go = np.pad(returns_to_go, ((0, pad_len), (0, 0)))
            timesteps = np.pad(timesteps, (0, pad_len))

        return {
            'states': torch.FloatTensor(states),
            'actions': torch.LongTensor(actions),
            'returns_to_go': torch.FloatTensor(returns_to_go),
            'timesteps': torch.LongTensor(timesteps),
        }


def train_decision_transformer(
    dataset_path='offline_dataset.pkl',
    epochs=50,
    batch_size=64,
    lr=1e-4,
):
    """Train Decision Transformer."""

    # Load dataset
    with open(dataset_path, 'rb') as f:
        episodes = pickle.load(f)

    # Create dataset and dataloader
    dataset = OfflineDataset(episodes, context_len=20)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = DecisionTransformer().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # WandB
    wandb.init(project="openscope-rl-dt", entity="jmzlx.ai")

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            states = batch['states'].cuda()
            actions = batch['actions'].cuda()
            returns_to_go = batch['returns_to_go'].cuda()
            timesteps = batch['timesteps'].cuda()

            # Forward pass
            action_preds = model(returns_to_go, states, actions, timesteps)

            # Compute loss (cross-entropy for each action component)
            loss = 0
            for i, key in enumerate(['aircraft_id', 'command_type', 'altitude',
                                    'heading', 'speed']):
                loss += nn.functional.cross_entropy(
                    action_preds[key].reshape(-1, action_preds[key].shape[-1]),
                    actions[:, :, i].reshape(-1)
                )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        wandb.log({"loss": avg_loss, "epoch": epoch})

    # Save model
    torch.save(model.state_dict(), "dt_model.pth")
    wandb.finish()


if __name__ == "__main__":
    train_decision_transformer()
```

### Step 4: Demo Notebook (3-4 hours)

Create `notebooks/05_decision_transformer_demo.ipynb` demonstrating:
1. Data collection
2. Training
3. Return conditioning (RTG=50 vs RTG=200)
4. Sample efficiency comparison vs PPO

## Expected Results

- **Offline learning**: Train from 1000 episodes without environment interaction
- **Return conditioning**: Higher RTG → better performance
- **Sample efficiency**: 5-10x better than PPO (uses fixed dataset!)

## Commit Your Work

```bash
cd .trees/05-decision-transformer
git add .
git commit -m "Implement Decision Transformer for offline RL"
git push origin experiment/05-decision-transformer
```
