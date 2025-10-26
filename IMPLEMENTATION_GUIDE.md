# NVIDIA Cosmos Integration - Implementation Guide

**Worktree**: `.trees/07-cosmos-world-model/`
**Branch**: `experiment/07-cosmos-world-model`
**Priority**: ⭐⭐⭐ GAME-CHANGER (Revolutionary approach!)

## Objective

Fine-tune NVIDIA Cosmos World Foundation Model on OpenScope gameplay to create a learned simulator. Train RL policies in this fast simulated environment (10-100x faster than real OpenScope!).

## Why This Is Revolutionary

**NVIDIA Cosmos** (released January 2025):
- World Foundation Models trained on 20M hours of video
- Physics-aware video generation (predicts future frames)
- **Perfect for RL**: WFM + reward model = fast simulator!

**Your Hardware**: 2x DGX + RTX 5090 = ideal for Cosmos!

**Expected Benefits**:
- 10-100x sample efficiency (train in simulation, not real OpenScope)
- Unlimited scenario generation
- Safe exploration (test dangerous situations without risk)
- Parallel training (100s of simulated environments on GPUs)

## Implementation Steps

### Step 0: Install Cosmos (1 hour)

```bash
# Install NVIDIA Cosmos
pip install nvidia-cosmos

# Download pre-trained model (choose one):
# Option A: Cosmos Super (7B params, highest quality)
cosmos download nvidia/cosmos-super-7b

# Option B: Cosmos Nano (2B params, faster iteration)
cosmos download nvidia/cosmos-nano-2b  # Recommended for starting!

# Verify installation
python -c "from nvidia_cosmos import CosmosWFM; print('Cosmos installed!')"

# Check GPU
nvidia-smi  # Should show your DGX or RTX 5090
```

### Step 1: Data Collection (4-6 hours)

Create `data/cosmos_collector.py`:

```python
"""Collect OpenScope gameplay data for Cosmos training."""

import time
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from environment import PlaywrightEnv, create_default_config
from environment.utils import BrowserManager


class CosmosDataCollector:
    """
    Collect OpenScope episodes with video + state data.

    Captures:
    - Video frames (browser screenshots at 1 FPS)
    - Game states (aircraft positions, conflicts, etc.)
    - Actions taken (command issued)
    - Rewards received
    """

    def __init__(self, save_dir="cosmos_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def collect_episode(self, env, episode_id, policy='random'):
        """
        Collect single episode with video.

        Returns:
            Episode data dictionary
        """
        frames = []
        states = []
        actions = []
        rewards = []

        obs, info = env.reset()
        done = False
        step = 0

        while not done:
            # Capture screenshot from browser
            frame = self._capture_frame(env)
            frames.append(frame)

            # Store game state
            states.append(info['raw_state'])

            # Get action
            if policy == 'random':
                action = env.action_space.sample()
            elif policy == 'heuristic':
                action = self._heuristic_policy(obs, info)

            actions.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            done = terminated or truncated
            step += 1

            # Limit episode length for data collection
            if step >= 300:
                break

            # Capture at 1 FPS (sleep to avoid capturing too fast)
            time.sleep(1.0)

        # Save episode
        episode_data = {
            'frames': frames,
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'episode_return': sum(rewards),
        }

        # Save video
        self._save_video(frames, episode_id)

        # Save metadata
        np.save(self.save_dir / f"episode_{episode_id}.npy", episode_data)

        return episode_data

    def _capture_frame(self, env):
        """Capture screenshot from browser."""
        # Get page screenshot as bytes
        screenshot_bytes = env.browser_manager.page.screenshot()

        # Convert to PIL Image
        image = Image.frombytes('RGB', screenshot_bytes)

        # Resize to standard size (e.g., 512x512)
        image = image.resize((512, 512))

        return np.array(image)

    def _save_video(self, frames, episode_id):
        """Save frames as MP4 video."""
        video_path = self.save_dir / f"episode_{episode_id}.mp4"

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 1  # 1 FPS (we capture at 1 frame per second)
        height, width = frames[0].shape[:2]

        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()

    def _heuristic_policy(self, obs, info):
        """Simple rule-based policy for better data."""
        # TODO: Implement heuristics
        # - If conflict predicted, issue heading change
        # - If near exit, continue straight
        # - If near runway, issue ILS
        return env.action_space.sample()  # Placeholder

    def collect_dataset(self, num_episodes=100):
        """Collect full dataset."""
        config = create_default_config(headless=False)  # Visible for screenshots
        env = PlaywrightEnv(**config.__dict__)

        episodes = []

        for i in range(num_episodes):
            print(f"Collecting episode {i+1}/{num_episodes}...")

            # Mix of policies for diversity
            policy = 'random' if i % 2 == 0 else 'heuristic'

            episode = self.collect_episode(env, episode_id=i, policy=policy)
            episodes.append(episode)

            print(f"  Episode {i+1}: {len(episode['frames'])} frames, "
                  f"return = {episode['episode_return']:.2f}")

        env.close()

        print(f"\nDataset collection complete!")
        print(f"  Total episodes: {len(episodes)}")
        print(f"  Average frames: {np.mean([len(ep['frames']) for ep in episodes]):.0f}")
        print(f"  Average return: {np.mean([ep['episode_return'] for ep in episodes]):.2f}")
        print(f"  Saved to: {self.save_dir}")

        return episodes


if __name__ == "__main__":
    collector = CosmosDataCollector()
    collector.collect_dataset(num_episodes=100)
```

**Data Collection Plan**:
- 100 episodes total
- Mix of random (50%) and heuristic (50%) policies
- Capture at 1 FPS (enough for ATC, reduces data size)
- ~10-20 hours of gameplay data
- Expected storage: ~10-20 GB

### Step 2: Cosmos Fine-Tuning (6-12 hours on 2x DGX)

Create `training/cosmos_finetuner.py`:

```python
"""Fine-tune NVIDIA Cosmos on OpenScope data."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np

# Cosmos imports (adjust based on actual Cosmos API)
from nvidia_cosmos import CosmosWFM, CosmosTrainer, CosmosConfig


class OpenScopeVideoDataset(Dataset):
    """Dataset of OpenScope videos for Cosmos training."""

    def __init__(self, data_dir="cosmos_data"):
        self.data_dir = Path(data_dir)
        self.episodes = sorted(self.data_dir.glob("episode_*.mp4"))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        # Load video
        video_path = self.episodes[idx]
        frames = self._load_video(video_path)

        # Load actions (for conditioning)
        metadata_path = video_path.with_suffix('.npy')
        metadata = np.load(metadata_path, allow_pickle=True).item()
        actions = metadata['actions']

        return {
            'frames': torch.FloatTensor(frames),
            'actions': actions,  # For action-conditioned generation
        }

    def _load_video(self, video_path):
        """Load video as tensor of frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

        cap.release()

        return np.array(frames)


class OpenScopeCosmosTrainer:
    """Fine-tune Cosmos on OpenScope gameplay."""

    def __init__(self, model_name="nvidia/cosmos-nano-2b"):
        # Load pre-trained Cosmos model
        print(f"Loading Cosmos model: {model_name}")
        self.model = CosmosWFM.from_pretrained(model_name)

        # Move to GPUs (DDP for multi-GPU)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model)

        self.model = self.model.cuda()

    def prepare_data(self, data_dir="cosmos_data"):
        """Prepare dataset and dataloader."""
        dataset = OpenScopeVideoDataset(data_dir)

        # Split into train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,  # Adjust based on GPU memory
            shuffle=True,
            num_workers=4,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
        )

        return train_loader, val_loader

    def train(self, epochs=10, lr=1e-5):
        """Fine-tune Cosmos model."""
        train_loader, val_loader = self.prepare_data()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for batch in train_loader:
                frames = batch['frames'].cuda()
                actions = batch['actions']  # TODO: Encode actions

                # Forward pass
                # Note: Actual Cosmos API may differ - check documentation
                loss = self.model(frames, actions)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    frames = batch['frames'].cuda()
                    actions = batch['actions']

                    loss = self.model(frames, actions)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}")

        # Save fine-tuned model
        self.model.save_pretrained("cosmos-openscope-finetuned")
        print("Fine-tuned model saved!")


if __name__ == "__main__":
    trainer = OpenScopeCosmosTrainer()
    trainer.train(epochs=10)
```

**Training Configuration**:
- Model: cosmos-nano-2b (2B params, faster)
- Batch size: 4-8 (DGX dependent)
- Epochs: 10-20
- Learning rate: 1e-5
- Expected time: 6-12 hours on 2x DGX
- Checkpointing: Every epoch

### Step 3: Create Cosmos-Based Environment (4-6 hours)

Create `environment/cosmos_env.py`:

```python
"""OpenScope environment simulated by Cosmos."""

import gymnasium as gym
import numpy as np
import torch

from nvidia_cosmos import CosmosWFM


class CosmosOpenScopeEnv(gym.Env):
    """
    OpenScope environment using Cosmos as world model.

    This is MUCH faster than real OpenScope (no browser, no JavaScript!).
    """

    def __init__(self, cosmos_model_path="cosmos-openscope-finetuned"):
        super().__init__()

        # Load fine-tuned Cosmos model
        self.cosmos_model = CosmosWFM.from_pretrained(cosmos_model_path).cuda()
        self.cosmos_model.eval()

        # Reward model (trained separately on collected episodes)
        self.reward_model = self._load_reward_model()

        # Current frame
        self.current_frame = None

        # Define spaces (same as PlaywrightEnv)
        self.observation_space = ...  # TODO: Define
        self.action_space = ...  # TODO: Define

    def _load_reward_model(self):
        """Load reward prediction model."""
        # Simple MLP: state → reward
        # Trained on collected episodes
        # TODO: Implement
        return None

    def reset(self, seed=None, options=None):
        """Reset to initial state."""
        super().reset(seed=seed)

        # Generate initial frame (or use random frame from dataset)
        self.current_frame = self._get_initial_frame()

        # Extract state from frame
        obs = self._extract_state(self.current_frame)

        return obs, {}

    def step(self, action):
        """Take action in Cosmos-simulated environment."""

        # Generate next frame using Cosmos
        next_frame = self._predict_next_frame(self.current_frame, action)

        # Extract state from generated frame
        next_state = self._extract_state(next_frame)

        # Compute reward using reward model
        reward = self.reward_model(next_state, action) if self.reward_model else 0.0

        # Check termination
        terminated = self._check_done(next_state)

        # Update current frame
        self.current_frame = next_frame

        return next_state, reward, terminated, False, {}

    def _predict_next_frame(self, current_frame, action):
        """Use Cosmos to predict next frame."""
        with torch.no_grad():
            # Encode action (TODO: depends on Cosmos API)
            action_embedding = self._encode_action(action)

            # Predict next frame
            next_frame = self.cosmos_model.generate(
                current_frame,
                action=action_embedding,
            )

        return next_frame

    def _extract_state(self, frame):
        """Extract game state from frame."""
        # Option A: Use vision model to detect aircraft positions
        # Option B: Use OCR to read game UI
        # Option C: Maintain shadow state (less accurate but faster)

        # TODO: Implement
        # For now, return dummy state
        return np.zeros((10, 14), dtype=np.float32)

    def _encode_action(self, action):
        """Encode action for Cosmos."""
        # Convert multi-discrete action to embedding or text
        # TODO: Implement
        return None

    def _get_initial_frame(self):
        """Get initial frame for episode start."""
        # Could be random frame from training data
        # TODO: Implement
        return np.zeros((512, 512, 3), dtype=np.float32)

    def _check_done(self, state):
        """Check if episode is done."""
        # TODO: Implement
        return False
```

### Step 4: Demo Notebook (4-6 hours)

Create `notebooks/07_cosmos_world_model_demo.ipynb` with sections:

1. **Data Collection** (run data collector)
2. **Cosmos Fine-Tuning** (may need to run separately on DGX)
3. **World Model Evaluation** (visual comparison: real vs predicted)
4. **RL Training in Cosmos** (train PPO, 10M steps in ~1-2 hours!)
5. **Transfer to Real OpenScope** (test learned policy)
6. **Sample Efficiency Comparison** (Cosmos vs baseline)

### Step 5: Known Challenges & Solutions

**Challenge 1: Action Encoding**
- **Problem**: How to tell Cosmos what action was taken?
- **Solutions**:
  - Text prompts: "Aircraft N123 turn to heading 270"
  - Structured embedding: Learn action encoder
  - Visual overlay: Draw action on frame

**Challenge 2: State Extraction**
- **Problem**: How to get aircraft positions from generated frames?
- **Solutions**:
  - Vision model (YOLO/SAM to detect aircraft)
  - OCR (read UI elements)
  - Shadow state (maintain approximate state, less accurate)

**Challenge 3: Reward Model**
- **Problem**: Can't compute reward without real game state
- **Solution**: Train reward model on collected episodes
  - Input: (state, action)
  - Output: reward
  - Simple MLP, train with supervised learning

## Expected Results

If successful:
- ✅ Cosmos learns OpenScope dynamics (visual similarity)
- ✅ RL policy trains 10-100x faster in Cosmos
- ✅ Policy transfers to real OpenScope with 60-80% of performance
- ✅ Total time: Data (10h) + Fine-tune (12h) + RL (2h) = ~24 hours

If partially successful:
- ✅ Cosmos generates plausible-looking frames
- ⚠️ Policy doesn't transfer perfectly (sim-to-real gap)
- ✅ Still valuable: Shows potential of world models for ATC

## Fallback Plan

If Cosmos API is complex or unavailable:
1. **Skip Cosmos**, focus on baseline PPO and Decision Transformer
2. **Or**: Use simpler world model (e.g., train your own video prediction model)
3. **Or**: Use Cosmos for synthetic data generation only (not full RL training)

## Commit Your Work

```bash
cd .trees/07-cosmos-world-model
git add .
git commit -m "Implement Cosmos world model integration"
git push origin experiment/07-cosmos-world-model
```

---

**This is cutting-edge research!** Cosmos was released in January 2025. Even partial success would be publishable. Don't be discouraged if it's challenging - that's what makes it exciting!
