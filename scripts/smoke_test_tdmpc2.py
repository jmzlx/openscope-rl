#!/usr/bin/env python3
"""
Smoke test for TD-MPC 2 implementation.

This script performs basic validation to ensure:
- All components can be imported
- Models can be instantiated
- Forward passes work with dummy data
- Planner can be created and used
- Trainer can be initialized

Run with: python scripts/smoke_test_tdmpc2.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

print("=" * 60)
print("TD-MPC 2 Smoke Test")
print("=" * 60)

# Test 1: Imports
print("\n[1/7] Testing imports...")
try:
    from models.tdmpc2 import (
        TDMPC2Model,
        TDMPC2Config,
        LatentEncoder,
        TransformerDynamics,
        RewardPredictor,
        TDMPC2QNetwork,
    )
    from training.tdmpc2_planner import MPCPlanner, MPCPlannerConfig
    from training.tdmpc2_trainer import TDMPC2Trainer, TDMPC2TrainingConfig, ReplayBuffer
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Config creation
print("\n[2/7] Testing config creation...")
try:
    model_config = TDMPC2Config(
        max_aircraft=10,
        latent_dim=256,  # Smaller for smoke test
        encoder_hidden_dim=128,
        dynamics_hidden_dim=256,
        device="cpu",  # Force CPU for smoke test to avoid MPS issues
    )
    planner_config = MPCPlannerConfig(
        planning_horizon=3,  # Shorter for smoke test
        num_samples=32,  # Fewer samples for speed
        num_elites=8,
        num_iterations=2,
        device="cpu",  # Force CPU for smoke test
    )
    training_config = TDMPC2TrainingConfig(
        model_config=model_config,
        planner_config=planner_config,
        num_steps=100,  # Small for smoke test
        batch_size=4,
    )
    print("✅ Configs created successfully")
except Exception as e:
    print(f"❌ Config creation failed: {e}")
    sys.exit(1)

# Test 3: Model instantiation
print("\n[3/7] Testing model instantiation...")
try:
    model = TDMPC2Model(model_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {param_count:,} parameters")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    sys.exit(1)

# Test 4: Encoder forward pass
print("\n[4/7] Testing encoder forward pass...")
try:
    device = model_config.device
    batch_size = 2
    max_aircraft = model_config.max_aircraft
    aircraft = torch.randn(batch_size, max_aircraft, model_config.aircraft_feature_dim).to(device)
    aircraft_mask = torch.ones(batch_size, max_aircraft, dtype=torch.bool).to(device)
    global_state = torch.randn(batch_size, model_config.global_feature_dim).to(device)
    
    latent = model.encode(aircraft, aircraft_mask, global_state)
    assert latent.shape == (batch_size, model_config.latent_dim), \
        f"Expected latent shape ({batch_size}, {model_config.latent_dim}), got {latent.shape}"
    print(f"✅ Encoder forward pass successful: {latent.shape}")
except Exception as e:
    print(f"❌ Encoder forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Dynamics and reward forward pass
print("\n[5/7] Testing dynamics and reward forward pass...")
try:
    action = torch.randn(batch_size, model_config.action_dim).to(device)
    
    next_latent = model.dynamics(latent, action)
    assert next_latent.shape == (batch_size, model_config.latent_dim), \
        f"Expected next_latent shape ({batch_size}, {model_config.latent_dim}), got {next_latent.shape}"
    
    reward = model.reward(next_latent)
    assert reward.shape == (batch_size, 1), \
        f"Expected reward shape ({batch_size}, 1), got {reward.shape}"
    
    q_value = model.q_network(latent, action)
    assert q_value.shape == (batch_size, 1), \
        f"Expected q_value shape ({batch_size}, 1), got {q_value.shape}"
    
    print(f"✅ Dynamics, reward, and Q-network forward passes successful")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: MPC Planner
print("\n[6/7] Testing MPC planner...")
try:
    planner = MPCPlanner(model, planner_config)
    
    # Test planning
    with torch.no_grad():
        planned_action = planner.plan(aircraft, aircraft_mask, global_state)
    
    assert planned_action.shape == (batch_size, model_config.action_dim), \
        f"Expected action shape ({batch_size}, {model_config.action_dim}), got {planned_action.shape}"
    print(f"✅ MPC planner successful: {planned_action.shape}")
except Exception as e:
    print(f"❌ MPC planner failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Replay Buffer
print("\n[7/7] Testing replay buffer...")
try:
    buffer = ReplayBuffer(capacity=100, device="cpu")
    
    # Add some transitions
    for i in range(5):
        obs = {
            "aircraft": np.random.randn(max_aircraft, model_config.aircraft_feature_dim).astype(np.float32),
            "aircraft_mask": np.ones(max_aircraft, dtype=bool),
            "global_state": np.random.randn(model_config.global_feature_dim).astype(np.float32),
        }
        action = np.random.randn(model_config.action_dim).astype(np.float32)
        next_obs = {
            "aircraft": np.random.randn(max_aircraft, model_config.aircraft_feature_dim).astype(np.float32),
            "aircraft_mask": np.ones(max_aircraft, dtype=bool),
            "global_state": np.random.randn(model_config.global_feature_dim).astype(np.float32),
        }
        buffer.add(obs, action, 0.5, next_obs, False)
    
    # Sample batch
    batch = buffer.sample(3)
    assert "aircraft" in batch
    assert batch["aircraft"].shape[0] == 3
    print(f"✅ Replay buffer successful: {len(buffer)} transitions, sampled batch size {batch['aircraft'].shape[0]}")
except Exception as e:
    print(f"❌ Replay buffer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
# Test 8: Trainer initialization (without environment)
print("\n[8/8] Testing trainer initialization (mock)...")
try:
    # Create a simple mock environment for testing
    class MockEnv:
        def reset(self):
            return {
                "aircraft": np.random.randn(max_aircraft, model_config.aircraft_feature_dim).astype(np.float32),
                "aircraft_mask": np.ones(max_aircraft, dtype=bool),
                "global_state": np.random.randn(model_config.global_feature_dim).astype(np.float32),
            }, {}
        
        def step(self, action):
            return {
                "aircraft": np.random.randn(max_aircraft, model_config.aircraft_feature_dim).astype(np.float32),
                "aircraft_mask": np.ones(max_aircraft, dtype=bool),
                "global_state": np.random.randn(model_config.global_feature_dim).astype(np.float32),
            }, 0.0, False, False, {}
    
    mock_env = MockEnv()
    trainer = TDMPC2Trainer(mock_env, training_config)
    print(f"✅ Trainer initialization successful")
    print(f"   - Training step: {trainer.training_step}")
    print(f"   - Environment step: {trainer.env_step}")
    print(f"   - Replay buffer capacity: {trainer.replay_buffer.capacity}")
except Exception as e:
    print(f"❌ Trainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL SMOKE TESTS PASSED")
print("=" * 60)
print("\nSummary:")
print(f"  - Model parameters: {param_count:,}")
print(f"  - Latent dimension: {model_config.latent_dim}")
print(f"  - Planning horizon: {planner_config.planning_horizon}")
print(f"  - Replay buffer: {len(buffer)} transitions")
print(f"  - Trainer initialized successfully")
print("\nThe TD-MPC 2 implementation is ready for use!")

