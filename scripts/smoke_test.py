#!/usr/bin/env python3
"""
Smoke test script for OpenScope RL entry points.

This script runs quick smoke tests on the main entry points to verify
they work correctly. OpenScope should be running on localhost:3003.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text.center(80)}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")


def print_test(name: str):
    """Print test name."""
    print(f"{YELLOW}Testing: {name}{RESET}")


def print_success(message: str):
    """Print success message."""
    print(f"{GREEN}✓ {message}{RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{RED}✗ {message}{RESET}")


def check_openscope_running() -> bool:
    """Check if OpenScope is running on localhost:3003."""
    try:
        import requests
        response = requests.get("http://localhost:3003", timeout=2)
        return response.status_code == 200
    except ImportError:
        # requests not available, try urllib instead
        try:
            from urllib.request import urlopen
            urlopen("http://localhost:3003", timeout=2)
            return True
        except Exception:
            return False
    except Exception:
        return False


def run_command(cmd: List[str], timeout: int = 60, use_uv: bool = True, env: dict = None) -> Tuple[bool, str, str]:
    """
    Run a command and return success status, stdout, and stderr.
    
    Args:
        cmd: Command to run as list of strings
        timeout: Timeout in seconds
        use_uv: Whether to use 'uv run' to execute the command
        env: Environment variables dict (will be merged with os.environ)
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        # Prepend 'uv run' if use_uv is True
        if use_uv:
            # Check if uv is available
            uv_check = subprocess.run(
                ["uv", "--version"],
                capture_output=True,
                timeout=5,
            )
            if uv_check.returncode == 0:
                cmd = ["uv", "run"] + cmd
            else:
                # uv not available, try without it
                pass
        
        # Merge environment variables
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,
            env=process_env,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def test_demo_env() -> bool:
    """Test the demo_env.py script."""
    print_test("demo_env.py")
    
    cmd = [
        "python",
        "scripts/demo_env.py",
        "--airport", "KLAS",
        "--max-aircraft", "5",
        "--n-steps", "20",
        "--headless",
    ]
    
    success, stdout, stderr = run_command(cmd, timeout=30, use_uv=True)
    
    if success:
        print_success("demo_env.py completed successfully")
        return True
    else:
        print_error(f"demo_env.py failed: {stderr}")
        return False


def test_rule_based_expert() -> bool:
    """Test the rule_based_expert.py script."""
    print_test("rule_based_expert.py")
    
    # Create test output directory
    test_dir = Path(__file__).parent.parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    cmd = [
        "python",
        "training/rule_based_expert.py",
        "--output", str(test_dir / "smoke_test_demos.pkl"),
        "--n-episodes", "1",
        "--max-aircraft", "3",
    ]
    
    success, stdout, stderr = run_command(cmd, timeout=180, use_uv=True)
    
    if success:
        print_success("rule_based_expert.py completed successfully")
        return True
    else:
        print_error(f"rule_based_expert.py failed: {stderr}")
        return False


def test_ppo_trainer_short() -> bool:
    """Test PPO trainer with minimal timesteps."""
    print_test("ppo_trainer.py (short run)")
    
    # Create test checkpoint directory
    test_dir = Path(__file__).parent.parent / "test_checkpoints"
    test_dir.mkdir(exist_ok=True)
    
    cmd = [
        "python",
        "training/ppo_trainer.py",
        "--total-timesteps", "50",
        "--n-envs", "1",
        "--max-aircraft", "2",
        "--save-dir", str(test_dir),
        "--eval-freq", "1000",  # Disable evaluation for smoke test
        "--checkpoint-freq", "1000",  # Disable checkpoints for smoke test
    ]
    
    # Set WANDB_MODE to disabled to avoid wandb initialization delays
    env_vars = {"WANDB_MODE": "disabled"}
    
    success, stdout, stderr = run_command(cmd, timeout=600, use_uv=True, env=env_vars)
    
    if success:
        print_success("ppo_trainer.py completed successfully")
        return True
    else:
        print_error(f"ppo_trainer.py failed: {stderr}")
        return False


def test_performance_benchmark() -> bool:
    """Test performance benchmark script."""
    print_test("performance_benchmark.py")
    
    cmd = [
        "python",
        "experiments/performance_benchmark.py",
        "--n-steps", "50",
        "--env-only",
    ]
    
    success, stdout, stderr = run_command(cmd, timeout=180, use_uv=True)
    
    if success:
        print_success("performance_benchmark.py completed successfully")
        return True
    else:
        print_error(f"performance_benchmark.py failed: {stderr}")
        return False


def test_tdmpc2_imports() -> bool:
    """Test that all TD-MPC 2 components can be imported."""
    print_test("TD-MPC 2 imports")
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
        print_success("All imports successful")
        return True
    except Exception as e:
        print_error(f"Import failed: {e}")
        return False


def test_tdmpc2_config_creation() -> tuple:
    """Test config creation. Returns (success, configs) tuple."""
    print_test("TD-MPC 2 config creation")
    try:
        from models.tdmpc2 import TDMPC2Config
        from training.tdmpc2_planner import MPCPlannerConfig
        from training.tdmpc2_trainer import TDMPC2TrainingConfig
        
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
        print_success("Configs created successfully")
        return True, (model_config, planner_config, training_config)
    except Exception as e:
        print_error(f"Config creation failed: {e}")
        return False, (None, None, None)


def test_tdmpc2_model_instantiation(model_config) -> tuple:
    """Test model instantiation. Returns (success, model, param_count) tuple."""
    print_test("TD-MPC 2 model instantiation")
    try:
        from models.tdmpc2 import TDMPC2Model
        
        model = TDMPC2Model(model_config)
        param_count = sum(p.numel() for p in model.parameters())
        print_success(f"Model created with {param_count:,} parameters")
        return True, model, param_count
    except Exception as e:
        print_error(f"Model creation failed: {e}")
        return False, None, 0


def test_tdmpc2_encoder_forward_pass(model, model_config) -> tuple:
    """Test encoder forward pass. Returns (success, latent) tuple."""
    print_test("TD-MPC 2 encoder forward pass")
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
        print_success(f"Encoder forward pass successful: {latent.shape}")
        return True, latent
    except Exception as e:
        print_error(f"Encoder forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_tdmpc2_dynamics_reward_forward_pass(model, model_config, latent, device) -> bool:
    """Test dynamics, reward, and Q-network forward passes."""
    print_test("TD-MPC 2 dynamics/reward forward pass")
    try:
        batch_size = latent.size(0)
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
        
        print_success("Dynamics, reward, and Q-network forward passes successful")
        return True
    except Exception as e:
        print_error(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tdmpc2_mpc_planner(model, planner_config, model_config, device) -> bool:
    """Test MPC planner."""
    print_test("TD-MPC 2 MPC planner")
    try:
        from training.tdmpc2_planner import MPCPlanner
        
        planner = MPCPlanner(model, planner_config)
        
        # Create test inputs
        batch_size = 2
        max_aircraft = model_config.max_aircraft
        aircraft = torch.randn(batch_size, max_aircraft, model_config.aircraft_feature_dim).to(device)
        aircraft_mask = torch.ones(batch_size, max_aircraft, dtype=torch.bool).to(device)
        global_state = torch.randn(batch_size, model_config.global_feature_dim).to(device)
        
        # Test planning
        with torch.no_grad():
            planned_action = planner.plan(aircraft, aircraft_mask, global_state)
        
        assert planned_action.shape == (batch_size, model_config.action_dim), \
            f"Expected action shape ({batch_size}, {model_config.action_dim}), got {planned_action.shape}"
        print_success(f"MPC planner successful: {planned_action.shape}")
        return True
    except Exception as e:
        print_error(f"MPC planner failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tdmpc2_replay_buffer(model_config) -> bool:
    """Test replay buffer."""
    print_test("TD-MPC 2 replay buffer")
    try:
        from training.tdmpc2_trainer import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=100, device="cpu")
        max_aircraft = model_config.max_aircraft
        
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
        print_success(f"Replay buffer successful: {len(buffer)} transitions, sampled batch size {batch['aircraft'].shape[0]}")
        return True
    except Exception as e:
        print_error(f"Replay buffer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tdmpc2_trainer_initialization(training_config, model_config) -> bool:
    """Test trainer initialization with mock environment."""
    print_test("TD-MPC 2 trainer initialization (mock)")
    try:
        from training.tdmpc2_trainer import TDMPC2Trainer
        
        # Create a simple mock environment for testing
        class MockEnv:
            def reset(self):
                return {
                    "aircraft": np.random.randn(model_config.max_aircraft, model_config.aircraft_feature_dim).astype(np.float32),
                    "aircraft_mask": np.ones(model_config.max_aircraft, dtype=bool),
                    "global_state": np.random.randn(model_config.global_feature_dim).astype(np.float32),
                }, {}
            
            def step(self, action):
                return {
                    "aircraft": np.random.randn(model_config.max_aircraft, model_config.aircraft_feature_dim).astype(np.float32),
                    "aircraft_mask": np.ones(model_config.max_aircraft, dtype=bool),
                    "global_state": np.random.randn(model_config.global_feature_dim).astype(np.float32),
                }, 0.0, False, False, {}
        
        mock_env = MockEnv()
        trainer = TDMPC2Trainer(mock_env, training_config)
        print_success("Trainer initialization successful")
        print(f"   - Training step: {trainer.training_step}")
        print(f"   - Environment step: {trainer.env_step}")
        print(f"   - Replay buffer capacity: {trainer.replay_buffer.capacity}")
        return True
    except Exception as e:
        print_error(f"Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tdmpc2_implementation() -> bool:
    """Test TD-MPC 2 implementation components."""
    print_test("TD-MPC 2 implementation")
    
    # Test 1: Imports
    if not test_tdmpc2_imports():
        print_error("Import test failed - cannot continue")
        return False
    
    # Import after test_imports succeeds
    from models.tdmpc2 import TDMPC2Model, TDMPC2Config
    from training.tdmpc2_planner import MPCPlannerConfig
    from training.tdmpc2_trainer import TDMPC2TrainingConfig
    
    # Test 2: Config creation
    success, (model_config, planner_config, training_config) = test_tdmpc2_config_creation()
    if not success:
        return False
    
    # Test 3: Model instantiation
    success, model, param_count = test_tdmpc2_model_instantiation(model_config)
    if not success:
        return False
    
    # Test 4: Encoder forward pass
    success, latent = test_tdmpc2_encoder_forward_pass(model, model_config)
    if not success:
        return False
    
    # Test 5: Dynamics and reward forward pass
    device = model_config.device
    success = test_tdmpc2_dynamics_reward_forward_pass(model, model_config, latent, device)
    if not success:
        return False
    
    # Test 6: MPC Planner
    success = test_tdmpc2_mpc_planner(model, planner_config, model_config, device)
    if not success:
        return False
    
    # Test 7: Replay Buffer
    success = test_tdmpc2_replay_buffer(model_config)
    if not success:
        return False
    
    # Test 8: Trainer initialization
    success = test_tdmpc2_trainer_initialization(training_config, model_config)
    if not success:
        return False
    
    print_success("All TD-MPC 2 implementation tests passed")
    return True


def main():
    """Run smoke tests."""
    parser = argparse.ArgumentParser(description="Run smoke tests for OpenScope RL")
    parser.add_argument(
        "--skip-openscope-check",
        action="store_true",
        help="Skip OpenScope server check",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["demo", "expert", "ppo", "benchmark", "tdmpc2", "all"],
        default=["all"],
        help="Which tests to run",
    )
    
    args = parser.parse_args()
    
    print_header("OpenScope RL Smoke Tests")
    
    # Check if OpenScope is running (skip for TD-MPC 2 test which doesn't need it)
    skip_openscope = args.skip_openscope_check or ("tdmpc2" in args.tests and len(args.tests) == 1)
    if not skip_openscope:
        print_test("OpenScope server connection")
        if check_openscope_running():
            print_success("OpenScope is running on localhost:3003")
        else:
            print_error("OpenScope is not running on localhost:3003")
            print("Please start OpenScope: cd ../openscope && npm start")
            return 1
    
    # Determine which tests to run
    run_all = "all" in args.tests
    tests_to_run = []
    
    if run_all or "demo" in args.tests:
        tests_to_run.append(("Environment Demo", test_demo_env))
    if run_all or "expert" in args.tests:
        tests_to_run.append(("Rule-Based Expert", test_rule_based_expert))
    if run_all or "ppo" in args.tests:
        tests_to_run.append(("PPO Trainer", test_ppo_trainer_short))
    if run_all or "benchmark" in args.tests:
        tests_to_run.append(("Performance Benchmark", test_performance_benchmark))
    if run_all or "tdmpc2" in args.tests:
        tests_to_run.append(("TD-MPC 2 Implementation", test_tdmpc2_implementation))
    
    # Run tests
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests_to_run:
        print(f"\n{'-' * 80}")
        result = test_func()
        results.append((test_name, result))
        time.sleep(1)  # Brief pause between tests
    
    # Print summary
    elapsed = time.time() - start_time
    print_header("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Time: {elapsed:.2f}s")
    
    if passed == total:
        print_success("\nAll smoke tests passed! ✓")
        return 0
    else:
        print_error(f"\n{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    import os
    sys.exit(main())

