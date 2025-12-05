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
from typing import List, Tuple

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
        choices=["demo", "expert", "ppo", "benchmark", "all"],
        default=["all"],
        help="Which tests to run",
    )
    
    args = parser.parse_args()
    
    print_header("OpenScope RL Smoke Tests")
    
    # Check if OpenScope is running
    if not args.skip_openscope_check:
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

