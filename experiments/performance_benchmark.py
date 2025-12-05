"""
Performance benchmarking utilities for OpenScope RL.

This module provides tools for measuring:
- Environment performance (FPS, throughput)
- Model inference speed
- Training throughput (samples/second)
"""

import time
import logging
from typing import Dict, Any, Optional
import numpy as np
import gymnasium as gym

logger = logging.getLogger(__name__)


def benchmark_env_performance(
    env: gym.Env,
    n_steps: int = 10000,
    warmup_steps: int = 100,
) -> Dict[str, float]:
    """
    Benchmark environment performance (FPS and throughput).
    
    Args:
        env: Environment to benchmark
        n_steps: Number of steps to run
        warmup_steps: Number of warmup steps before timing
        
    Returns:
        Dictionary with performance metrics:
        - fps: Frames per second
        - steps_per_second: Steps per second
        - avg_step_time: Average step time in seconds
        - total_time: Total time in seconds
    """
    logger.info(f"Benchmarking environment performance: {n_steps} steps")
    
    # Warmup
    obs, info = env.reset()
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()
    
    # Actual benchmark
    obs, info = env.reset()
    start_time = time.time()
    
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, info = env.reset()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    fps = n_steps / total_time
    steps_per_second = fps
    avg_step_time = total_time / n_steps
    
    results = {
        'fps': fps,
        'steps_per_second': steps_per_second,
        'avg_step_time': avg_step_time,
        'total_time': total_time,
        'n_steps': n_steps,
    }
    
    logger.info(f"Environment benchmark results: {fps:.2f} FPS, "
               f"{avg_step_time*1000:.2f}ms per step")
    
    return results


def benchmark_model_inference(
    model: Any,
    env: gym.Env,
    n_steps: int = 1000,
    warmup_steps: int = 50,
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model with predict() method
        env: Environment for inference
        n_steps: Number of inference steps
        warmup_steps: Number of warmup steps
        
    Returns:
        Dictionary with inference metrics:
        - inferences_per_second: Inferences per second
        - avg_inference_time: Average inference time in seconds
        - total_time: Total time in seconds
    """
    logger.info(f"Benchmarking model inference: {n_steps} steps")
    
    # Warmup
    obs, info = env.reset()
    for _ in range(warmup_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()
    
    # Actual benchmark
    obs, info = env.reset()
    start_time = time.time()
    
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            obs, info = env.reset()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    inferences_per_second = n_steps / total_time
    avg_inference_time = total_time / n_steps
    
    results = {
        'inferences_per_second': inferences_per_second,
        'avg_inference_time': avg_inference_time,
        'total_time': total_time,
        'n_steps': n_steps,
    }
    
    logger.info(f"Inference benchmark results: {inferences_per_second:.2f} inf/s, "
               f"{avg_inference_time*1000:.2f}ms per inference")
    
    return results


def benchmark_training_throughput(
    trainer: Any,
    n_steps: int = 10000,
) -> Dict[str, float]:
    """
    Benchmark training throughput (samples per second).
    
    Args:
        trainer: Trainer object with step() or similar method
        n_steps: Number of training steps
        
    Returns:
        Dictionary with training metrics:
        - samples_per_second: Training samples per second
        - avg_step_time: Average training step time
        - total_time: Total time in seconds
    """
    logger.info(f"Benchmarking training throughput: {n_steps} steps")
    
    start_time = time.time()
    
    # This is a template - actual implementation depends on trainer interface
    # For now, we'll measure generic step time
    for i in range(n_steps):
        # Trainer-specific training step
        if hasattr(trainer, 'step'):
            trainer.step()
        elif hasattr(trainer, 'train_step'):
            trainer.train_step()
        else:
            logger.warning("Trainer doesn't have step() or train_step() method")
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    samples_per_second = n_steps / total_time if total_time > 0 else 0
    avg_step_time = total_time / n_steps if n_steps > 0 else 0
    
    results = {
        'samples_per_second': samples_per_second,
        'avg_step_time': avg_step_time,
        'total_time': total_time,
        'n_steps': n_steps,
    }
    
    logger.info(f"Training benchmark results: {samples_per_second:.2f} samples/s, "
               f"{avg_step_time*1000:.2f}ms per step")
    
    return results


def run_all_benchmarks(
    env: gym.Env,
    model: Optional[Any] = None,
    trainer: Optional[Any] = None,
    n_steps: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Run all available benchmarks.
    
    Args:
        env: Environment to benchmark
        model: Optional model for inference benchmark
        trainer: Optional trainer for training benchmark
        n_steps: Number of steps for each benchmark
        
    Returns:
        Dictionary with all benchmark results
    """
    results = {}
    
    # Environment benchmark
    logger.info("Running environment benchmark...")
    results['environment'] = benchmark_env_performance(env, n_steps=n_steps)
    
    # Model inference benchmark
    if model is not None:
        logger.info("Running model inference benchmark...")
        results['inference'] = benchmark_model_inference(model, env, n_steps=n_steps)
    
    # Training benchmark
    if trainer is not None:
        logger.info("Running training throughput benchmark...")
        results['training'] = benchmark_training_throughput(trainer, n_steps=n_steps)
    
    return results


if __name__ == "__main__":
    """CLI entry point for benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark OpenScope RL performance")
    parser.add_argument("--n-steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps")
    parser.add_argument("--env-only", action="store_true", help="Only benchmark environment")
    
    args = parser.parse_args()
    
    # Example usage - would need actual env/model setup
    print("Performance benchmarking utilities")
    print("Use this module programmatically or import functions as needed")

