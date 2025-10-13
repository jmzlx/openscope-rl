"""
Visualize training progress from logs
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict


def load_logs(log_file: Path) -> List[Dict]:
    """Load training logs from JSONL file"""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
    return logs


def plot_training_progress(logs: List[Dict], output_dir: Path):
    """Create comprehensive training progress plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    steps = [log.get('global_step', 0) for log in logs]
    
    # Rewards
    rewards = [log.get('mean_reward', 0) for log in logs if 'mean_reward' in log]
    reward_steps = [log.get('global_step', 0) for log in logs if 'mean_reward' in log]
    
    # Losses
    policy_losses = [log.get('policy_loss', 0) for log in logs if 'policy_loss' in log]
    value_losses = [log.get('value_loss', 0) for log in logs if 'value_loss' in log]
    entropy = [log.get('entropy', 0) for log in logs if 'entropy' in log]
    loss_steps = [log.get('global_step', 0) for log in logs if 'policy_loss' in log]
    
    # Episode stats
    ep_lengths = [log.get('mean_length', 0) for log in logs if 'mean_length' in log]
    ep_steps = [log.get('global_step', 0) for log in logs if 'mean_length' in log]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Mean Reward over time
    ax1 = fig.add_subplot(gs[0, :])
    if rewards:
        ax1.plot(reward_steps, rewards, linewidth=2, label='Mean Reward')
        # Add smoothed line
        if len(rewards) > 10:
            window = min(50, len(rewards) // 5)
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(reward_steps[window-1:], smoothed, linewidth=3, 
                    color='red', alpha=0.7, label=f'Smoothed (window={window})')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Training Reward Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Policy Loss
    ax2 = fig.add_subplot(gs[1, 0])
    if policy_losses:
        ax2.plot(loss_steps, policy_losses, linewidth=1, alpha=0.6)
        if len(policy_losses) > 10:
            window = min(50, len(policy_losses) // 5)
            smoothed = np.convolve(policy_losses, np.ones(window)/window, mode='valid')
            ax2.plot(loss_steps[window-1:], smoothed, linewidth=2, color='red')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Policy Loss')
    ax2.set_title('Policy Loss')
    ax2.grid(True, alpha=0.3)
    
    # 3. Value Loss
    ax3 = fig.add_subplot(gs[1, 1])
    if value_losses:
        ax3.plot(loss_steps, value_losses, linewidth=1, alpha=0.6)
        if len(value_losses) > 10:
            window = min(50, len(value_losses) // 5)
            smoothed = np.convolve(value_losses, np.ones(window)/window, mode='valid')
            ax3.plot(loss_steps[window-1:], smoothed, linewidth=2, color='red')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Value Loss')
    ax3.set_title('Value Loss')
    ax3.grid(True, alpha=0.3)
    
    # 4. Entropy
    ax4 = fig.add_subplot(gs[2, 0])
    if entropy:
        ax4.plot(loss_steps, entropy, linewidth=1, alpha=0.6)
        if len(entropy) > 10:
            window = min(50, len(entropy) // 5)
            smoothed = np.convolve(entropy, np.ones(window)/window, mode='valid')
            ax4.plot(loss_steps[window-1:], smoothed, linewidth=2, color='red')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Policy Entropy')
    ax4.grid(True, alpha=0.3)
    
    # 5. Episode Length
    ax5 = fig.add_subplot(gs[2, 1])
    if ep_lengths:
        ax5.plot(ep_steps, ep_lengths, linewidth=1, alpha=0.6)
        if len(ep_lengths) > 10:
            window = min(50, len(ep_lengths) // 5)
            smoothed = np.convolve(ep_lengths, np.ones(window)/window, mode='valid')
            ax5.plot(ep_steps[window-1:], smoothed, linewidth=2, color='red')
    ax5.set_xlabel('Training Steps')
    ax5.set_ylabel('Episode Length')
    ax5.set_title('Mean Episode Length')
    ax5.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_progress.png'}")
    plt.close()
    
    # Additional plot: reward histogram
    if rewards:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rewards, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(rewards), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        ax.set_xlabel('Mean Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution Across Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'reward_distribution.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'reward_distribution.png'}")
        plt.close()


def print_summary(logs: List[Dict]):
    """Print training summary"""
    if not logs:
        print("No logs found")
        return
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    last_log = logs[-1]
    print(f"Total Steps:     {last_log.get('global_step', 0):,}")
    print(f"Total Episodes:  {last_log.get('episode_count', 0):,}")
    
    # Get reward statistics
    rewards = [log.get('mean_reward', 0) for log in logs if 'mean_reward' in log]
    if rewards:
        print(f"\nReward Statistics:")
        print(f"  Final:         {rewards[-1]:.2f}")
        print(f"  Mean:          {np.mean(rewards):.2f}")
        print(f"  Std:           {np.std(rewards):.2f}")
        print(f"  Max:           {np.max(rewards):.2f}")
        print(f"  Min:           {np.min(rewards):.2f}")
    
    # Get loss statistics
    policy_losses = [log.get('policy_loss', 0) for log in logs if 'policy_loss' in log]
    if policy_losses:
        print(f"\nFinal Losses:")
        print(f"  Policy:        {policy_losses[-1]:.4f}")
        print(f"  Value:         {logs[-1].get('value_loss', 0):.4f}")
        print(f"  Entropy:       {logs[-1].get('entropy', 0):.4f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize training progress")
    parser.add_argument("--log-file", type=str, required=True,
                        help="Path to training log file (.jsonl)")
    parser.add_argument("--output-dir", type=str, default="training_plots",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    # Load logs
    print(f"Loading logs from: {args.log_file}")
    logs = load_logs(Path(args.log_file))
    print(f"Loaded {len(logs)} log entries")
    
    # Print summary
    print_summary(logs)
    
    # Create plots
    print(f"\nCreating plots...")
    plot_training_progress(logs, Path(args.output_dir))
    
    print("\nDone!")


if __name__ == "__main__":
    main()

