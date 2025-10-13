"""
Logging utilities for training
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
from datetime import datetime


class Logger:
    """Simple logger for training metrics"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.metrics = []
    
    def log(self, metrics: Dict[str, Any]):
        """Log metrics"""
        # Convert numpy types to python types
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                clean_metrics[key] = value.item()
            elif isinstance(value, (list, np.ndarray)):
                clean_metrics[key] = [float(v) for v in value]
            else:
                clean_metrics[key] = value
        
        # Add timestamp
        clean_metrics['timestamp'] = datetime.now().isoformat()
        
        # Append to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(clean_metrics) + '\n')
        
        self.metrics.append(clean_metrics)
    
    def get_metrics(self) -> list:
        """Get all logged metrics"""
        return self.metrics
    
    def save_summary(self):
        """Save summary statistics"""
        if not self.metrics:
            return
        
        summary = {
            'total_steps': self.metrics[-1].get('global_step', 0),
            'total_episodes': self.metrics[-1].get('episode_count', 0),
            'final_mean_reward': self.metrics[-1].get('mean_reward', 0),
        }
        
        summary_file = self.log_dir / 'training_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

