"""
Simple GPU monitoring for WSL2
Run this while training is happening
"""

import torch
import time
import os

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def format_bytes(bytes_val):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0

print("GPU Monitor for WSL2")
print("Press Ctrl+C to stop")
print("="*60)

try:
    while True:
        if not torch.cuda.is_available():
            print("CUDA not available!")
            break
        
        clear_screen()
        print("\n" + "="*60)
        print("GPU MONITOR - RTX 5090")
        print("="*60 + "\n")
        
        # GPU Info
        device = torch.cuda.current_device()
        print(f"Device: {torch.cuda.get_device_name(device)}")
        
        # Memory stats
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        
        print(f"\nMemory Usage:")
        print(f"  Allocated:     {format_bytes(allocated):>12} / {format_bytes(total)}")
        print(f"  Reserved:      {format_bytes(reserved):>12}")
        print(f"  Peak Allocated:{format_bytes(max_allocated):>12}")
        print(f"  Utilization:   {allocated/total*100:>11.1f}%")
        
        # Usage bar
        bar_width = 40
        used_bars = int((allocated / total) * bar_width)
        bar = "█" * used_bars + "░" * (bar_width - used_bars)
        print(f"\n  [{bar}] {allocated/total*100:.1f}%")
        
        print("\n" + "="*60)
        print("Updating every 2 seconds... (Ctrl+C to stop)")
        
        time.sleep(2)
        
except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")

