#!/usr/bin/env python3
"""
Launch multiple Temporal workers for parallel video processing
Each worker handles 1 activity at a time
"""

import subprocess
import sys
import time
import signal
import os
from typing import List

# Store worker processes
worker_processes: List[subprocess.Popen] = []

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shutdown all workers"""
    print("\n🛑 Shutting down all workers...")
    for process in worker_processes:
        process.terminate()
    
    # Wait for processes to terminate
    for process in worker_processes:
        process.wait()
    
    print("✅ All workers stopped")
    sys.exit(0)

def start_workers(num_workers: int = 1):
    """Start multiple Temporal workers"""
    print(f"🚀 Starting {num_workers} Temporal workers...")
    print(f"   Each worker processes 1 activity at a time")
    print(f"   Total parallel activities: {num_workers}")
    print("")
    
    for i in range(1, num_workers + 1):
        worker_id = f"gpu-worker-{i}"
        
        # Start worker process
        env = os.environ.copy()
        env["WORKER_ID"] = worker_id
        
        process = subprocess.Popen(
            ["python", "temporal_worker_gpu.py", worker_id],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        worker_processes.append(process)
        print(f"✅ Started worker {i}/{num_workers}: {worker_id} (PID: {process.pid})")
        
        # Small delay between worker starts to avoid race conditions
        time.sleep(0.5)
    
    print(f"\n📊 Worker Pool Summary:")
    print(f"   Workers: {num_workers}")
    print(f"   Max activities per worker: 1")
    print(f"   Total concurrent activities: {num_workers}")
    print(f"   Task queue: video-analysis-gpu-queue")
    print(f"\n   Press Ctrl+C to stop all workers...")
    
    # Monitor worker outputs
    try:
        while True:
            for i, process in enumerate(worker_processes):
                # Check if process is still running
                if process.poll() is not None:
                    print(f"⚠️ Worker {i+1} exited with code {process.returncode}")
                    # Optionally restart the worker
                    worker_id = f"gpu-worker-{i+1}"
                    env = os.environ.copy()
                    env["WORKER_ID"] = worker_id
                    
                    new_process = subprocess.Popen(
                        ["python", "temporal_worker_gpu.py", worker_id],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    worker_processes[i] = new_process
                    print(f"🔄 Restarted worker {i+1}: {worker_id} (PID: {new_process.pid})")
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get number of workers from command line or default to 3
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    
    # Limit workers based on available resources
    max_workers = 10  # Adjust based on your GPU memory
    if num_workers > max_workers:
        print(f"⚠️ Limiting to {max_workers} workers (max allowed)")
        num_workers = max_workers
    
    start_workers(num_workers)
