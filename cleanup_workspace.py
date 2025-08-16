#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files and folders
Keeps only essential components for the video analysis system
"""

import os
import shutil
import glob

def cleanup():
    """Remove unnecessary files and folders"""
    
    # Files to remove
    files_to_remove = [
        "=1.0.0",  # Error file
        "main.log",
        "temporal.log", 
        "temporal_server.log",
        "example_usage.py",
        "simple_temporal_worker.py",
        "start_temporal_worker.py",
        "intelligent_video_processor.py",
        "download_qwen.py",
        "qwen25_setup.py",
        "test_temporal_integration.py",
        "curl_examples.sh",
        "deploy_video_api.sh",
        "setup.sh",
        "Dockerfile",
        "Dockerfile.api",
        "docker-compose.yml",
        "docker-compose-api.yml",
        "temporal"  # Binary can be redownloaded
    ]
    
    # Folders to remove
    folders_to_remove = [
        "__pycache__",
        "venv",
        "temp_frames",
        "session_data"
    ]
    
    print("🧹 Starting workspace cleanup...")
    print("-" * 50)
    
    # Remove files
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ Removed file: {file}")
            except Exception as e:
                print(f"❌ Failed to remove {file}: {e}")
    
    # Remove folders
    for folder in folders_to_remove:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"✅ Removed folder: {folder}/")
            except Exception as e:
                print(f"❌ Failed to remove {folder}: {e}")
    
    # Clean up any .pyc files
    pyc_files = glob.glob("**/*.pyc", recursive=True)
    for pyc in pyc_files:
        try:
            os.remove(pyc)
            print(f"✅ Removed: {pyc}")
        except:
            pass
    
    print("-" * 50)
    print("✨ Cleanup complete!")
    
    # List remaining files
    print("\n📁 Remaining essential files:")
    essential_files = [
        "temporal_worker_gpu.py",
        "video_api_json.py", 
        "model_manager.py",
        "start_multiple_workers.py",
        "requirements.txt",
        "api_requirements.txt",
        ".env",
        "README.md",
        "LICENSE"
    ]
    
    for file in essential_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✓ {file} ({size:,} bytes)")
    
    # Check models folder
    if os.path.exists("models"):
        model_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk("models")
                        for filename in filenames)
        print(f"   ✓ models/ ({model_size / (1024**3):.2f} GB)")

if __name__ == "__main__":
    response = input("⚠️  This will remove many files. Continue? (y/n): ")
    if response.lower() == 'y':
        cleanup()
    else:
        print("Cleanup cancelled.")
