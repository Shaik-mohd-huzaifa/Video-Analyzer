#!/usr/bin/env python3
"""
GPU-Enabled Temporal Worker for Video Analysis
Integrates real model calls with GPU memory management
"""

import asyncio
import logging
from typing import Dict, List, Any
import os
import gc
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import only Temporal modules at top level to avoid sandbox issues
try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.worker import Worker
    from temporalio.common import RetryPolicy
    from datetime import timedelta
    import json
    
    # Note: ModelOrchestrator imports deferred to activities to avoid sandbox restrictions
    
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    logger.error("❌ Temporal packages not installed")
    exit(1)


# Global model cache (will be initialized in activities)
_model_cache = {}


@activity.defn
async def analyze_frame_activity(task_data: dict) -> dict:
    """Frame analysis activity with real GPU model inference"""
    import torch
    import time
    from PIL import Image
    import io
    import base64
    
    # Import heavy modules only when activity runs
    from model_manager import ModelOrchestrator
    
    global _model_cache
    
    try:
        # Extract task parameters
        frame_index = task_data.get('frame_index', 0)
        frame_path = task_data.get('frame_path', '')
        prompt = task_data.get('prompt', 'Describe what you see in this image')
        session_id = task_data.get('session_id', 'default')
        
        # Use cached orchestrator or create new one
        if 'orchestrator' not in _model_cache:
            # Import at runtime to avoid sandbox restrictions
            from model_manager import ModelOrchestrator, ModelConfig, ModelType
            
            # Set GPU environment
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            orchestrator = ModelOrchestrator()
            
            # Load local Qwen model for vision tasks
            model_config = ModelConfig(
                name='qwen_vision_local',
                type=ModelType.QWEN_VISION,
                model_id='models/Qwen_Qwen2.5-VL-7B-Instruct',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                max_memory='8GB',
                precision='float16'
            )
            
            # Check GPU availability
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_properties(0)
                logger.info(f"🎮 GPU detected: {gpu_info.name}")
                logger.info(f"   Total memory: {gpu_info.total_memory / 1024**3:.2f} GB")
                logger.info(f"   Current allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            else:
                logger.warning("⚠️ No GPU available, using CPU (will be slower)")
            
            try:
                orchestrator.register_model(model_config)
                _model_cache['orchestrator'] = orchestrator
                logger.info("✅ Model loaded successfully on GPU")
            except Exception as e:
                logger.error(f"❌ Failed to register vision model: {e}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
                raise
        
        orchestrator = _model_cache['orchestrator']
        
        # Load and preprocess image
        start_time = time.time()
        
        if os.path.exists(frame_path):
            # Process frame with vision model
            logger.info(f"🔍 Analyzing frame {frame_index}: {frame_path}")
            
            # Load and prepare image for model input
            from PIL import Image
            pil_image = Image.open(frame_path)
            
            # Prepare content for vision model with detailed prompt
            content = {
                'prompt': f"Frame {frame_index}: Provide a detailed description of this video frame. Include all visible objects, actions, people, text, settings, and any notable details. Be specific and comprehensive.",
                'images': [pil_image],
                'params': {
                    'temperature': 0.7,
                    'max_new_tokens': 300
                }
            }
            
            # Run inference on GPU
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():  # Use automatic mixed precision
                    result = await orchestrator.route_request('qwen_vision_local', content)
            else:
                result = await orchestrator.route_request('qwen_vision_local', content)
            
            # Extract analysis result
            if result and 'response' in result:
                analysis = result['response']
            else:
                analysis = f"Frame {frame_index} analyzed (no detailed response)"
                
            # Log GPU memory usage
            if torch.cuda.is_available():
                logger.info(f"   GPU memory after inference: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
        else:
            analysis = f"Frame {frame_index} - file not found: {frame_path}"
            logger.warning(f"⚠️ Frame file not found: {frame_path}")
        
        process_time = time.time() - start_time
        
        # Calculate timestamp based on frame index and fps
        # Assuming 1 fps extraction rate (1 frame per second)
        timestamp_seconds = frame_index
        timestamp_minutes = timestamp_seconds // 60
        timestamp_secs = timestamp_seconds % 60
        timestamp = f"{timestamp_minutes:02d}:{timestamp_secs:02d}"
        
        return {
            'frame_index': frame_index,
            'frame_path': frame_path,
            'timestamp': timestamp,
            'description': analysis,
            'status': 'completed'
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Frame {frame_index} analysis failed: {str(e)}"
        logger.error(f"❌ Error analyzing frame {frame_index}:")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception message: {str(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return {
            'frame_index': frame_index,
            'frame_path': frame_path,
            'description': error_msg,
            'status': 'error'
        }


@activity.defn
async def generate_summary_activity(session_id: str, frame_results: List[Dict]) -> Dict[str, Any]:
    """Process frame results and prepare structured data (no cloud model needed)"""
    logger.info(f"📝 Processing results for session {session_id} with {len(frame_results)} frames")
    
    try:
        # Extract timestamp-description mappings
        timestamp_descriptions = {}
        analyzed_frames = 0
        
        for result in frame_results:
            if result.get('status') == 'completed':
                timestamp = result.get('timestamp', f"00:{result['frame_index']:02d}")
                description = result.get('description', 'No description available')
                timestamp_descriptions[timestamp] = description
                analyzed_frames += 1
        
        logger.info(f"✅ Processed {analyzed_frames}/{len(frame_results)} frames successfully")
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"🧹 Cleared GPU cache. Current usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        return {
            'session_id': session_id,
            'timestamp_descriptions': timestamp_descriptions,
            'total_frames': len(frame_results),
            'analyzed_frames': analyzed_frames,
            'status': 'completed',
            'message': f'Successfully analyzed {analyzed_frames} frames with detailed descriptions'
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Summary generation failed: {str(e)}"
        logger.error(f"❌ Error generating summary:")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception message: {str(e)}")
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return {
            'session_id': session_id,
            'summary': error_msg,
            'frame_count': len(frame_results),
            'status': 'error'
        }


@activity.defn
async def cleanup_gpu_activity() -> dict:
    """Activity to cleanup GPU memory"""
    import torch
    import gc
    
    try:
        if torch.cuda.is_available():
            before_memory = torch.cuda.memory_allocated() / 1024**3
            
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            after_memory = torch.cuda.memory_allocated() / 1024**3
            freed_memory = before_memory - after_memory
            
            logger.info(f"🧹 GPU cleanup: freed {freed_memory:.2f} GB")
            
            return {
                'before_gb': before_memory,
                'after_gb': after_memory,
                'freed_gb': freed_memory,
                'status': 'success'
            }
        else:
            return {'status': 'no_gpu'}
            
    except Exception as e:
        logger.error(f"❌ GPU cleanup error: {e}")
        return {'status': 'error', 'error': str(e)}


# Define the workflow class
@workflow.defn
class VideoAnalysisWorkflow:
    """GPU-accelerated Temporal workflow for video analysis"""
    
    @workflow.run
    async def run(self, input_data: dict) -> dict:
        """Execute the video analysis workflow with GPU acceleration"""
        
        # Extract input parameters
        session_id = input_data.get('session_id', 'unknown')
        frame_paths = input_data.get('frame_paths', [])
        prompt = input_data.get('prompt', 'Analyze this video')
        batch_size = input_data.get('batch_size', 5)  # Process frames in batches
        
        workflow.logger.info(f"🎬 Starting workflow for session {session_id}")
        workflow.logger.info(f"   Frames to process: {len(frame_paths)}")
        workflow.logger.info(f"   Batch size: {batch_size}")
        
        # Process frames in batches for better GPU utilization
        all_results = []
        
        for batch_start in range(0, len(frame_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(frame_paths))
            batch_paths = frame_paths[batch_start:batch_end]
            
            workflow.logger.info(f"📦 Processing batch: frames {batch_start}-{batch_end}")
            
            # Create tasks for this batch
            batch_tasks = []
            for idx, frame_path in enumerate(batch_paths):
                global_idx = batch_start + idx
                task_data = {
                    'frame_index': global_idx,
                    'frame_path': frame_path,
                    'prompt': prompt,
                    'session_id': session_id
                }
                
                # Execute activity with appropriate timeout
                task = workflow.execute_activity(
                    analyze_frame_activity,
                    task_data,
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=RetryPolicy(
                        maximum_attempts=2,
                        initial_interval=timedelta(seconds=1),
                        maximum_interval=timedelta(seconds=10),
                    )
                )
                batch_tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
            
            # Cleanup GPU memory after each batch
            if batch_end < len(frame_paths):  # Not the last batch
                await workflow.execute_activity(
                    cleanup_gpu_activity,
                    start_to_close_timeout=timedelta(seconds=10)
                )
        
        workflow.logger.info(f"✅ All frames processed, generating summary...")
        
        # Generate summary
        summary_result = await workflow.execute_activity(
            generate_summary_activity,
            args=[session_id, all_results, prompt],
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=2),
            )
        )
        
        # Final GPU cleanup
        cleanup_result = await workflow.execute_activity(
            cleanup_gpu_activity,
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        # Calculate statistics
        successful_frames = sum(1 for r in all_results if r.get('status') == 'success')
        total_process_time = sum(r.get('process_time', 0) for r in all_results)
        
        return {
            'session_id': session_id,
            'total_frames': len(frame_paths),
            'successful_frames': successful_frames,
            'frame_results': all_results,
            'summary': summary_result,
            'gpu_cleanup': cleanup_result,
            'total_process_time': total_process_time,
            'status': 'completed'
        }


async def main():
    """Main function to run the GPU-enabled Temporal worker"""
    
    # Get worker ID from environment or command line
    worker_id = os.getenv("WORKER_ID", sys.argv[1] if len(sys.argv) > 1 else "worker-1")
    logger.info(f"🚀 Starting Temporal GPU Worker: {worker_id}")
    
    try:
        # Connect to Temporal server
        client = await Client.connect("localhost:7233")
        logger.info("✅ Connected to Temporal server")
        
        # Create and run worker with limited concurrency
        worker = Worker(
            client,
            task_queue="video-analysis-gpu-queue",
            workflows=[VideoAnalysisWorkflow],
            activities=[
                analyze_frame_activity,
                generate_summary_activity,
                cleanup_gpu_activity
            ],
            # IMPORTANT: Limit to 1 activity at a time per worker
            max_concurrent_activities=1,
            max_concurrent_workflow_tasks=1,
            # Unique identity for this worker
            identity=worker_id
        )
        
        logger.info(f"✅ GPU-enabled Temporal worker '{worker_id}' started successfully!")
        logger.info("   Task Queue: video-analysis-gpu-queue")
        logger.info("   Workflows: VideoAnalysisWorkflow")
        logger.info("   Activities: analyze_frame, generate_summary, cleanup_gpu")
        logger.info("   Max Concurrent Activities: 1 (sequential processing)")
        logger.info("   Max Concurrent Workflow Tasks: 1")
        logger.info("   Press Ctrl+C to stop...")
        
        await worker.run()
        
    except Exception as e:
        logger.error(f"❌ Failed to start Temporal worker: {e}")
        logger.info("   Make sure Temporal server is running on localhost:7233")


if __name__ == "__main__":
    if TEMPORAL_AVAILABLE:
        asyncio.run(main())
    else:
        logger.error("Cannot run without Temporal packages")
