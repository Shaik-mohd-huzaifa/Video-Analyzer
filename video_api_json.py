#!/usr/bin/env python3
"""
JSON-Based Video Analysis API
Stores all session data in JSON files instead of Redis
"""

import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from threading import Lock
import tempfile
import shutil

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import uvicorn

# Video processing imports
import cv2
import numpy as np
from PIL import Image

# Model imports
from model_manager import ModelOrchestrator

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# Logging Configuration
# ==========================================

class SafeFormatter(logging.Formatter):
    """Custom formatter that prevents binary data from being logged"""
    
    def format(self, record):
        # Convert any non-string arguments to safe representations
        if hasattr(record, 'args') and record.args:
            safe_args = []
            for arg in record.args:
                if isinstance(arg, bytes):
                    safe_args.append(f"<binary data: {len(arg)} bytes>")
                elif isinstance(arg, str) and not arg.isprintable():
                    safe_args.append(f"<non-printable string: {len(arg)} chars>")
                else:
                    safe_args.append(arg)
            record.args = tuple(safe_args)
        
        return super().format(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Apply safe formatter to all handlers
for handler in logging.root.handlers:
    handler.setFormatter(SafeFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

logger = logging.getLogger(__name__)

# ==========================================
# JSON Storage Manager
# ==========================================

class JSONStorageManager:
    """Manages JSON file-based storage for session data"""
    
    def __init__(self, storage_dir: str = "./session_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.lock = Lock()  # Thread safety
        logger.info(f"✅ JSON storage initialized at {self.storage_dir}")
        
    def _get_session_file(self, session_id: str) -> Path:
        """Get the JSON file path for a session"""
        return self.storage_dir / f"session_{session_id}.json"
    
    def _load_session_data(self, session_id: str) -> Dict[str, Any]:
        """Load session data from JSON file"""
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "status": "created",
                "progress_percentage": 0.0,
                "processed_frames": 0,
                "total_frames": 0,
                "processed_batches": 0,
                "total_batches": 0,
                "batches": {},
                "metadata": {}
            }
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load session {session_id}: {e}, creating new")
            return self._load_session_data(session_id)  # Recursive call to create new
    
    def _save_session_data(self, session_id: str, data: Dict[str, Any]):
        """Save session data to JSON file"""
        session_file = self._get_session_file(session_id)
        data["updated_at"] = datetime.now().isoformat()
        
        with self.lock:
            try:
                # Write to temporary file first, then rename (atomic operation)
                temp_file = session_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                temp_file.replace(session_file)
            except Exception as e:
                logger.error(f"Failed to save session {session_id}: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                raise
    
    def get_all_batch_summaries(self, session_id: str) -> List[Dict[str, Any]]:
        """Get ALL batch frame summaries for a session - passed as context by default"""
        data = self._load_session_data(session_id)
        batches = data.get("batches", {})
        
        # Convert batch data to list format
        batch_summaries = []
        for batch_id, batch_data in batches.items():
            if isinstance(batch_data, dict) and "summary" in batch_data:
                batch_summaries.append({
                    "batch_id": batch_id,
                    "frames_processed": batch_data.get("frames_processed", 0),
                    "start_frame": batch_data.get("start_frame", 0),
                    "end_frame": batch_data.get("end_frame", 0),
                    "summary": batch_data["summary"],
                    "timestamp": batch_data.get("timestamp", "")
                })
        
        # Sort by batch_id to maintain order
        batch_summaries.sort(key=lambda x: int(x["batch_id"].split("_")[-1]) if "_" in x["batch_id"] else 0)
        
        logger.info(f"📚 Retrieved {len(batch_summaries)} batch summaries for session {session_id}")
        return batch_summaries
    
    def save_batch_summary(self, session_id: str, batch_id: int, summary: str, 
                          start_frame: int, end_frame: int, frames_processed: int,
                          timestamp_descriptions: dict = None):
        """Save batch summary with frame details and timestamp descriptions"""
        data = self._load_session_data(session_id)
        
        if 'batch_summaries' not in data:
            data['batch_summaries'] = []
        
        # Add or update batch summary
        batch_summary = {
            "batch_id": batch_id,
            "summary": summary,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames_processed": frames_processed,
            "timestamp": datetime.now().isoformat()
        }
        
        # Remove existing batch if updating
        data['batch_summaries'] = [
            b for b in data['batch_summaries'] if b['batch_id'] != batch_id
        ]
        data['batch_summaries'].append(batch_summary)
        
        # Sort by batch_id for consistency
        data['batch_summaries'].sort(key=lambda x: x['batch_id'])
        
        # Store timestamp descriptions separately for detailed access
        if timestamp_descriptions:
            if 'timestamp_descriptions' not in data:
                data['timestamp_descriptions'] = {}
            data['timestamp_descriptions'].update(timestamp_descriptions)
        
        self._save_session_data(session_id, data)
    
    def update_progress(self, session_id: str, status: str, progress_percentage: float,
                       processed_frames: int, total_frames: int, 
                       processed_batches: int, total_batches: int):
        """Update session progress"""
        data = self._load_session_data(session_id)
        
        data.update({
            "status": status,
            "progress_percentage": progress_percentage,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "processed_batches": processed_batches,
            "total_batches": total_batches
        })
        
        self._save_session_data(session_id, data)
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get session status and progress"""
        data = self._load_session_data(session_id)
        return {
            "session_id": session_id,
            "status": data.get("status", "unknown"),
            "progress_percentage": data.get("progress_percentage", 0.0),
            "processed_frames": data.get("processed_frames", 0),
            "total_frames": data.get("total_frames", 0),
            "processed_batches": data.get("processed_batches", 0),
            "total_batches": data.get("total_batches", 0),
            "json_storage": True  # Indicate we're using JSON storage
        }
    
    def list_all_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all sessions with their status"""
        sessions = []
        
        try:
            for session_file in self.storage_dir.glob("session_*.json"):
                if len(sessions) >= limit:
                    break
                    
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        sessions.append({
                            "session_id": data.get("session_id", "unknown"),
                            "status": data.get("status", "unknown"),
                            "progress_percentage": data.get("progress_percentage", 0.0),
                            "created_at": data.get("created_at", ""),
                            "updated_at": data.get("updated_at", ""),
                            "total_batches": len(data.get("batches", {}))
                        })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping corrupted session file {session_file}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            
        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data"""
        session_file = self._get_session_file(session_id)
        
        with self.lock:
            try:
                if session_file.exists():
                    session_file.unlink()
                    logger.info(f"🗑️ Deleted session {session_id}")
                    return True
                else:
                    logger.warning(f"Session {session_id} not found for deletion")
                    return False
            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {e}")
                return False
    
    def get_full_context(self, session_id: str) -> Dict[str, Any]:
        """Get complete session context"""
        data = self._load_session_data(session_id)
        batch_summaries = self.get_all_batch_summaries(session_id)
        
        return {
            "session_id": session_id,
            "status": data.get("status", "unknown"),
            "progress_percentage": data.get("progress_percentage", 0.0),
            "processed_frames": data.get("processed_frames", 0),
            "total_frames": data.get("total_frames", 0),
            "batch_summaries": batch_summaries,
            "total_batches": len(batch_summaries),
            "metadata": data.get("metadata", {}),
            "created_at": data.get("created_at", ""),
            "updated_at": data.get("updated_at", "")
        }

# ==========================================
# Initialize JSON Storage Manager
# ==========================================
storage_manager = JSONStorageManager()

# Global model orchestrator (initialized in startup_event)
model_orchestrator = None

# Global Temporal client (initialized in startup_event)
temporal_client = None

# ==========================================
# Pydantic Models
# ==========================================

class ChatRequest(BaseModel):
    session_id: str
    question: str
    include_all_batches: bool = True
    max_context_batches: Optional[int] = None

class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    context_batches_used: int
    total_frames_in_context: int
    progress_percentage: float

class StatusResponse(BaseModel):
    session_id: str
    status: str
    progress_percentage: float
    processed_frames: int
    total_frames: int
    processed_batches: int
    total_batches: int
    json_storage: bool

# ==========================================
# FastAPI App Setup
# ==========================================

app = FastAPI(
    title="JSON-Based Video Analysis API",
    description="Video analysis API with JSON file storage for session management",
    version="1.1.0"
)

# Custom error handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors gracefully to prevent Unicode decode issues"""
    logger.warning(f"Validation error for {request.url}: {str(exc)[:200]}...")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request format",
            "detail": "Please ensure you're uploading a valid video file using multipart/form-data",
            "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
        }
    )

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting JSON-based Video Analysis API...")
    logger.info(f"📁 Session data will be stored in: {storage_manager.storage_dir}")
    
    # Initialize model orchestrator with proper model registration
    logger.info("Initializing hybrid model setup: Local vision + Cloud chat...")
    
    # Import required classes
    from model_manager import ModelConfig, ModelType
    
    # Initialize the global model orchestrator for the API
    global model_orchestrator
    model_orchestrator = ModelOrchestrator()
    
    # Register local Qwen2.5-VL-7B-Instruct for vision tasks
    local_vision_config = ModelConfig(
        name="qwen_vision_local",
        type=ModelType.QWEN_VISION,
        model_id="/workspace/models/Qwen_Qwen2.5-VL-7B-Instruct",
        device="cuda",
        precision="float16",
        max_memory="6GB"  # Adjust based on your GPU memory
    )
    model_orchestrator.register_model(local_vision_config)
    
    # Register Nebius GPT-OSS-120B for chat/Q&A tasks
    cloud_chat_config = ModelConfig(
        name="gpt_oss_120b",
        type=ModelType.NEBIUS_GPT,
        endpoint=os.getenv("NEBIUS_GPT_ENDPOINT")
    )
    model_orchestrator.register_model(cloud_chat_config)
    
    logger.info("✅ Hybrid setup ready: Local Qwen2.5-VL-7B + Cloud GPT-OSS-120B!")
    
    # Initialize Temporal client for parallel workflow execution
    global temporal_client
    try:
        from temporalio.client import Client
        temporal_client = await Client.connect("localhost:7233")
        logger.info("✅ Connected to Temporal server for parallel processing")
        logger.info("🚀 JSON-based API with Temporal parallel batch processing enabled")
    except Exception as e:
        logger.warning(f"⚠️ Temporal connection failed: {e}")
        logger.warning("🔄 Falling back to sequential processing mode")
        temporal_client = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    session_count = len(list(storage_manager.storage_dir.glob("session_*.json")))
    
    return {
        "status": "healthy",
        "service": "video-analysis-api",
        "version": "1.1.0",
        "storage": "json_files",
        "storage_location": str(storage_manager.storage_dir),
        "active_sessions": session_count,
        "timestamp": datetime.now().isoformat()
    }

# Comprehensive default system prompt
DEFAULT_SYSTEM_PROMPT = """
You are an expert video analyst. Analyze these video frames comprehensively and provide detailed insights covering:

**SCENE & ENVIRONMENT:**
- Location type, setting, lighting conditions, weather, time of day
- Background elements, architecture, landscape features
- Spatial layout and scene composition

**OBJECTS & ELEMENTS:**
- All visible objects, items, tools, vehicles, equipment
- Brands, text, signs, symbols, logos
- Colors, materials, textures, sizes, conditions

**PEOPLE ANALYSIS:**
- Number of people, demographics, clothing, accessories
- Facial expressions, body language, posture, gestures
- Interactions between people, relationships, social dynamics

**ACTIONS & MOVEMENTS:**
- Specific actions being performed by people and objects
- Movement patterns, speed, direction, coordination
- Sequential activities and their progression

**EVENTS & ACTIVITIES:**
- Main events occurring, their significance and context
- Cause-and-effect relationships, before/during/after states
- Notable moments, changes, transitions, or developments

**TECHNICAL DETAILS:**
- Camera angles, shots, movements, focus
- Audio cues if applicable, visual effects
- Production quality, editing techniques

**CONTEXT & MEANING:**
- Purpose or goal of activities shown
- Cultural, social, or professional context
- Historical or temporal context if relevant

**EMOTIONS & ATMOSPHERE:**
- Mood, tone, emotional undertones
- Tension, excitement, calmness, or other atmospheric qualities
- Viewer impact and emotional response

**SEMANTIC UNDERSTANDING:**
- Overall narrative or story being told
- Themes, messages, or lessons conveyed
- Significance and broader implications

Provide a rich, detailed analysis that captures all important aspects of what's happening in the video.
"""

def validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file"""
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported formats: {', '.join(allowed_extensions)}"
        )

@app.post("/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    prompt: str = Form("")
):
    """
    Start video analysis workflow and return session ID immediately.
    Processing happens in background via Temporal workflow.
    """
    
    # Validate the uploaded file
    validate_video_file(video)
    
    session_id = str(uuid.uuid4())
    logger.info(f"🎥 Starting video analysis workflow for session {session_id}")
    
    # Use comprehensive default prompt if none provided
    effective_prompt = prompt.strip() if prompt.strip() else DEFAULT_SYSTEM_PROMPT
    
    temp_video_path = None
    
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as temp_file:
            temp_video_path = temp_file.name
            shutil.copyfileobj(video.file, temp_file)
        
        logger.info(f"💾 Saved video to temporary file: {temp_video_path}")
        
        # Initialize video capture to get basic stats
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"📊 Video stats - Frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        # Calculate batching
        frames_per_batch = 50
        total_batches = max(1, (total_frames + frames_per_batch - 1) // frames_per_batch)
        
        # Initialize session with 'starting' status
        storage_manager.update_progress(
            session_id=session_id,
            status="starting",
            progress_percentage=0.0,
            processed_frames=0,
            total_frames=total_frames,
            processed_batches=0,
            total_batches=total_batches
        )
        
        # Start Temporal workflow for parallel processing
        await start_temporal_video_workflow(
            session_id=session_id,
            video_path=temp_video_path,
            prompt=effective_prompt,
            total_frames=total_frames,
            total_batches=total_batches,
            frames_per_batch=frames_per_batch
        )
        
        logger.info(f"🚀 Background workflow started for session {session_id}")
        
        # Return session info immediately (non-blocking)
        return {
            "session_id": session_id,
            "status": "starting",
            "message": "Video analysis workflow started successfully",
            "total_frames": total_frames,
            "total_batches": total_batches,
            "duration_seconds": duration,
            "frames_per_batch": frames_per_batch,
            "prompt_used": "comprehensive_default" if not prompt.strip() else "custom",
            "storage": "json_files",
            "next_steps": {
                "check_status": f"/status/{session_id}",
                "chat_when_ready": f"/chat",
                "get_full_context": f"/context/{session_id}"
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Error starting video analysis workflow: {e}")
        
        # Clean up temp file on error
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Failed to start video analysis workflow: {str(e)}")


async def start_temporal_video_workflow(
    session_id: str,
    video_path: str,
    prompt: str,
    total_frames: int,
    total_batches: int,
    frames_per_batch: int
):
    """Start Temporal workflow for parallel video processing"""
    
    global temporal_client
    
    # If Temporal is not available, fall back to sequential processing
    if temporal_client is None:
        logger.warning("⚠️ Temporal not available, starting fallback sequential processing")
        asyncio.create_task(process_video_fallback(
            session_id, video_path, prompt, total_frames, total_batches, frames_per_batch
        ))
        return
    
    try:
        # Extract frames and prepare for Temporal workflow
        frame_paths = await extract_frames_for_temporal(video_path, session_id)
        
        # Prepare workflow input
        workflow_input = {
            "session_id": session_id,
            "frame_paths": frame_paths,
            "prompt": prompt,
            "batch_size": frames_per_batch,
            "storage_manager": "json_files"  # Indicate we're using JSON storage
        }
        
        logger.info(f"🚀 Starting Temporal workflow for session {session_id}")
        logger.info(f"   Total frames: {len(frame_paths)}")
        logger.info(f"   Batch size: {frames_per_batch}")
        
        # Start Temporal workflow (non-blocking)
        handle = await temporal_client.start_workflow(
            "VideoAnalysisWorkflow",
            workflow_input,
            id=f"video-analysis-{session_id}",
            task_queue="video-analysis-gpu-queue",
        )
        
        logger.info(f"✅ Temporal workflow started with ID: {handle.id}")
        
        # Start background task to monitor workflow progress and update JSON storage
        asyncio.create_task(monitor_temporal_workflow(handle, session_id, total_frames, total_batches))
        
    except Exception as e:
        logger.error(f"❌ Error starting Temporal workflow for session {session_id}: {e}")
        logger.warning("🔄 Falling back to sequential processing")
        
        # Fall back to sequential processing
        asyncio.create_task(process_video_fallback(
            session_id, video_path, prompt, total_frames, total_batches, frames_per_batch
        ))


async def extract_frames_for_temporal(video_path: str, session_id: str) -> List[str]:
    """Extract frames from video and save as individual files for Temporal processing"""
    
    frame_dir = Path(f"./temp_frames/{session_id}")
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = frame_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            frame_idx += 1
            
    finally:
        cap.release()
        # Clean up original video file
        if os.path.exists(video_path):
            os.unlink(video_path)
    
    logger.info(f"📁 Extracted {len(frame_paths)} frames for Temporal processing")
    return frame_paths


async def monitor_temporal_workflow(handle, session_id: str, total_frames: int, total_batches: int):
    """Monitor Temporal workflow progress and update JSON storage"""
    
    try:
        logger.info(f"📊 Monitoring Temporal workflow for session {session_id}")
        
        # Wait for workflow completion
        result = await handle.result()
        
        logger.info(f"✅ Temporal workflow completed for session {session_id}")
        
        # Update final status in JSON storage
        storage_manager.update_progress(
            session_id=session_id,
            status="completed",
            progress_percentage=100.0,
            processed_frames=total_frames,
            total_frames=total_frames,
            processed_batches=total_batches,
            total_batches=total_batches
        )
        
        # Clean up temporary frame directory
        frame_dir = Path(f"./temp_frames/{session_id}")
        if frame_dir.exists():
            import shutil
            shutil.rmtree(frame_dir)
            logger.info(f"🧹 Cleaned up temporary frames for session {session_id}")
            
    except Exception as e:
        logger.error(f"❌ Error monitoring Temporal workflow for session {session_id}: {e}")
        
        # Update error status
        storage_manager.update_progress(
            session_id=session_id,
            status="error",
            progress_percentage=0.0,
            processed_frames=0,
            total_frames=total_frames,
            processed_batches=0,
            total_batches=total_batches
        )


async def process_video_fallback(
    session_id: str,
    video_path: str,
    prompt: str,
    total_frames: int,
    total_batches: int,
    frames_per_batch: int
):
    """Fallback sequential processing when Temporal is not available"""
    
    logger.info(f"🔄 Starting fallback sequential processing for session {session_id}")
    
    try:
        # Update status to processing
        storage_manager.update_progress(
            session_id=session_id,
            status="processing",
            progress_percentage=0.0,
            processed_frames=0,
            total_frames=total_frames,
            processed_batches=0,
            total_batches=total_batches
        )
        
        # Use global model orchestrator
        global model_orchestrator
        if model_orchestrator is None:
            raise Exception("Model orchestrator not initialized")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        processed_frames = 0
        
        # Process video in batches (sequential)
        for batch_num in range(total_batches):
            start_frame = batch_num * frames_per_batch
            end_frame = min((batch_num + 1) * frames_per_batch, total_frames)
            
            logger.info(f"🔄 Processing batch {batch_num + 1}/{total_batches} (frames {start_frame}-{end_frame-1})")
            
            # Extract frames for this batch
            batch_frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                batch_frames.append(pil_image)
            
            if not batch_frames:
                logger.warning(f"No frames extracted for batch {batch_num + 1}")
                continue
            
            # Analyze batch frames using route_request
            try:
                result = await model_orchestrator.route_request(
                    request_type="vision",
                    content={
                        "prompt": prompt,
                        "images": batch_frames,
                        "params": {}
                    }
                )
                analysis_result = result["response"]
                
                batch_id = f"batch_{batch_num + 1}"
                
                # Store batch summary
                storage_manager.store_batch_summary(
                    session_id=session_id,
                    batch_id=batch_id,
                    frames_processed=len(batch_frames),
                    start_frame=start_frame,
                    end_frame=end_frame - 1,
                    summary=analysis_result
                )
                
                processed_frames += len(batch_frames)
                progress_percentage = (processed_frames / total_frames) * 100
                
                # Update progress
                storage_manager.update_progress(
                    session_id=session_id,
                    status="processing",
                    progress_percentage=progress_percentage,
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                    processed_batches=batch_num + 1,
                    total_batches=total_batches
                )
                
                logger.info(f"✅ Completed batch {batch_num + 1}/{total_batches} ({progress_percentage:.1f}%)")
            
            except Exception as e:
                logger.error(f"❌ Error processing batch {batch_num + 1}: {e}")
                continue
        
        cap.release()
        
        # Mark as completed
        storage_manager.update_progress(
            session_id=session_id,
            status="completed",
            progress_percentage=100.0,
            processed_frames=processed_frames,
            total_frames=total_frames,
            processed_batches=total_batches,
            total_batches=total_batches
        )
        
        logger.info(f"✅ Fallback processing completed for session {session_id}")
    
    except Exception as e:
        logger.error(f"❌ Error in fallback processing for session {session_id}: {e}")
        
        # Update session with error status
        storage_manager.update_progress(
            session_id=session_id,
            status="error",
            progress_percentage=0.0,
            processed_frames=0,
            total_frames=total_frames,
            processed_batches=0,
            total_batches=total_batches
        )
    
    finally:
        # Clean up temporary file
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
                logger.info(f"🧹 Cleaned up temporary file: {video_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")

@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    """Get processing status with progress percentage"""
    status_data = storage_manager.get_session_status(session_id)
    return StatusResponse(**status_data)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_video(request: ChatRequest):
    """
    Chat about video using available frame descriptions with cloud GPT model
    """
    
    # Load session data to get timestamp descriptions
    session_data = storage_manager._load_session_data(request.session_id)
    
    if not session_data:
        raise HTTPException(
            status_code=404, 
            detail=f"No video analysis found for session {request.session_id}"
        )
    
    # Get timestamp descriptions (frame-by-frame analysis)
    timestamp_descriptions = session_data.get('timestamp_descriptions', {})
    
    if not timestamp_descriptions:
        raise HTTPException(
            status_code=404,
            detail=f"No frame descriptions available yet for session {request.session_id}. Analysis may still be in progress."
        )
    
    # Build context from available frame descriptions
    context_parts = []
    sorted_timestamps = sorted(timestamp_descriptions.keys())
    
    # Use all available descriptions or limit if too many
    max_frames_for_context = 100  # Limit to avoid token overflow
    timestamps_to_use = sorted_timestamps[:max_frames_for_context] if len(sorted_timestamps) > max_frames_for_context else sorted_timestamps
    
    for timestamp in timestamps_to_use:
        context_parts.append(f"[{timestamp}] {timestamp_descriptions[timestamp]}")
    
    combined_context = "\n".join(context_parts)
    
    # Get current session status for progress info
    session_status = storage_manager.get_session_status(request.session_id)
    progress_info = f"Analysis progress: {session_status['progress_percentage']:.1f}% ({len(timestamp_descriptions)} frames analyzed)"
    
    try:
        # Use global model orchestrator for cloud GPT
        global model_orchestrator
        if model_orchestrator is None:
            raise HTTPException(status_code=500, detail="Model orchestrator not initialized")
        
        # Prepare prompt for cloud GPT model
        chat_prompt = f"""You are analyzing a video based on frame-by-frame descriptions.

{progress_info}

Video Frame Descriptions:
{combined_context}

User Question: {request.question}

Provide a detailed and accurate answer based on the video content described above."""
        
        # Route to cloud GPT model
        result = await model_orchestrator.route_request(
            request_type="gpt_oss_120b",  # Use the cloud model directly
            content={
                "prompt": chat_prompt,
                "params": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
        )
        answer = result.get("response", "Unable to generate response")
        
        logger.info(f"💬 Chat response generated for session {request.session_id}")
        
        return ChatResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            context_batches_used=len(timestamps_to_use),
            total_frames_in_context=len(timestamps_to_use),
            progress_percentage=session_status["progress_percentage"]
        )
    
    except Exception as e:
        logger.error(f"❌ Error generating chat response: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/sessions")
async def list_sessions(limit: int = 50):
    """List all active sessions with progress"""
    sessions = storage_manager.list_all_sessions(limit=limit)
    return {
        "sessions": sessions,
        "total_returned": len(sessions),
        "storage": "json_files"
    }

@app.get("/context/{session_id}")
async def get_full_context(session_id: str):
    """Get full context for a session (all batch summaries, progress, etc.)"""
    context = storage_manager.get_full_context(session_id)
    
    if context["status"] == "unknown":
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return context

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its data from JSON storage"""
    success = storage_manager.delete_session(session_id)
    
    if success:
        return {
            "message": f"Session {session_id} deleted successfully",
            "session_id": session_id,
            "storage": "json_files"
        }
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('API_PORT', 8000))
    logger.info(f"🚀 Starting JSON-based Video Analysis API on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
