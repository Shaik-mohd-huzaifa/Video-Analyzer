# AI-Powered Video Analysis System

A production-ready, scalable video analysis system that combines local vision AI models with cloud-based language models for intelligent video understanding. Built with Temporal workflow orchestration for robust parallel processing.

## 🏗️ Architecture Overview

### System Components

1. **Vision AI Processing** (`temporal_worker_gpu.py`)
   - Local **Qwen2.5-VL-7B-Instruct** vision model for frame analysis
   - GPU-accelerated inference with 4-bit/8-bit quantization
   - Detailed frame-by-frame descriptions with timestamp mapping
   - Automatic mixed precision for optimal performance

2. **Workflow Orchestration** (Temporal)
   - Distributed task queue for parallel frame processing
   - Fault-tolerant execution with automatic retries
   - Scalable worker pool management

3. **API Server** (`video_api_json.py`)
   - FastAPI-based REST endpoints
   - Asynchronous video upload and processing
   - Real-time progress tracking
   - Chat interface with cloud GPT integration

4. **Model Orchestration** (`model_manager.py`)
   - Multi-model management system
   - Hybrid architecture: Qwen2.5-VL-7B (local) + Nebius GPT-OSS-120B (cloud)
   - Intelligent request routing (vision → local, chat → cloud)
   - GPU memory optimization with 4-bit/8-bit quantization
   - Automatic cleanup and fallback mechanisms

## 📊 Code Quality

### Design Principles
- **Separation of Concerns**: Vision AI, workflow orchestration, and chat functionality are cleanly separated
- **Scalability**: Worker pool architecture allows linear scaling (1-10+ workers)
- **Fault Tolerance**: Automatic retries, worker restart, and graceful degradation
- **Resource Efficiency**: GPU memory optimization, batch processing, and intelligent caching
- **Type Safety**: Comprehensive type hints and dataclass usage
- **Async/Await**: Non-blocking I/O for maximum throughput

### Performance Features
- **Parallel Processing**: Multiple workers process frames concurrently
- **GPU Optimization**: Mixed precision inference, memory cleanup between batches
- **Incremental Updates**: Real-time progress tracking and partial results
- **Efficient Storage**: JSON-based session management with timestamp indexing

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for multi-worker)
- CUDA 11.8+ and cuDNN
- 20-30GB free disk space for Qwen2.5-VL-7B model
- RAM: 16GB+ system memory

### 1. Clone and Navigate
```bash
git clone https://github.com/Shaik-mohd-huzaifa/Video-Analyzer.git
cd Video-Analyzer
```

### 2. Install Dependencies

#### Core Dependencies
```bash
pip install -r requirements.txt
pip install -r api_requirements.txt
```

#### Install Temporal CLI
```bash
curl -sSf https://temporal.download/cli.sh | sh
sudo mv temporal /usr/local/bin/
```

### 3. Download Vision Model

#### Available Models

| Model | Size | Purpose | Memory Required | Download Command |
|-------|------|---------|-----------------|------------------|
| **Qwen2.5-VL-7B-Instruct** | 15GB | Vision Analysis (Current) | ~8GB (int8) | See below |
| Qwen2.5-VL-2B-Instruct | 5GB | Vision (Lightweight) | ~3GB (int8) | Alternative option |

#### Download Qwen2.5-VL-7B (Currently Used)

The system will automatically download the model on first run, or manually:
```bash
# Using Hugging Face CLI (recommended)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/Qwen_Qwen2.5-VL-7B-Instruct

# Alternative: Using Python
python -c "from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='./models')"
```

**Note**: The model files are ~15GB. Ensure you have sufficient disk space and a stable internet connection.

### 4. Configure Environment

Create `.env` file:
```bash
# Cloud GPT Configuration
NEBIUS_GPT_ENDPOINT=https://api.studio.nebius.ai/v1/chat/completions
NEBIUS_API_KEY=your_api_key_here

# Worker Configuration (optional)
MAX_WORKERS=5
MAX_BATCH_SIZE=10
FRAMES_PER_SECOND=1

# GPU Settings (optional)
CUDA_VISIBLE_DEVICES=0
```

## 🎯 Running the System

### Quick Start (All-in-One)

```bash
# 1. Start Temporal Server
temporal server start-dev &

# 2. Start Video API Server
python video_api_json.py &

# 3. Start Worker Pool (3 workers)
python start_multiple_workers.py 3
```

### Component-by-Component

#### 1. Start Temporal Server
```bash
temporal server start-dev
# Access UI at http://localhost:8233
```

#### 2. Start API Server
```bash
python video_api_json.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

#### 3. Start Workers

**Single Worker** (1 frame at a time):
```bash
python temporal_worker_gpu.py
```

**Multiple Workers** (parallel processing):
```bash
# Start 5 workers
python start_multiple_workers.py 5

# Start 10 workers (maximum recommended)
python start_multiple_workers.py 10
```

## 📡 API Usage

### Analyze Video
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "video=@video.mp4" \
  -F "prompt=Describe what happens in this video"
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716",
  "status": "starting",
  "message": "Video analysis workflow started successfully",
  "total_frames": 150,
  "total_batches": 15,
  "duration_seconds": 150.5,
  "frames_per_batch": 10,
  "prompt_used": "custom",
  "storage": "json_files",
  "next_steps": {
    "check_status": "/status/550e8400-e29b-41d4-a716",
    "chat_when_ready": "/chat",
    "get_full_context": "/context/550e8400-e29b-41d4-a716"
  }
}
```

### Check Progress
```bash
curl "http://localhost:8000/status/{session_id}"
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716",
  "status": "processing",
  "progress_percentage": 45.5,
  "processed_frames": 68,
  "total_frames": 150
}
```

### Chat with Video
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716",
    "question": "What activities are happening in this video?"
  }'
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716",
  "question": "What activities are happening in this video?",
  "answer": "Based on the frame analysis, the video shows...",
  "context_batches_used": 45,
  "total_frames_in_context": 45,
  "progress_percentage": 100.0
}
```

### List All Sessions
```bash
curl "http://localhost:8000/sessions?limit=50"
```

Response:
```json
{
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716",
      "status": "completed",
      "progress_percentage": 100.0,
      "total_frames": 150,
      "processed_frames": 150
    }
  ],
  "total_returned": 1,
  "storage": "json_files"
}
```

### Get Full Context
```bash
curl "http://localhost:8000/context/{session_id}"
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716",
  "status": "completed",
  "progress": {
    "processed_frames": 150,
    "total_frames": 150,
    "progress_percentage": 100.0
  },
  "batch_summaries": [...],
  "total_batches": 15
}
```

### Delete Session
```bash
curl -X DELETE "http://localhost:8000/sessions/{session_id}"
```

Response:
```json
{
  "message": "Session 550e8400-e29b-41d4-a716 deleted successfully",
  "session_id": "550e8400-e29b-41d4-a716",
  "storage": "json_files"
}
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "temporal_connected": true,
  "cloud_gpt_configured": true,
  "sessions_active": 3,
  "storage": "json_files"
}
```

## ⚙️ Configuration Options

### Worker Scaling
```python
# In temporal_worker_gpu.py
max_concurrent_activities=1  # Activities per worker
max_concurrent_workflow_tasks=1  # Workflow tasks per worker
```

### Batch Processing
```python
# In video_api_json.py
batch_size = min(10, total_frames)  # Frames per batch
max_workers = 5  # Parallel batches
```

### Model Parameters
```python
# In temporal_worker_gpu.py
'temperature': 0.7,  # Generation randomness (0.0-1.0)
'max_new_tokens': 300  # Max tokens per frame description
```

## 🎛️ Performance Tuning

### GPU Memory Management

| Workers | GPU VRAM | Recommended GPU | Notes |
|---------|----------|-----------------|-------|
| 1       | 8-10 GB  | RTX 3080/4070   | Qwen2.5-VL-7B with int8 quantization |
| 2-3     | 16-20 GB | RTX 4090        | Multiple instances with memory sharing |
| 4-5     | 24-32 GB | A5000/A6000     | Optimal for production workloads |
| 6-10    | 40-48 GB | A100            | Enterprise-scale processing |

### Optimization Tips

1. **Frame Extraction Rate**
   - Default: 1 fps (balanced)
   - Fast motion: 2-3 fps
   - Slow/static: 0.5 fps

2. **Batch Size**
   - Small videos (<100 frames): 5-10 frames/batch
   - Large videos (>500 frames): 20-30 frames/batch

3. **Worker Count**
   - CPU bound: 1-2 workers
   - GPU bound: Match GPU memory capacity
   - Network bound: 5-10 workers

## 📦 Project Structure

```
video-analysis-system/
├── temporal_worker_gpu.py      # GPU worker for frame analysis
├── video_api_json.py           # FastAPI server
├── model_manager.py            # Model orchestration
├── start_multiple_workers.py   # Multi-worker launcher
├── requirements.txt            # Python dependencies
├── api_requirements.txt        # API dependencies
├── .env                        # Environment configuration
├── models/                     # Model files
│   ├── Qwen_Qwen2.5-VL-7B-Instruct/  # Vision model (15GB)
│   │   ├── *.json             # Config files (included in repo)
│   │   ├── *.txt              # Vocab files (included in repo)
│   │   └── *.safetensors      # Model weights (download separately)
│   └── models--Qwen--Qwen2.5-VL-7B-Instruct/  # HF cache
└── session_data/              # Session storage (auto-created)
```

## 🔍 Monitoring

### Temporal UI
- URL: http://localhost:8233
- Monitor workflows, activities, and worker health
- View execution history and errors

### API Metrics
- Endpoint: http://localhost:8000/docs
- Interactive API documentation
- Request/response monitoring

### Logs
```bash
# API logs
tail -f video_api.log

# Worker logs
tail -f temporal_worker.log

# Temporal server logs
tail -f temporal_server.log
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce worker count
   - Decrease batch size
   - Enable 4-bit quantization

2. **Slow Processing**
   - Increase worker count
   - Check GPU utilization (`nvidia-smi`)
   - Optimize frame extraction rate

3. **Worker Crashes**
   - Check GPU memory
   - Verify model files
   - Review worker logs

4. **API Timeout**
   - Check Temporal server status
   - Verify worker registration
   - Monitor task queue

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please follow:
1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

## 📧 Support

For issues and questions:
- GitHub Issues: [https://github.com/Shaik-mohd-huzaifa/Video-Analyzer/issues](https://github.com/Shaik-mohd-huzaifa/Video-Analyzer/issues)
- Repository: [https://github.com/Shaik-mohd-huzaifa/Video-Analyzer](https://github.com/Shaik-mohd-huzaifa/Video-Analyzer)
