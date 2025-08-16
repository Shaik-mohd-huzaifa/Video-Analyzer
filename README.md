# Multi-Model Orchestration: Nebius GPT + Local Qwen

A production-ready system for orchestrating cloud-based Nebius GPT with locally-running Qwen models, optimized for GPU memory and performance.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Model Orchestrator                 │
├──────────────┬──────────────┬───────────────┤
│  Nebius GPT  │  Qwen Vision │ Qwen Structured│
│   (Cloud)    │   (Local)    │    (Local)    │
├──────────────┴──────────────┴───────────────┤
│         GPU Memory Manager                   │
└─────────────────────────────────────────────┘
```

## 📋 Prerequisites

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **CUDA**: 11.8 or higher
- **Python**: 3.8+
- **Disk Space**: 20-50GB depending on models
- **RAM**: 16GB+ system memory

## 🚀 Quick Start

### Option 1: Direct Setup (No Docker)

```bash
# 1. Clone/navigate to project directory
cd /workspace

# 2. Make setup script executable
chmod +x setup.sh

# 3. Run setup (creates venv, installs dependencies)
./setup.sh

# 4. Configure environment
cp .env.example .env
# Edit .env with your Nebius API key
nano .env

# 5. Activate environment
source activate.sh

# 6. Download Qwen model (start with smallest)
python download_qwen.py qwen2-vl-2b

# 7. Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 8. Run example
python example_usage.py
```

### Option 2: Docker Deployment

```bash
# 1. Build Docker image
docker-compose build

# 2. Set environment variables
export NEBIUS_API_KEY="your_key_here"
export NEBIUS_GPT_ENDPOINT="https://api.nebius.ai/v1/completions"

# 3. Download models first (outside container)
python download_qwen.py qwen2-vl-2b

# 4. Start services
docker-compose up -d

# 5. Check logs
docker-compose logs -f model-server
```

## 📦 Model Download Guide

### Available Models

| Model | Size | Purpose | Memory (int8) | Command |
|-------|------|---------|---------------|---------|
| qwen2-vl-2b | 5GB | Vision (small) | ~3GB | `python download_qwen.py qwen2-vl-2b` |
| qwen2-vl-7b | 15GB | Vision (large) | ~8GB | `python download_qwen.py qwen2-vl-7b` |
| qwen-7b-chat | 14GB | Chat/Structured | ~7GB | `python download_qwen.py qwen-7b-chat` |
| qwen-14b-chat | 28GB | Chat (large) | ~14GB | `python download_qwen.py qwen-14b-chat` |

### Download Commands

```bash
# List supported models
python download_qwen.py list

# Download specific model
python download_qwen.py qwen2-vl-2b

# Verify download
python download_qwen.py qwen2-vl-2b --verify-only

# Use alternative download method
python download_qwen.py qwen-7b-chat --method snapshot
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Nebius Configuration
NEBIUS_API_KEY=your_api_key_here
NEBIUS_GPT_ENDPOINT=https://api.nebius.ai/v1/completions

# GPU Settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Paths
HF_HOME=./cache
TRANSFORMERS_CACHE=./cache
```

### Memory Optimization

```python
# In model_manager.py, adjust precision:
ModelConfig(
    name="qwen_vision",
    type=ModelType.QWEN_VISION,
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    precision="int4",  # Options: float16, int8, int4
    max_memory="8GB"   # Limit per model
)
```

## 💻 Usage Examples

### Basic Usage

```python
from model_manager import ModelOrchestrator, ModelConfig, ModelType

# Initialize
orchestrator = ModelOrchestrator()

# Register models
orchestrator.register_model(ModelConfig(
    name="nebius_gpt",
    type=ModelType.NEBIUS_GPT,
    endpoint="https://api.nebius.ai/v1/completions"
))

# Route request
response = await orchestrator.route_request(
    "chat",
    {"prompt": "Explain quantum computing"}
)
```

### Vision Pipeline

```python
# Analyze image with Qwen, then discuss with Nebius GPT
pipeline = MultiModelPipeline()
result = await pipeline.vision_to_chat_pipeline("image.jpg")
print(result["insights"])
```

## 🔍 Verification Steps

### 1. GPU Check
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 2. Model Loading Test
```python
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', cache_dir='./models')
print('✓ Model loading works')
"
```

### 3. Memory Monitor
```bash
# Watch GPU memory during inference
watch -n 1 nvidia-smi
```

### 4. API Connection Test
```bash
curl -X POST $NEBIUS_GPT_ENDPOINT \
  -H "Authorization: Bearer $NEBIUS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

## ⚠️ Common Issues & Solutions

### Out of Memory (OOM)

```python
# Solution 1: Use smaller model
model_id="Qwen/Qwen2-VL-2B-Instruct"  # Instead of 7B

# Solution 2: Increase quantization
precision="int4"  # Instead of int8 or float16

# Solution 3: Clear cache between models
orchestrator.memory_manager.cleanup()
```

### Slow Download

```bash
# Use HF CLI with resume
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
  --cache-dir ./models \
  --resume-download \
  --local-dir ./models/Qwen_Qwen2-VL-2B-Instruct
```

### CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

## 📊 Performance Tips

1. **Model Loading Strategy**
   - Load Nebius GPT client once (lightweight)
   - Swap local models as needed
   - Keep only one large model in memory

2. **Batch Processing**
   ```python
   # Process multiple items efficiently
   for batch in chunks(items, batch_size=4):
       responses = await asyncio.gather(*[
           orchestrator.route_request("vision", item) 
           for item in batch
       ])
   ```

3. **GPU Memory Management**
   ```python
   # Always cleanup after large operations
   with memory_manager.managed_memory():
       result = model.generate(prompt)
   # Memory automatically cleaned
   ```

## 🔗 API Endpoints (Optional)

If you want to add a REST API:

```python
# Add to example_usage.py or create api_server.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/chat")
async def chat(prompt: str):
    return await orchestrator.route_request("chat", {"prompt": prompt})

@app.post("/vision")
async def vision(image_url: str, prompt: str):
    return await orchestrator.route_request("vision", {
        "prompt": prompt,
        "images": [load_image(image_url)]
    })

# Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## 📚 Prompt Engineering Guide

### Nebius GPT (Conversation/QA)

```python
prompt = """
You are an expert assistant. Follow these guidelines:
1. Be concise but comprehensive
2. Use structured output when appropriate
3. Cite sources when possible

Question: {user_question}
"""
```

### Qwen Vision

```python
prompt = """
Analyze this image and provide:
1. Main objects and their positions
2. Text content (OCR)
3. Overall context and scene description
4. Quality assessment

Output format: JSON
"""
```

### Qwen Structured Output

```python
prompt = f"""
Generate a JSON response following this schema:
{json.dumps(schema, indent=2)}

Requirements:
- All fields are required
- Use appropriate data types
- Follow the schema exactly

Task: {task_description}
"""
```

## 🐛 Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In code
logger.debug(f"Model loading: {model_id}")
logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

## 📈 Monitoring

```bash
# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
    echo -e "\n=== Process Status ==="
    ps aux | grep python | head -5
    echo -e "\n=== Disk Usage ==="
    df -h | grep -E "Filesystem|/workspace"
    sleep 2
done
EOF
chmod +x monitor.sh
```

## 🚢 Production Deployment

For production on RunPod or similar:

1. **Use the Docker image**
2. **Set resource limits**
3. **Enable health checks**
4. **Use environment-specific configs**
5. **Implement request queuing**
6. **Add monitoring/alerting**

## 📞 Support

- Check GPU compatibility: `torch.cuda.get_device_capability()`
- Verify model files: `python download_qwen.py list`
- Test memory limits: `python example_usage.py`

---

**Next Steps:**
1. Configure your Nebius API key in `.env`
2. Download your first model
3. Run the example to verify everything works
4. Customize the orchestrator for your use case

# AI-Powered Video Analysis System

A production-ready, scalable video analysis system that combines local vision AI models with cloud-based language models for intelligent video understanding. Built with Temporal workflow orchestration for robust parallel processing.

## 🏗️ Architecture Overview

### System Components

1. **Vision AI Processing** (`temporal_worker_gpu.py`)
   - Local Qwen2.5-VL-7B vision model for frame analysis
   - GPU-accelerated inference with automatic mixed precision
   - Detailed frame-by-frame descriptions with timestamp mapping

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
   - Intelligent request routing
   - GPU memory optimization with 4-bit quantization
   - Automatic fallback mechanisms

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
- NVIDIA GPU with 16GB+ VRAM (recommended: 24GB for multi-worker)
- CUDA 11.8+ and cuDNN
- 50GB+ free disk space for models

### 1. Clone and Navigate
```bash
git clone <repository>
cd video-analysis-system
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

### 3. Download Qwen Vision Model

The system will automatically download the model on first run, or manually:
```bash
# Using Hugging Face CLI
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/Qwen_Qwen2.5-VL-7B-Instruct
```

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

### Upload Video
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@video.mp4"
```

Response:
```json
{
  "session_id": "550e8400-e29b-41d4-a716",
  "message": "Video uploaded successfully", 
  "total_frames": 150,
  "workflow_id": "video-analysis-550e8400"
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

### Get Available Summaries
```bash
curl "http://localhost:8000/summaries/{session_id}"
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

| Workers | GPU VRAM | Recommended GPU |
|---------|----------|-----------------|
| 1       | 8-10 GB  | RTX 3080        |
| 2-3     | 16-20 GB | RTX 4090        |
| 4-5     | 24-32 GB | A5000/A6000     |
| 6-10    | 40-48 GB | A100            |

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
├── models/                     # Model files (auto-downloaded)
│   └── Qwen_Qwen2.5-VL-7B-Instruct/
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
- GitHub Issues: [Create Issue]
- Documentation: [Wiki]
- Contact: support@example.com
