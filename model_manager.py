"""
Multi-Model Manager for orchestrating Nebius GPT and Local Qwen models
Optimized for GPU memory management and performance
"""
import os
import gc
import torch
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
import json
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum for different model types"""
    NEBIUS_GPT = "nebius_gpt"
    QWEN_VISION = "qwen_vision"
    QWEN_CHAT = "qwen_chat"
    QWEN_STRUCTURED = "qwen_structured"


@dataclass
class ModelConfig:
    """Configuration for each model"""
    name: str
    type: ModelType
    endpoint: Optional[str] = None
    model_id: Optional[str] = None
    device: str = "cuda"
    max_memory: Optional[str] = None  # e.g., "8GB"
    precision: str = "float16"
    cache_dir: str = "./model_cache"


class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup"""
    
    def __init__(self, reserved_memory_gb: float = 1.0):
        self.reserved_memory = reserved_memory_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.current_model = None
        
    @contextmanager
    def managed_memory(self):
        """Context manager for GPU memory management"""
        try:
            yield
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Force GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"GPU memory cleaned. Available: {self.get_available_memory():.2f}GB")
    
    def get_available_memory(self) -> float:
        """Get available GPU memory in GB"""
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / (1024**3)
        return 0.0
    
    def can_load_model(self, required_memory_gb: float) -> bool:
        """Check if model can be loaded"""
        available = self.get_available_memory()
        return available > (required_memory_gb + self.reserved_memory / (1024**3))


class NebiusGPTClient:
    """Client for Nebius GPT API using OpenAI SDK"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.endpoint = config.endpoint or os.getenv("NEBIUS_GPT_ENDPOINT")
        self.api_key = os.getenv("NEBIUS_API_KEY")
        
        if not self.endpoint:
            raise ValueError("Nebius GPT endpoint not configured")
        if not self.api_key:
            raise ValueError("Nebius API key not configured")
            
        # Initialize OpenAI client with Nebius endpoint
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key
        )
            
    async def generate(self, prompt: str, images=None, **kwargs) -> str:
        """Generate response from Nebius using OpenAI SDK (supports both text and vision)"""
        try:
            # Determine model based on whether images are provided
            if images and len(images) > 0:
                model = "Qwen/Qwen2.5-VL-72B-Instruct"
                # Prepare vision message with images
                content = [{"type": "text", "text": prompt}]
                
                # Add images to content
                for image in images:
                    if hasattr(image, 'save'):  # PIL Image
                        import io
                        import base64
                        buffer = io.BytesIO()
                        image.save(buffer, format='JPEG')
                        image_data = base64.b64encode(buffer.getvalue()).decode()
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        })
                
                messages = [{"role": "user", "content": content}]
            else:
                # Text-only model for Q&A
                model = "openai/gpt-oss-120b"
                messages = [{"role": "user", "content": prompt}]
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Nebius API error (model: {model if 'model' in locals() else 'unknown'}): {e}")
            raise


class QwenModelWrapper:
    """Wrapper for local Qwen models with optimizations"""
    
    def __init__(self, config: ModelConfig, memory_manager: GPUMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.model = None
        self.tokenizer = None
        self.processor = None  # For vision models
        
    def load_model(self):
        """Load Qwen model with optimizations"""
        from transformers import AutoTokenizer, BitsAndBytesConfig
        
        logger.info(f"Loading {self.config.name}...")
        
        # Check available memory
        required_memory = self._estimate_model_memory()
        if not self.memory_manager.can_load_model(required_memory):
            raise RuntimeError(f"Insufficient GPU memory for {self.config.name}")
        
        # Quantization config for memory optimization
        quantization_config = None
        if self.config.precision == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        elif self.config.precision == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Model loading kwargs
        model_kwargs = {
            "cache_dir": self.config.cache_dir,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.config.precision == "float16" else torch.float32,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Handle different model types
        if self.config.type == ModelType.QWEN_VISION:
            # Use Auto classes to automatically detect the correct architecture
            # This handles both Qwen2-VL and Qwen2.5-VL models correctly
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                
                # Load processor for vision models
                self.processor = AutoProcessor.from_pretrained(
                    self.config.model_id,
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=True
                )
                
                # Load the vision-language model
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.config.model_id,
                    **model_kwargs
                )
                
                # Processor handles tokenization for VL models
                self.tokenizer = self.processor.tokenizer
                
            except Exception as e:
                logger.error(f"Failed to load vision model with Auto classes: {e}")
                # Fallback: Try older Qwen vision models approach
                from transformers import AutoModelForCausalLM, AutoProcessor
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_id,
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    **model_kwargs
                )
                
                self.processor = AutoProcessor.from_pretrained(
                    self.config.model_id,
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=True
                )
        else:
            # Standard text models
            from transformers import AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )
        
        logger.info(f"Model {self.config.name} loaded successfully")
        return self
    
    def unload_model(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self.processor:
            del self.processor
            self.processor = None
        self.memory_manager.cleanup()
        logger.info(f"Model {self.config.name} unloaded")
    
    def _estimate_model_memory(self) -> float:
        """Estimate required memory in GB"""
        # Rough estimates based on model size and precision
        model_sizes = {
            "Qwen/Qwen-VL-Chat": 9.0,  # 9B params
            "Qwen/Qwen-7B-Chat": 7.0,  # 7B params
            "Qwen/Qwen-14B-Chat": 14.0,  # 14B params
            "Qwen/Qwen2-VL-2B-Instruct": 2.5,  # 2B params
            "Qwen/Qwen2-VL-7B-Instruct": 7.5,  # 7B params
            "Qwen/Qwen2.5-VL-7B-Instruct": 8.0,  # 7B params, newer architecture
        }
        
        base_size = model_sizes.get(self.config.model_id, 7.0)
        
        # Adjust for precision
        if self.config.precision == "int8":
            return base_size * 0.5
        elif self.config.precision == "int4":
            return base_size * 0.25
        return base_size
    
    async def generate(self, 
                       prompt: str = None,
                       images: List = None,
                       max_new_tokens: int = 512,
                       temperature: float = 0.7,
                       **kwargs) -> str:
        """Generate response from Qwen model"""
        
        if not self.model:
            raise RuntimeError(f"Model {self.config.name} not loaded")
        
        # Run synchronous GPU operations in thread pool to avoid blocking
        import asyncio
        import concurrent.futures
        
        def _sync_generate():
            with torch.no_grad():
                if self.config.type == ModelType.QWEN_VISION and images:
                    # Handle Qwen2.5-VL and Qwen2-VL models differently
                    if "Qwen2.5-VL" in self.config.model_id or "Qwen2-VL" in self.config.model_id:
                        # Simple approach for Qwen2-VL models
                        # Format as conversation
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt}
                                ]
                            }
                        ]
                        
                        # Add image to conversation if provided
                        if images and len(images) > 0:
                            # Insert image before text in content
                            conversation[0]["content"].insert(0, {"type": "image"})
                        
                        # Apply chat template
                        text = self.processor.apply_chat_template(
                            conversation, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        
                        # Process with the processor
                        inputs = self.processor(
                            text=text,
                            images=images[0] if images else None,
                            return_tensors="pt"
                        ).to(self.config.device)
                    else:
                        # Older Qwen vision models
                        inputs = self.processor(
                            text=prompt,
                            images=images,
                            return_tensors="pt"
                        ).to(self.config.device)
                else:
                    # Text-only input
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048
                    ).to(self.config.device)
                
                # Generate with memory-efficient settings
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                }
                
                # Add tokenizer-specific tokens if available
                if hasattr(self.tokenizer, 'pad_token_id'):
                    generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                if hasattr(self.tokenizer, 'eos_token_id'):
                    generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
                
                generation_kwargs.update(kwargs)
                
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
                
                # Decode response - handle different output formats
                if "input_ids" in inputs:
                    # Skip the input tokens when decoding
                    generated_ids = outputs[:, inputs['input_ids'].shape[-1]:]
                else:
                    # For models that don't have input_ids in the same format
                    generated_ids = outputs
                
                response = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                return response
        
        # Run sync GPU operations in thread pool executor
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, _sync_generate)


class ModelOrchestrator:
    """Main orchestrator for multiple models"""
    
    def __init__(self):
        self.models: Dict[str, Union[NebiusGPTClient, QwenModelWrapper]] = {}
        self.memory_manager = GPUMemoryManager(reserved_memory_gb=1.0)
        self.configs: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        
    def register_model(self, config: ModelConfig):
        """Register a model configuration"""
        self.configs[config.name] = config
        logger.info(f"Registered model: {config.name}")
        
    def load_model(self, name: str):
        """Load a specific model"""
        if name not in self.configs:
            raise ValueError(f"Model {name} not registered")
            
        config = self.configs[name]
        
        # Check if we need to unload current model (for local models)
        if self.active_model and config.type != ModelType.NEBIUS_GPT:
            if self.active_model != name:
                self.unload_model(self.active_model)
        
        # Load appropriate model
        if config.type == ModelType.NEBIUS_GPT:
            self.models[name] = NebiusGPTClient(config)
        else:
            wrapper = QwenModelWrapper(config, self.memory_manager)
            self.models[name] = wrapper.load_model()
            self.active_model = name
            
        return self.models[name]
    
    def unload_model(self, name: str):
        """Unload a specific model"""
        if name in self.models:
            model = self.models[name]
            if isinstance(model, QwenModelWrapper):
                model.unload_model()
            del self.models[name]
            if self.active_model == name:
                self.active_model = None
                
    async def route_request(self, 
                           request_type: str,
                           content: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate model: Local Qwen for vision, Cloud GPT for chat"""
        
        # Determine which model to use based on request type and content
        if request_type == "vision" or content.get("images"):
            # Use local Qwen2.5-VL-7B-Instruct for vision tasks
            model_name = "qwen_vision_local"
            processing_type = "local_gpu"
            model_used = "Qwen2.5-VL-7B-Instruct-Local"
        else:
            # Use cloud GPT-OSS-120B for chat/Q&A tasks
            model_name = "gpt_oss_120b"
            processing_type = "cloud_api"
            model_used = "GPT-OSS-120B"
            
        # Load model if not loaded
        if model_name not in self.models:
            self.load_model(model_name)
            
        model = self.models[model_name]
        
        # Process request using appropriate model
        response = await model.generate(
            prompt=content.get("prompt", ""),
            images=content.get("images"),
            **content.get("params", {})
        )
            
        return {
            "model": f"{model_name}_{request_type}",
            "response": response,
            "metadata": {
                "processing_type": processing_type,
                "model_used": model_used
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "registered_models": list(self.configs.keys()),
            "loaded_models": list(self.models.keys()),
            "active_model": self.active_model,
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_available_gb": self.memory_manager.get_available_memory()
        }


if __name__ == "__main__":
    # Example usage
    orchestrator = ModelOrchestrator()
    
    # Register models
    orchestrator.register_model(ModelConfig(
        name="nebius_gpt",
        type=ModelType.NEBIUS_GPT,
        endpoint="https://api.nebius.ai/v1/completions"  # Update with actual endpoint
    ))
    
    orchestrator.register_model(ModelConfig(
        name="qwen_vision",
        type=ModelType.QWEN_VISION,
        model_id="Qwen/Qwen-VL-Chat",
        precision="int8",
        cache_dir="./models"
    ))
    
    print(json.dumps(orchestrator.get_status(), indent=2))
