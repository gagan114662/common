"""
GPU Acceleration and 384D Embeddings for Enterprise Trading System
High-performance computing with CUDA/OpenCL and advanced vector similarity
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# GPU computing libraries (with fallbacks)
try:
    import cupy as cp  # NVIDIA CUDA
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        TORCH_CUDA_AVAILABLE = True
    else:
        TORCH_CUDA_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

try:
    import faiss  # Facebook AI Similarity Search
    FAISS_AVAILABLE = True
    if hasattr(faiss, 'StandardGpuResources'):
        FAISS_GPU_AVAILABLE = True
    else:
        FAISS_GPU_AVAILABLE = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False

from tier1_core.logger import get_logger

class ComputeBackend(Enum):
    """Available compute backends"""
    CPU_NUMPY = "cpu_numpy"
    CUDA_CUPY = "cuda_cupy"
    CUDA_TORCH = "cuda_torch"
    OPENCL = "opencl"
    HYBRID = "hybrid"

class VectorOperation(Enum):
    """Vector operations that can be accelerated"""
    SIMILARITY_SEARCH = "similarity_search"
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    EMBEDDING_GENERATION = "embedding_generation"
    NEURAL_INFERENCE = "neural_inference"
    OPTIMIZATION = "optimization"
    CLUSTERING = "clustering"

@dataclass
class GPUDevice:
    """GPU device information"""
    device_id: int
    name: str
    memory_gb: float
    compute_capability: str
    backend: ComputeBackend
    is_available: bool
    utilization: float = 0.0

@dataclass
class EmbeddingModel:
    """384-dimensional embedding model configuration"""
    model_name: str
    embedding_dim: int = 384
    max_sequence_length: int = 512
    vocab_size: int = 50000
    model_weights: Optional[np.ndarray] = None
    gpu_optimized: bool = False

class GPUAccelerationEngine:
    """
    High-performance GPU acceleration engine
    
    Features:
    - CUDA acceleration with CuPy and PyTorch
    - FAISS GPU vector similarity search
    - 384-dimensional embeddings
    - Automatic CPU/GPU fallback
    - Memory optimization
    - Batch processing
    - Multi-GPU support
    """
    
    def __init__(self, preferred_backend: ComputeBackend = ComputeBackend.HYBRID):
        self.logger = get_logger(__name__)
        self.preferred_backend = preferred_backend
        
        # Device management
        self.available_devices: List[GPUDevice] = []
        self.active_backend = ComputeBackend.CPU_NUMPY
        self.primary_device = None
        
        # Vector similarity search
        self.faiss_index: Optional[Any] = None
        self.faiss_gpu_resources = None
        self.embedding_dimension = 384
        
        # Embedding models
        self.embedding_models: Dict[str, EmbeddingModel] = {}
        self.default_embedding_model = None
        
        # Performance monitoring
        self.operation_times: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, float] = {}
        
        # Threading for CPU fallback
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Memory management
        self.gpu_memory_limit = 0.8  # Use 80% of GPU memory
        self.batch_size_limit = 10000
        
    async def initialize(self) -> None:
        """Initialize GPU acceleration engine"""
        
        self.logger.info("🚀 Initializing GPU acceleration engine")
        
        # Detect available devices
        await self._detect_gpu_devices()
        
        # Initialize compute backend
        await self._initialize_compute_backend()
        
        # Initialize FAISS for vector similarity
        await self._initialize_faiss()
        
        # Load embedding models
        await self._initialize_embedding_models()
        
        # Optimize memory allocation
        await self._optimize_memory_allocation()
        
        self.logger.info(f"✅ GPU acceleration initialized with {self.active_backend.value}")
    
    async def _detect_gpu_devices(self) -> None:
        """Detect available GPU devices"""
        
        devices = []
        
        # Detect CUDA devices
        if CUDA_AVAILABLE:
            try:
                num_devices = cp.cuda.runtime.getDeviceCount()
                for i in range(num_devices):
                    with cp.cuda.Device(i):
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        memory_info = cp.cuda.runtime.memGetInfo()
                        
                        device = GPUDevice(
                            device_id=i,
                            name=props['name'].decode(),
                            memory_gb=memory_info[1] / (1024**3),
                            compute_capability=f"{props['major']}.{props['minor']}",
                            backend=ComputeBackend.CUDA_CUPY,
                            is_available=True
                        )
                        devices.append(device)
                        
                self.logger.info(f"Detected {len(devices)} CUDA devices")
                
            except Exception as e:
                self.logger.warning(f"CUDA detection failed: {str(e)}")
        
        # Detect PyTorch CUDA devices
        if TORCH_CUDA_AVAILABLE:
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    
                    # Check if already detected by CuPy
                    existing = any(d.device_id == i and d.backend == ComputeBackend.CUDA_CUPY for d in devices)
                    if not existing:
                        device = GPUDevice(
                            device_id=i,
                            name=props.name,
                            memory_gb=memory_gb,
                            compute_capability=f"{props.major}.{props.minor}",
                            backend=ComputeBackend.CUDA_TORCH,
                            is_available=True
                        )
                        devices.append(device)
                        
            except Exception as e:
                self.logger.warning(f"PyTorch CUDA detection failed: {str(e)}")
        
        # Fallback to CPU
        if not devices:
            cpu_device = GPUDevice(
                device_id=-1,
                name="CPU",
                memory_gb=8.0,  # Assume 8GB available RAM
                compute_capability="N/A",
                backend=ComputeBackend.CPU_NUMPY,
                is_available=True
            )
            devices.append(cpu_device)
        
        self.available_devices = devices
        
        # Select primary device
        if devices:
            # Prefer CUDA devices with most memory
            cuda_devices = [d for d in devices if d.backend in [ComputeBackend.CUDA_CUPY, ComputeBackend.CUDA_TORCH]]
            if cuda_devices:
                self.primary_device = max(cuda_devices, key=lambda d: d.memory_gb)
            else:
                self.primary_device = devices[0]
                
        self.logger.info(f"Primary device: {self.primary_device.name} ({self.primary_device.memory_gb:.1f}GB)")
    
    async def _initialize_compute_backend(self) -> None:
        """Initialize the compute backend"""
        
        if self.preferred_backend == ComputeBackend.HYBRID:
            # Choose best available backend
            if self.primary_device.backend in [ComputeBackend.CUDA_CUPY, ComputeBackend.CUDA_TORCH]:
                self.active_backend = self.primary_device.backend
            else:
                self.active_backend = ComputeBackend.CPU_NUMPY
        else:
            # Use preferred backend if available
            if any(d.backend == self.preferred_backend for d in self.available_devices):
                self.active_backend = self.preferred_backend
            else:
                self.logger.warning(f"Preferred backend {self.preferred_backend.value} not available, falling back to CPU")
                self.active_backend = ComputeBackend.CPU_NUMPY
        
        # Initialize backend-specific resources
        if self.active_backend == ComputeBackend.CUDA_CUPY and CUDA_AVAILABLE:
            cp.cuda.Device(self.primary_device.device_id).use()
            self.logger.info("CuPy CUDA backend initialized")
            
        elif self.active_backend == ComputeBackend.CUDA_TORCH and TORCH_CUDA_AVAILABLE:
            torch.cuda.set_device(self.primary_device.device_id)
            self.logger.info("PyTorch CUDA backend initialized")
    
    async def _initialize_faiss(self) -> None:
        """Initialize FAISS for vector similarity search"""
        
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, vector similarity search will be limited")
            return
        
        try:
            # Create FAISS index for 384-dimensional vectors
            if FAISS_GPU_AVAILABLE and self.active_backend in [ComputeBackend.CUDA_CUPY, ComputeBackend.CUDA_TORCH]:
                # GPU FAISS index
                self.faiss_gpu_resources = faiss.StandardGpuResources()
                
                # Create GPU index
                cpu_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
                self.faiss_index = faiss.index_cpu_to_gpu(
                    self.faiss_gpu_resources,
                    self.primary_device.device_id,
                    cpu_index
                )
                
                self.logger.info("FAISS GPU index initialized for 384D vectors")
                
            else:
                # CPU FAISS index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
                self.logger.info("FAISS CPU index initialized for 384D vectors")
                
        except Exception as e:
            self.logger.error(f"FAISS initialization failed: {str(e)}")
            self.faiss_index = None
    
    async def _initialize_embedding_models(self) -> None:
        """Initialize 384-dimensional embedding models"""
        
        # Initialize default transformer-style embedding model
        default_model = EmbeddingModel(
            model_name="trading_transformer_384d",
            embedding_dim=384,
            max_sequence_length=512,
            vocab_size=50000,
            gpu_optimized=self.active_backend != ComputeBackend.CPU_NUMPY
        )
        
        # Generate random weights (in production, load pre-trained weights)
        if self.active_backend == ComputeBackend.CUDA_CUPY and CUDA_AVAILABLE:
            # Generate weights on GPU
            default_model.model_weights = cp.random.normal(
                0, 0.02, (default_model.vocab_size, default_model.embedding_dim)
            ).astype(cp.float32)
        elif self.active_backend == ComputeBackend.CUDA_TORCH and TORCH_CUDA_AVAILABLE:
            # Generate weights as PyTorch tensors
            default_model.model_weights = torch.randn(
                default_model.vocab_size, default_model.embedding_dim,
                device=f"cuda:{self.primary_device.device_id}",
                dtype=torch.float32
            ) * 0.02
        else:
            # CPU weights
            default_model.model_weights = np.random.normal(
                0, 0.02, (default_model.vocab_size, default_model.embedding_dim)
            ).astype(np.float32)
        
        self.embedding_models["default"] = default_model
        self.default_embedding_model = default_model
        
        # Initialize strategy-specific embedding model
        strategy_model = EmbeddingModel(
            model_name="strategy_encoder_384d",
            embedding_dim=384,
            max_sequence_length=256,
            vocab_size=10000,  # Smaller vocab for strategy parameters
            gpu_optimized=default_model.gpu_optimized
        )
        
        # Generate strategy model weights
        if self.active_backend == ComputeBackend.CUDA_CUPY and CUDA_AVAILABLE:
            strategy_model.model_weights = cp.random.normal(
                0, 0.02, (strategy_model.vocab_size, strategy_model.embedding_dim)
            ).astype(cp.float32)
        elif self.active_backend == ComputeBackend.CUDA_TORCH and TORCH_CUDA_AVAILABLE:
            strategy_model.model_weights = torch.randn(
                strategy_model.vocab_size, strategy_model.embedding_dim,
                device=f"cuda:{self.primary_device.device_id}",
                dtype=torch.float32
            ) * 0.02
        else:
            strategy_model.model_weights = np.random.normal(
                0, 0.02, (strategy_model.vocab_size, strategy_model.embedding_dim)
            ).astype(np.float32)
        
        self.embedding_models["strategy"] = strategy_model
        
        self.logger.info(f"Initialized {len(self.embedding_models)} embedding models with 384D vectors")
    
    async def _optimize_memory_allocation(self) -> None:
        """Optimize GPU memory allocation"""
        
        if self.active_backend == ComputeBackend.CUDA_CUPY and CUDA_AVAILABLE:
            # Set CuPy memory pool
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=int(self.primary_device.memory_gb * self.gpu_memory_limit * 1024**3))
            
        elif self.active_backend == ComputeBackend.CUDA_TORCH and TORCH_CUDA_AVAILABLE:
            # Set PyTorch memory fraction
            torch.cuda.set_per_process_memory_fraction(
                self.gpu_memory_limit, 
                device=self.primary_device.device_id
            )
        
        self.logger.info(f"Memory allocation optimized ({self.gpu_memory_limit*100:.0f}% limit)")
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model_name: str = "default",
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate 384-dimensional embeddings for text inputs"""
        
        start_time = time.time()
        
        model = self.embedding_models.get(model_name, self.default_embedding_model)
        if batch_size is None:
            batch_size = min(len(texts), self.batch_size_limit)
        
        # Tokenize texts (simplified)
        token_ids = []
        for text in texts:
            # Simple hash-based tokenization (in production, use proper tokenizer)
            tokens = [hash(word) % model.vocab_size for word in text.split()]
            tokens = tokens[:model.max_sequence_length]  # Truncate
            tokens += [0] * (model.max_sequence_length - len(tokens))  # Pad
            token_ids.append(tokens)
        
        # Convert to appropriate tensor format
        if self.active_backend == ComputeBackend.CUDA_CUPY and CUDA_AVAILABLE:
            embeddings = await self._generate_embeddings_cupy(token_ids, model, batch_size)
        elif self.active_backend == ComputeBackend.CUDA_TORCH and TORCH_CUDA_AVAILABLE:
            embeddings = await self._generate_embeddings_torch(token_ids, model, batch_size)
        else:
            embeddings = await self._generate_embeddings_cpu(token_ids, model, batch_size)
        
        # Record performance
        elapsed = time.time() - start_time
        self._record_operation_time("generate_embeddings", elapsed)
        
        self.logger.debug(f"Generated {len(texts)} embeddings in {elapsed:.3f}s")
        
        return embeddings
    
    async def _generate_embeddings_cupy(
        self,
        token_ids: List[List[int]],
        model: EmbeddingModel,
        batch_size: int
    ) -> np.ndarray:
        """Generate embeddings using CuPy"""
        
        all_embeddings = []
        
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i + batch_size]
            
            # Convert to CuPy array
            batch_array = cp.array(batch, dtype=cp.int32)
            
            # Embedding lookup
            embeddings = model.model_weights[batch_array]  # Shape: (batch, seq_len, embed_dim)
            
            # Mean pooling
            batch_embeddings = cp.mean(embeddings, axis=1)  # Shape: (batch, embed_dim)
            
            # L2 normalization for cosine similarity
            norms = cp.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings = batch_embeddings / (norms + 1e-8)
            
            all_embeddings.append(cp.asnumpy(batch_embeddings))
        
        return np.vstack(all_embeddings)
    
    async def _generate_embeddings_torch(
        self,
        token_ids: List[List[int]],
        model: EmbeddingModel,
        batch_size: int
    ) -> np.ndarray:
        """Generate embeddings using PyTorch"""
        
        all_embeddings = []
        
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i + batch_size]
            
            # Convert to PyTorch tensor
            batch_tensor = torch.tensor(batch, dtype=torch.long, device=f"cuda:{self.primary_device.device_id}")
            
            # Embedding lookup
            embeddings = model.model_weights[batch_tensor]  # Shape: (batch, seq_len, embed_dim)
            
            # Mean pooling
            batch_embeddings = torch.mean(embeddings, dim=1)  # Shape: (batch, embed_dim)
            
            # L2 normalization
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    async def _generate_embeddings_cpu(
        self,
        token_ids: List[List[int]],
        model: EmbeddingModel,
        batch_size: int
    ) -> np.ndarray:
        """Generate embeddings using CPU"""
        
        all_embeddings = []
        
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i + batch_size]
            
            # Convert to numpy array
            batch_array = np.array(batch, dtype=np.int32)
            
            # Embedding lookup
            embeddings = model.model_weights[batch_array]  # Shape: (batch, seq_len, embed_dim)
            
            # Mean pooling
            batch_embeddings = np.mean(embeddings, axis=1)  # Shape: (batch, embed_dim)
            
            # L2 normalization
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings = batch_embeddings / (norms + 1e-8)
            
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    async def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform high-speed vector similarity search"""
        
        start_time = time.time()
        
        if self.faiss_index is None:
            raise ValueError("FAISS index not initialized")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure correct dimension
        if query_embedding.shape[1] != self.embedding_dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} != {self.embedding_dimension}")
        
        # Normalize query for cosine similarity
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / (query_norm + 1e-8)
        
        # Perform search
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), k)
        
        # Filter by threshold
        mask = scores[0] >= threshold
        filtered_scores = scores[0][mask]
        filtered_indices = indices[0][mask]
        
        # Record performance
        elapsed = time.time() - start_time
        self._record_operation_time("similarity_search", elapsed)
        
        return filtered_scores, filtered_indices
    
    async def add_vectors_to_index(self, embeddings: np.ndarray) -> None:
        """Add vectors to the similarity search index"""
        
        if self.faiss_index is None:
            raise ValueError("FAISS index not initialized")
        
        # Ensure correct dimension
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != {self.embedding_dimension}")
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        # Add to index
        self.faiss_index.add(normalized_embeddings.astype(np.float32))
        
        self.logger.debug(f"Added {len(embeddings)} vectors to index")
    
    async def accelerated_matrix_multiply(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """Accelerated matrix multiplication"""
        
        start_time = time.time()
        
        if self.active_backend == ComputeBackend.CUDA_CUPY and CUDA_AVAILABLE:
            # CuPy matrix multiplication
            a_gpu = cp.asarray(a)
            b_gpu = cp.asarray(b)
            result_gpu = cp.dot(a_gpu, b_gpu)
            result = cp.asnumpy(result_gpu)
            
        elif self.active_backend == ComputeBackend.CUDA_TORCH and TORCH_CUDA_AVAILABLE:
            # PyTorch matrix multiplication
            a_tensor = torch.from_numpy(a).to(f"cuda:{self.primary_device.device_id}")
            b_tensor = torch.from_numpy(b).to(f"cuda:{self.primary_device.device_id}")
            result_tensor = torch.matmul(a_tensor, b_tensor)
            result = result_tensor.cpu().numpy()
            
        else:
            # CPU NumPy
            result = np.dot(a, b)
        
        elapsed = time.time() - start_time
        self._record_operation_time("matrix_multiply", elapsed)
        
        return result
    
    def _record_operation_time(self, operation: str, elapsed_time: float) -> None:
        """Record operation timing for performance analysis"""
        
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        
        self.operation_times[operation].append(elapsed_time)
        
        # Keep only recent times (last 100)
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation] = self.operation_times[operation][-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get GPU acceleration performance statistics"""
        
        stats = {
            "active_backend": self.active_backend.value,
            "primary_device": self.primary_device.name if self.primary_device else "None",
            "embedding_dimension": self.embedding_dimension,
            "available_models": list(self.embedding_models.keys()),
            "faiss_enabled": self.faiss_index is not None,
            "operation_times": {}
        }
        
        # Compute operation statistics
        for operation, times in self.operation_times.items():
            if times:
                stats["operation_times"][operation] = {
                    "count": len(times),
                    "mean_ms": np.mean(times) * 1000,
                    "std_ms": np.std(times) * 1000,
                    "min_ms": np.min(times) * 1000,
                    "max_ms": np.max(times) * 1000
                }
        
        # GPU utilization if available
        if self.primary_device and self.primary_device.backend != ComputeBackend.CPU_NUMPY:
            try:
                if CUDA_AVAILABLE:
                    gpu_memory = cp.cuda.runtime.memGetInfo()
                    stats["gpu_memory_used_gb"] = (gpu_memory[1] - gpu_memory[0]) / (1024**3)
                    stats["gpu_memory_total_gb"] = gpu_memory[1] / (1024**3)
                    stats["gpu_memory_utilization"] = (gpu_memory[1] - gpu_memory[0]) / gpu_memory[1]
            except:
                pass
        
        return stats
    
    async def cleanup(self) -> None:
        """Cleanup GPU resources"""
        
        self.logger.info("Cleaning up GPU acceleration resources")
        
        # Clear FAISS resources
        if self.faiss_gpu_resources:
            del self.faiss_gpu_resources
        
        # Clear memory pools
        if self.active_backend == ComputeBackend.CUDA_CUPY and CUDA_AVAILABLE:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            
        elif self.active_backend == ComputeBackend.CUDA_TORCH and TORCH_CUDA_AVAILABLE:
            torch.cuda.empty_cache()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("GPU acceleration cleanup complete")