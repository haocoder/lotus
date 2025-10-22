"""GPU-accelerated FAISS vector store implementation."""

import logging
import os
import pickle
from typing import Any, Optional

import faiss
import numpy as np
from numpy.typing import NDArray

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """Manages GPU resources for FAISS operations."""
    
    def __init__(self) -> None:
        """Initialize GPU resource manager."""
        self._gpu_available: Optional[bool] = None
        self._gpu_resources: Optional[faiss.StandardGpuResources] = None
        self._gpu_count: int = 0
        self._current_gpu_id: int = 0
        
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for FAISS operations."""
        if self._gpu_available is None:
            try:
                self._gpu_count = faiss.get_num_gpus()
                self._gpu_available = self._gpu_count > 0
                if self._gpu_available:
                    logger.info(f"Found {self._gpu_count} GPU(s) available for FAISS")
                else:
                    logger.info("No GPUs available, falling back to CPU")
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}, falling back to CPU")
                self._gpu_available = False
        return self._gpu_available
    
    def get_gpu_resources(self) -> Optional[faiss.StandardGpuResources]:
        """Get GPU resources for FAISS operations."""
        if not self.is_gpu_available():
            return None
            
        if self._gpu_resources is None:
            try:
                self._gpu_resources = faiss.StandardGpuResources()
                # Set memory pool to use 75% of available GPU memory
                # This leaves room for other operations
                self._gpu_resources.setDefaultNullStreamAllTempMemory(int(0.75 * 1024 * 1024 * 1024))  # 768MB default
                logger.info("Initialized GPU resources for FAISS")
            except Exception as e:
                logger.error(f"Failed to initialize GPU resources: {e}")
                self._gpu_resources = None
                self._gpu_available = False
        return self._gpu_resources
    
    def get_gpu_id(self) -> int:
        """Get current GPU ID for operations."""
        return self._current_gpu_id
    
    def set_gpu_id(self, gpu_id: int) -> None:
        """Set GPU ID for operations."""
        if 0 <= gpu_id < self._gpu_count:
            self._current_gpu_id = gpu_id
            logger.info(f"Set GPU ID to {gpu_id}")
        else:
            logger.warning(f"Invalid GPU ID {gpu_id}, keeping current GPU {self._current_gpu_id}")


class FaissGPUVS(VS):
    """GPU-accelerated FAISS vector store with automatic CPU fallback."""
    
    def __init__(
        self, 
        factory_string: str = "Flat", 
        metric: int = faiss.METRIC_INNER_PRODUCT,
        use_gpu: bool = True,
        gpu_id: int = 0,
        batch_size: int = 10000
    ) -> None:
        """
        Initialize GPU-accelerated FAISS vector store.
        
        Args:
            factory_string: FAISS index factory string
            metric: Distance metric to use
            use_gpu: Whether to attempt GPU acceleration
            gpu_id: GPU device ID to use
            batch_size: Batch size for GPU operations to manage memory
        """
        super().__init__()
        self.factory_string = factory_string
        self.metric = metric
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.index_dir: Optional[str] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.vecs: Optional[NDArray[np.float64]] = None
        
        # Initialize GPU resource manager
        self.gpu_manager = GPUResourceManager()
        self.gpu_manager.set_gpu_id(gpu_id)
        
        # Track whether current index is on GPU
        self._is_gpu_index = False
        
    def _create_index(self, dimension: int, use_gpu: bool = None) -> faiss.Index:
        """
        Create FAISS index with optional GPU acceleration.
        
        Args:
            dimension: Vector dimension
            use_gpu: Whether to use GPU (None for auto-detect)
            
        Returns:
            FAISS index (GPU or CPU)
        """
        if use_gpu is None:
            use_gpu = self.use_gpu
            
        # Create CPU index first
        cpu_index = faiss.index_factory(dimension, self.factory_string, self.metric)
        
        # Try to move to GPU if requested and available
        if use_gpu and self.gpu_manager.is_gpu_available():
            try:
                gpu_resources = self.gpu_manager.get_gpu_resources()
                if gpu_resources is not None:
                    gpu_index = faiss.index_cpu_to_gpu(
                        gpu_resources, 
                        self.gpu_manager.get_gpu_id(), 
                        cpu_index
                    )
                    self._is_gpu_index = True
                    logger.info(f"Created GPU index on GPU {self.gpu_manager.get_gpu_id()}")
                    return gpu_index
            except Exception as e:
                logger.warning(f"Failed to create GPU index, falling back to CPU: {e}")
        
        self._is_gpu_index = False
        logger.info("Using CPU index")
        return cpu_index
    
    def _batch_add_vectors(self, index: faiss.Index, vectors: NDArray[np.float64]) -> None:
        """
        Add vectors to index in batches to manage GPU memory.
        
        Args:
            index: FAISS index
            vectors: Vectors to add
        """
        if self._is_gpu_index and len(vectors) > self.batch_size:
            # Process in batches for GPU to avoid memory issues
            for i in range(0, len(vectors), self.batch_size):
                batch = vectors[i:i + self.batch_size]
                index.add(batch.astype(np.float32))
                logger.debug(f"Added batch {i//self.batch_size + 1}/{(len(vectors) + self.batch_size - 1)//self.batch_size}")
        else:
            # Add all at once for CPU or small datasets
            index.add(vectors.astype(np.float32))
            
    def index(self, docs: list[str], embeddings: NDArray[np.float64], index_dir: str, **kwargs: dict[str, Any]) -> None:
        """
        Create and store GPU-accelerated index.
        
        Args:
            docs: List of documents
            embeddings: Document embeddings
            index_dir: Directory to save index
            **kwargs: Additional arguments
        """
        try:
            # Create index with GPU acceleration
            self.faiss_index = self._create_index(embeddings.shape[1])
            
            # Add vectors in batches
            self._batch_add_vectors(self.faiss_index, embeddings)
            
            self.index_dir = index_dir
            
            # Save index and vectors
            os.makedirs(index_dir, exist_ok=True)
            
            # Save vectors as numpy array
            with open(f"{index_dir}/vecs", "wb") as fp:
                pickle.dump(embeddings, fp)
            
            # Save index (move to CPU first if on GPU to ensure compatibility)
            index_to_save = self.faiss_index
            if self._is_gpu_index:
                try:
                    index_to_save = faiss.index_gpu_to_cpu(self.faiss_index)
                except Exception as e:
                    logger.warning(f"Failed to move index to CPU for saving: {e}")
            
            faiss.write_index(index_to_save, f"{index_dir}/index")
            
            # Save GPU configuration
            gpu_config = {
                'is_gpu_index': self._is_gpu_index,
                'gpu_id': self.gpu_manager.get_gpu_id(),
                'factory_string': self.factory_string,
                'metric': self.metric
            }
            with open(f"{index_dir}/gpu_config.pkl", "wb") as fp:
                pickle.dump(gpu_config, fp)
                
            logger.info(f"Indexed {len(embeddings)} vectors with {'GPU' if self._is_gpu_index else 'CPU'} acceleration")
            
        except Exception as e:
            logger.error(f"Failed to create GPU index: {e}")
            # Fallback to CPU
            self._create_cpu_fallback_index(docs, embeddings, index_dir, **kwargs)
    
    def _create_cpu_fallback_index(self, docs: list[str], embeddings: NDArray[np.float64], index_dir: str, **kwargs: dict[str, Any]) -> None:
        """Create CPU fallback index when GPU fails."""
        logger.info("Creating CPU fallback index")
        self.faiss_index = self._create_index(embeddings.shape[1], use_gpu=False)
        self.faiss_index.add(embeddings.astype(np.float32))
        self.index_dir = index_dir
        
        os.makedirs(index_dir, exist_ok=True)
        with open(f"{index_dir}/vecs", "wb") as fp:
            pickle.dump(embeddings, fp)
        faiss.write_index(self.faiss_index, f"{index_dir}/index")
    
    def load_index(self, index_dir: str) -> None:
        """
        Load index with GPU acceleration if available.
        
        Args:
            index_dir: Directory containing the index
        """
        self.index_dir = index_dir
        
        try:
            # Load GPU configuration if available
            gpu_config_path = f"{index_dir}/gpu_config.pkl"
            use_gpu = self.use_gpu
            
            if os.path.exists(gpu_config_path):
                with open(gpu_config_path, "rb") as fp:
                    gpu_config = pickle.load(fp)
                    logger.info(f"Loaded GPU config: {gpu_config}")
            
            # Load CPU index first
            cpu_index = faiss.read_index(f"{index_dir}/index")
            
            # Try to move to GPU if requested and available
            if use_gpu and self.gpu_manager.is_gpu_available():
                try:
                    gpu_resources = self.gpu_manager.get_gpu_resources()
                    if gpu_resources is not None:
                        self.faiss_index = faiss.index_cpu_to_gpu(
                            gpu_resources, 
                            self.gpu_manager.get_gpu_id(), 
                            cpu_index
                        )
                        self._is_gpu_index = True
                        logger.info(f"Loaded index to GPU {self.gpu_manager.get_gpu_id()}")
                    else:
                        self.faiss_index = cpu_index
                        self._is_gpu_index = False
                except Exception as e:
                    logger.warning(f"Failed to load index to GPU, using CPU: {e}")
                    self.faiss_index = cpu_index
                    self._is_gpu_index = False
            else:
                self.faiss_index = cpu_index
                self._is_gpu_index = False
            
            # Load vectors
            with open(f"{index_dir}/vecs", "rb") as fp:
                self.vecs = pickle.load(fp)
                
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64]:
        """
        Retrieve vectors from stored index.
        
        Args:
            index_dir: Index directory
            ids: Vector IDs to retrieve
            
        Returns:
            Retrieved vectors
        """
        with open(f"{index_dir}/vecs", "rb") as fp:
            vecs: NDArray[np.float64] = pickle.load(fp)
        return vecs[ids]
    
    def __call__(
        self, 
        query_vectors: NDArray[np.float64], 
        K: int, 
        ids: Optional[list[int]] = None, 
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """
        Perform GPU-accelerated nearest neighbor search.
        
        Args:
            query_vectors: Query vectors
            K: Number of nearest neighbors
            ids: Optional list of IDs to search within
            **kwargs: Additional arguments
            
        Returns:
            Search results
        """
        if self.faiss_index is None or self.index_dir is None:
            raise ValueError("Index not loaded")
        
        try:
            if ids is not None:
                # For subset search, use temporary index approach
                # This could be optimized further with GPU-specific subset search
                subset_vecs = self.get_vectors_from_index(self.index_dir, ids)
                
                # Create temporary index (prefer GPU if available)
                tmp_index = self._create_index(subset_vecs.shape[1])
                self._batch_add_vectors(tmp_index, subset_vecs)
                
                # Perform search
                query_vectors_f32 = query_vectors.astype(np.float32)
                distances, sub_indices = tmp_index.search(query_vectors_f32, K)
                
                # Remap indices
                subset_ids = np.array(ids)
                indices = np.array([subset_ids[sub_indices[i]] for i in range(len(sub_indices))]).tolist()
            else:
                # Full index search
                query_vectors_f32 = query_vectors.astype(np.float32)
                distances, indices = self.faiss_index.search(query_vectors_f32, K)
            
            return RMOutput(distances=distances, indices=indices)  # type: ignore
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # If GPU search fails, try CPU fallback
            if self._is_gpu_index:
                logger.info("Attempting CPU fallback for search")
                return self._cpu_fallback_search(query_vectors, K, ids, **kwargs)
            raise
    
    def _cpu_fallback_search(
        self, 
        query_vectors: NDArray[np.float64], 
        K: int, 
        ids: Optional[list[int]] = None, 
        **kwargs: dict[str, Any]
    ) -> RMOutput:
        """CPU fallback search when GPU search fails."""
        try:
            # Move index to CPU
            cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            
            if ids is not None:
                subset_vecs = self.get_vectors_from_index(self.index_dir, ids)
                tmp_index = faiss.index_factory(subset_vecs.shape[1], self.factory_string, self.metric)
                tmp_index.add(subset_vecs.astype(np.float32))
                distances, sub_indices = tmp_index.search(query_vectors.astype(np.float32), K)
                subset_ids = np.array(ids)
                indices = np.array([subset_ids[sub_indices[i]] for i in range(len(sub_indices))]).tolist()
            else:
                distances, indices = cpu_index.search(query_vectors.astype(np.float32), K)
            
            return RMOutput(distances=distances, indices=indices)  # type: ignore
        except Exception as e:
            logger.error(f"CPU fallback search also failed: {e}")
            raise
    
    def get_memory_info(self) -> dict[str, Any]:
        """Get GPU memory usage information."""
        info = {
            'is_gpu_index': self._is_gpu_index,
            'gpu_available': self.gpu_manager.is_gpu_available(),
            'gpu_count': self.gpu_manager._gpu_count,
            'current_gpu_id': self.gpu_manager.get_gpu_id()
        }
        
        if self._is_gpu_index and self.gpu_manager.is_gpu_available():
            try:
                gpu_resources = self.gpu_manager.get_gpu_resources()
                if gpu_resources is not None:
                    # Note: GPU memory info might require additional CUDA calls
                    info['gpu_memory_info'] = 'GPU memory tracking not implemented'
            except Exception as e:
                info['gpu_memory_error'] = str(e)
        
        return info
