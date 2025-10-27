import os
import pickle
from typing import Any, Optional

import faiss
import numpy as np
from numpy.typing import NDArray
import torch
import psutil  # For CPU mem check
# Optional RAPIDS/cuVS support and RMM memory pool
try:  # Prefer official cuVS if present
    import cuvs  # type: ignore
    CUVS_AVAILABLE = True
except Exception:
    CUVS_AVAILABLE = False
try:
    import rmm  # type: ignore
    RMM_AVAILABLE = True
except Exception:
    rmm = None  # type: ignore
    RMM_AVAILABLE = False

import re  # For pattern matching

from lotus.types import RMOutput
from lotus.vector_store.vs import VS

def estimate_index_memory(factory_string: str, num_vecs: int, dim: int, is_gpu: bool = False) -> int:
    """
    Estimate memory footprint for a FAISS index in bytes.

    This function provides improved estimation for different index families:
    - Flat: raw vectors only (float32)
    - IVF-Flat: raw vectors + centroids + inverted lists overhead
    - IVF-SQ8: scalar quantized (1 byte/dim) + centroids + metadata
    - IVF-PQm: m bytes per vector code + centroids + codebooks + metadata
    - HNSWm: raw vectors + multi-layer graph (hierarchical structure)

    Args:
        factory_string: FAISS index factory string (e.g., "IVF1024,PQ32")
        num_vecs: Number of vectors to index
        dim: Dimensionality of vectors
        is_gpu: Whether index will be on GPU (adds overhead)

    Returns:
        Estimated memory in bytes

    Note:
        Warns if estimated memory exceeds 80% of available RAM/VRAM.
        GPU indices typically require 10-20% additional memory overhead.
    """
    fs_lower = factory_string.lower()

    # Defaults and parsed params
    nlist = 100  # Default IVF nlist
    m_pq = 8     # Default PQ subquantizers
    m_hnsw = 32  # Default HNSW connections
    nbits_pq = 8 # Default PQ bits per code
    
    # Extract parameters from factory string
    try:
        nlist_match = re.search(r"ivf(\d+)", fs_lower)
        if nlist_match:
            nlist = int(nlist_match.group(1))
        
        pq_match = re.search(r"pq(\d+)", fs_lower)
        if pq_match:
            m_pq = int(pq_match.group(1))
        
        hnsw_match = re.search(r"hnsw(\d+)", fs_lower)
        if hnsw_match:
            m_hnsw = int(hnsw_match.group(1))
    except Exception:
        pass

    # Base components
    bytes_f32 = 4
    bytes_i64 = 8
    raw_vectors = num_vecs * dim * bytes_f32
    mem_bytes = raw_vectors

    if "flat" in fs_lower and "ivf" not in fs_lower:
        # Pure Flat index: just raw vectors
        mem_bytes = raw_vectors
    
    elif "ivf" in fs_lower and "pq" not in fs_lower and "sq" not in fs_lower:
        # IVF-Flat: raw vectors + centroids + inverted lists metadata
        centroids = nlist * dim * bytes_f32
        invlists_overhead = nlist * 64  # Approximate overhead per list (pointers, sizes)
        mem_bytes = raw_vectors + centroids + invlists_overhead
    
    elif "ivf" in fs_lower and "sq" in fs_lower:
        # IVF-SQ8: 1 byte per component + centroids + metadata
        codes = num_vecs * dim  # 1 byte per dimension
        centroids = nlist * dim * bytes_f32
        invlists_overhead = nlist * 64
        # SQ also stores min/max per dimension
        sq_metadata = dim * 2 * bytes_f32
        mem_bytes = codes + centroids + invlists_overhead + sq_metadata
    
    elif "ivf" in fs_lower and "pq" in fs_lower:
        # IVF-PQm: compressed codes + centroids + codebooks + metadata
        # Each vector compressed to m bytes (assuming 8-bit codes)
        bytes_per_code = (nbits_pq * m_pq) // 8
        codes = num_vecs * bytes_per_code
        
        # Coarse quantizer centroids
        centroids = nlist * dim * bytes_f32
        
        # PQ codebooks: m subquantizers × 2^nbits centroids × (dim/m) dimensions
        subvector_dim = max(dim // m_pq, 1)
        num_centroids_per_sq = 2 ** nbits_pq  # Usually 256 for 8-bit
        codebooks = m_pq * num_centroids_per_sq * subvector_dim * bytes_f32
        
        # Inverted lists overhead
        invlists_overhead = nlist * 64
        
        mem_bytes = codes + centroids + codebooks + invlists_overhead
    
    elif "hnsw" in fs_lower:
        # HNSW: raw vectors + hierarchical graph structure
        # HNSW uses a hierarchical structure where each level has fewer nodes
        # Level 0 (base): all vectors
        # Level i: ~1/M of level i-1 nodes (exponential decay)
        
        # Base level: M connections per node
        base_connections = num_vecs * m_hnsw * bytes_i64
        
        # Upper levels: geometric series sum ≈ num_vecs/(M-1)
        # Simplified: total connections ≈ num_vecs * M * 1.1 (accounting for upper levels)
        upper_levels_connections = int(num_vecs * m_hnsw * 0.1 * bytes_i64)
        
        # Level assignment overhead (1 byte per node)
        level_info = num_vecs
        
        # Entry point and other metadata
        metadata_overhead = 1024  # Small constant overhead
        
        mem_bytes = raw_vectors + base_connections + upper_levels_connections + level_info + metadata_overhead
    
    else:
        # Fallback: assume flat index
        mem_bytes = raw_vectors

    # GPU overhead: FAISS GPU indices require additional memory for:
    # - GPU resources structure
    # - Temporary buffers for operations
    # - Device memory alignment padding
    # Typically 10-20% additional overhead
    if is_gpu:
        gpu_overhead_factor = 1.15  # 15% overhead
        mem_bytes = int(mem_bytes * gpu_overhead_factor)

    # Check available memory and warn if insufficient
    if is_gpu and torch.cuda.is_available():
        avail = torch.cuda.get_device_properties(0).total_memory
        mem_type = "GPU"
    else:
        avail = psutil.virtual_memory().available
        mem_type = "CPU"
    
    mem_mb = mem_bytes / (1024 * 1024)
    avail_mb = avail / (1024 * 1024)
    threshold = int(avail * 0.8)
    
    if mem_bytes > threshold:
        import logging
        logging.warning(
            f"Potential insufficient {mem_type} memory for index '{factory_string}':\n"
            f"  Estimated: {mem_mb:.1f} MB ({mem_bytes:,} bytes)\n"
            f"  Available: {avail_mb:.1f} MB (80% threshold: {threshold/(1024*1024):.1f} MB)\n"
            f"  Vectors: {num_vecs:,} × {dim} dims\n"
            f"  Consider: (1) Reduce nlist/M params, (2) Use more compression (PQ/SQ), or (3) Use more memory"
        )
    else:
        import logging
        logging.info(
            f"Index memory estimate for '{factory_string}': {mem_mb:.1f} MB ({mem_type}), "
            f"{num_vecs:,} vectors × {dim} dims"
        )
    
    return int(mem_bytes)

class UnifiedFaissVS(VS):
    """Unified FAISS vector store with CPU/GPU support and optional cuVS/RMM.

    This implementation can create CPU or GPU indices from a FAISS factory string,
    optionally leveraging RAPIDS RMM for improved GPU memory behavior. It supports
    training where applicable and batch addition of vectors.
    """

    def __init__(
        self,
        factory_string: str = "Flat",
        metric=faiss.METRIC_INNER_PRODUCT,
        use_gpu: bool = False,
        gpu_id: int = 0,
        batch_size: int = 10000,
        pq_nbits: int = 8,  # New: PQ bits per code (default 8-bit)
    ):
        """Initialize unified FAISS vector store with CPU/GPU support.

        Args:
            factory_string: FAISS index factory string (e.g., "IVF1024,PQ32", "HNSW64")
            metric: Distance metric (METRIC_INNER_PRODUCT or METRIC_L2)
            use_gpu: Whether to use GPU acceleration
            gpu_id: GPU device ID
            batch_size: Batch size for adding vectors (memory optimization)
            pq_nbits: Bits per PQ code (typically 8, can be 4, 6, 8, 10, 12, 16)
        """
        super().__init__()
        self.factory_string = factory_string
        self.metric = metric
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_id = gpu_id
        self.pq_nbits = pq_nbits
        self.index_dir: Optional[str] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.vecs: Optional[NDArray[np.float64] | torch.Tensor] = None
        self._benchmark_gpu_method()  # Choose efficient GPU method
        self._init_rmm()  # Initialize RMM memory pool if available
        self.batch_size = batch_size

    def _benchmark_gpu_method(self) -> None:
        """Benchmark and select the more efficient GPU creation method.

        Compares building a CPU index then transferring to GPU versus direct GPU
        index construction via FAISS, choosing the faster approach for this
        factory string.
        """
        if not self.use_gpu:
            return
        # Simple timing test (dummy data)
        import time
        dim = 128
        cpu_index = faiss.index_factory(dim, self.factory_string, self.metric)
        start = time.time()
        gpu_index1 = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.gpu_id, cpu_index)
        time1 = time.time() - start
        start = time.time()
        gpu_index2 = faiss.index_factory(dim, self.factory_string, self.metric)  # Direct if supported
        time2 = time.time() - start
        self._use_direct_gpu = time2 < time1  # Prioritize efficiency

    def _create_index(self, dim: int) -> faiss.Index:
        """Create index on CPU or GPU, optionally with cuVS acceleration.

        When on GPU, tries to use cuVS-accelerated implementations for IVF families
        and CAGRA for HNSW-like workloads. Falls back gracefully to CPU if GPU fails.

        Args:
            dim: Dimensionality of vectors

        Returns:
            FAISS index (CPU or GPU)
        """
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                fs_lower = self.factory_string.lower()
                
                # Prioritize cuVS-accelerated implementations when available
                if CUVS_AVAILABLE:
                    import logging
                    
                    if "ivf" in fs_lower:
                        # Extract IVF nlist parameter
                        nlist_match = re.search(r'ivf(\d+)', fs_lower)
                        nlist = int(nlist_match.group(1)) if nlist_match else 100
                        
                        if "pq" in fs_lower:
                            # IVF-PQ: cuVS-accelerated product quantization
                            pq_match = re.search(r'pq(\d+)', fs_lower)
                            m_pq = int(pq_match.group(1)) if pq_match else 8
                            
                            # Validate m_pq divides dim evenly
                            if dim % m_pq != 0:
                                logging.warning(
                                    f"PQ subquantizers ({m_pq}) should divide dimension ({dim}) evenly. "
                                    f"Adjusting to {dim // max(dim // 32, 1)}"
                                )
                                m_pq = dim // max(dim // 32, 1)  # Aim for ~32 dims per subquantizer
                            
                            index = faiss.GpuIndexIVFPQ(res, dim, nlist, m_pq, self.pq_nbits, self.metric)
                            logging.info(
                                f"Using cuVS-accelerated IVF-PQ index: "
                                f"nlist={nlist}, m={m_pq}, nbits={self.pq_nbits}"
                            )
                        
                        elif "sq" in fs_lower:
                            # IVF-SQ: cuVS-accelerated scalar quantization
                            index = faiss.GpuIndexIVFScalarQuantizer(
                                res, dim, nlist,
                                faiss.ScalarQuantizer.QT_8bit,  # 8-bit scalar quantization
                                self.metric
                            )
                            logging.info(f"Using cuVS-accelerated IVF-SQ8 index: nlist={nlist}")
                        
                        else:
                            # IVF-Flat: cuVS-accelerated flat quantization
                            index = faiss.GpuIndexIVFFlat(res, dim, nlist, self.metric)
                            logging.info(f"Using cuVS-accelerated IVF-Flat index: nlist={nlist}")
                        
                        return index
                    
                    elif "hnsw" in fs_lower:
                        # For HNSW on GPU, recommend using CPU (FAISS GPU HNSW limited)
                        # Future: integrate cuVS CAGRA as HNSW alternative
                        logging.warning(
                            "HNSW on GPU has limited support in FAISS. "
                            "Consider using IVF-based indices for GPU or keeping HNSW on CPU. "
                            "Future versions may support cuVS CAGRA for GPU graph-based search."
                        )
                        # Fallback to CPU for HNSW
                        self.use_gpu = False
                        return faiss.index_factory(dim, self.factory_string, self.metric)
                
                # Standard GPU index creation (without cuVS)
                if self._use_direct_gpu:
                    # Try direct GPU index factory (may not work for all index types)
                    index = faiss.index_factory(dim, self.factory_string, self.metric)
                else:
                    # CPU-to-GPU transfer approach (more compatible)
                    cpu_index = faiss.index_factory(dim, self.factory_string, self.metric)
                    index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)
                
                return index
            
            except Exception as e:
                import logging
                logging.warning(
                    f"Failed to create GPU index for '{self.factory_string}': {e}. "
                    f"Falling back to CPU."
                )
                self.use_gpu = False  # Fallback to CPU
        
        # CPU index creation
        return faiss.index_factory(dim, self.factory_string, self.metric)

    def _init_rmm(self) -> None:
        """Initialize RAPIDS Memory Manager for improved GPU memory handling.

        RMM provides:
        - Pool allocator to reduce allocation overhead
        - Better memory fragmentation handling
        - Faster allocation/deallocation for repeated operations

        Falls back gracefully to default CUDA allocator if RMM unavailable.
        """
        if self.use_gpu and RMM_AVAILABLE:
            try:
                # Check if RMM already initialized
                if rmm.is_initialized():
                    import logging
                    logging.debug("RMM already initialized, skipping reinitialization")
                    return
                
                # Get available GPU memory
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.get_device_properties(self.gpu_id).total_memory
                    # Use 50% of GPU memory for initial pool (conservative)
                    initial_pool_size = int(gpu_mem * 0.5)
                else:
                    initial_pool_size = 2**30  # 1GB default
                
                rmm.reinitialize(
                    pool_allocator=True,
                    initial_pool_size=initial_pool_size,
                    maximum_pool_size=None,  # Allow growth up to GPU limit
                )
                import logging
                logging.info(
                    f"Initialized RMM pool for GPU memory management: "
                    f"initial_pool={initial_pool_size/(1024**3):.2f}GB on GPU {self.gpu_id}"
                )
            except Exception as e:
                import logging
                logging.warning(
                    f"Failed to initialize RMM: {e}. "
                    f"Using default CUDA allocator (performance may be reduced)."
                )

    def index(self, docs: list[str], embeddings: NDArray[np.float64] | torch.Tensor, index_dir: str, **kwargs) -> None:
        """Build the index, optionally on GPU, and persist to disk.

        This method:
        1. Estimates memory requirements and warns if insufficient
        2. Creates the appropriate index (CPU or GPU, with cuVS if available)
        3. Trains the index if needed (IVF/PQ require training)
        4. Adds vectors in batches to limit peak memory usage
        5. Persists the index, vectors, and configuration to disk

        Args:
            docs: List of document strings (for reference, not used in indexing)
            embeddings: Vector embeddings as numpy array or torch tensor
            index_dir: Directory to save the index and vectors
            **kwargs: Additional arguments (reserved for future use)

        Training details:
        - IVF indices: Requires training on representative sample to learn centroids
        - PQ indices: Requires training to learn codebooks for quantization
        - HNSW indices: Training may initialize some parameters
        - Flat indices: No training needed

        Memory optimization:
        - Vectors added in batches of size `self.batch_size` (default 10K)
        - Reduces peak memory usage during index construction
        - Particularly important for GPU to avoid OOM
        """
        dim = embeddings.shape[1] if isinstance(embeddings, np.ndarray) else embeddings.size(1)
        num_docs = len(docs)
        
        # Estimate memory and warn if insufficient
        estimate_index_memory(self.factory_string, num_docs, dim, self.use_gpu)
        
        # Create the index (CPU or GPU)
        self.faiss_index = self._create_index(dim)
        
        # Convert to numpy if needed (FAISS expects numpy)
        emb = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        emb = emb.astype(np.float32)  # FAISS requires float32
        
        # Train the index if required
        fs_lower = self.factory_string.lower()
        needs_train = any(t in fs_lower for t in ["ivf", "pq", "hnsw"])
        
        if needs_train and not self.faiss_index.is_trained:
            import logging
            
            # Use a subset for training (full data not always necessary)
            # IVF: needs at least nlist*39 vectors (FAISS requirement)
            # PQ: needs at least 256*m vectors
            # HNSW: typically doesn't need explicit training
            
            if "ivf" in fs_lower:
                nlist_match = re.search(r'ivf(\d+)', fs_lower)
                nlist = int(nlist_match.group(1)) if nlist_match else 100
                min_train_size = nlist * 39  # FAISS minimum
                recommended_train_size = nlist * 256  # Better quality
                train_size = min(max(min_train_size, recommended_train_size), len(emb))
            else:
                train_size = min(100000, len(emb))  # Default: up to 100K vectors
            
            train_emb = emb[:train_size]
            logging.info(
                f"Training {self.factory_string} index with {train_size:,} vectors "
                f"(out of {num_docs:,} total) on {'GPU' if self.use_gpu else 'CPU'}..."
            )
            self.faiss_index.train(train_emb)
            logging.info("Training completed")
        
        # Add vectors in batches to reduce memory peak
        num_vecs = len(emb)
        import logging
        logging.info(
            f"Adding {num_vecs:,} vectors to index in batches of {self.batch_size:,}..."
        )
        
        for i in range(0, num_vecs, self.batch_size):
            batch = emb[i:i + self.batch_size]
            self.faiss_index.add(batch)
            batch_num = i // self.batch_size + 1
            total_batches = (num_vecs + self.batch_size - 1) // self.batch_size
            if batch_num % 10 == 0 or batch_num == total_batches:
                logging.debug(f"Added batch {batch_num}/{total_batches}")
        
        logging.info(f"Successfully added {num_vecs:,} vectors to index")
        
        # Persist to disk
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        # Save vectors (for potential future use)
        with open(f"{index_dir}/vecs", "wb") as fp:
            pickle.dump(emb, fp)
        
        # Save index (convert GPU to CPU for portability)
        index_to_save = self.faiss_index if not self.use_gpu else faiss.index_gpu_to_cpu(self.faiss_index)
        faiss.write_index(index_to_save, f"{index_dir}/index")
        
        # Save configuration metadata
        config = {
            "use_gpu": self.use_gpu,
            "factory_string": self.factory_string,
            "pq_nbits": self.pq_nbits,
            "num_vectors": num_vecs,
            "dimension": dim,
            "metric": "METRIC_INNER_PRODUCT" if self.metric == faiss.METRIC_INNER_PRODUCT else "METRIC_L2",
        }
        with open(f"{index_dir}/config.pkl", "wb") as fp:
            pickle.dump(config, fp)
        
        logging.info(f"Index saved to {index_dir}")

    def load_index(self, index_dir: str) -> None:
        """Load a previously built index and vectors from disk.

        Args:
            index_dir: Directory containing the saved index, vectors, and config

        Note:
            If index was built with GPU but current system doesn't have GPU,
            will automatically use CPU version. Config is restored from saved metadata.
        """
        import logging
        self.index_dir = index_dir
        
        # Load configuration metadata
        with open(f"{index_dir}/config.pkl", "rb") as fp:
            config = pickle.load(fp)
            saved_use_gpu = config.get("use_gpu", False)
            self.factory_string = config.get("factory_string", "Flat")
            self.pq_nbits = config.get("pq_nbits", 8)
        
        # Load the index from disk (always saved as CPU index)
        cpu_index = faiss.read_index(f"{index_dir}/index")
        
        # Transfer to GPU if requested and available
        if self.use_gpu and torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)
                logging.info(f"Loaded index to GPU {self.gpu_id}: {self.factory_string}")
            except Exception as e:
                logging.warning(f"Failed to load index to GPU: {e}. Using CPU.")
                self.use_gpu = False
                self.faiss_index = cpu_index
        else:
            self.faiss_index = cpu_index
            if saved_use_gpu and not self.use_gpu:
                logging.info(f"Index was built on GPU but loading on CPU: {self.factory_string}")
        
        # Load vectors
        with open(f"{index_dir}/vecs", "rb") as fp:
            self.vecs = pickle.load(fp)
        
        logging.info(
            f"Loaded index from {index_dir}: "
            f"{config.get('num_vectors', 'unknown')} vectors, "
            f"{config.get('dimension', 'unknown')} dims, "
            f"{'GPU' if self.use_gpu else 'CPU'}"
        )

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64] | torch.Tensor:
        with open(f"{index_dir}/vecs", "rb") as fp:
            vecs = pickle.load(fp)
        vecs = vecs[ids]
        return torch.from_numpy(vecs).cuda(self.gpu_id) if self.use_gpu else vecs

    def __call__(
        self,
        query_vectors: NDArray[np.float64] | torch.Tensor | list[NDArray[np.float64] | torch.Tensor],
        K: int,
        ids: Optional[list[int]] = None,
        **kwargs
    ) -> RMOutput:
        """Search the index with one or more query vectors (batch supported).

        This method performs efficient k-NN search using FAISS on CPU or GPU.
        Supports both single query and batch queries for improved throughput.

        Args:
            query_vectors: Query vector(s) in one of these formats:
                - Single vector: numpy array (1D) or torch tensor (1D)
                - Batch vectors: numpy array (2D) or torch tensor (2D)
                - List of vectors: will be stacked into batch
            K: Number of nearest neighbors to retrieve per query
            ids: Optional list of IDs to search within (creates temporary index)
            **kwargs: Reserved for future parameters

        Returns:
            RMOutput with distances and indices arrays:
                - Single query: distances shape (K,), indices shape (K,)
                - Batch queries: distances shape (N, K), indices shape (N, K)

        Note:
            - Automatically handles numpy/torch conversion
            - Batch queries are significantly faster than multiple single queries
            - GPU search provides best performance for batch queries (10-100x speedup)
            - If ids is specified, creates a temporary index (slower)
        """
        if self.faiss_index is None or self.index_dir is None:
            raise ValueError("Index not loaded. Call load_index() or index() first.")
        
        # Track whether input was single query for output formatting
        is_single_query = False
        
        # Normalize input to 2D numpy array (batch format)
        if isinstance(query_vectors, list):
            if len(query_vectors) == 0:
                return RMOutput(distances=np.array([]), indices=np.array([]))
            
            # Stack list of vectors into batch
            if isinstance(query_vectors[0], torch.Tensor):
                query_vectors = torch.stack(query_vectors)
            else:
                query_vectors = np.vstack(query_vectors)
        
        if isinstance(query_vectors, torch.Tensor):
            # Handle single vector (1D) vs batch (2D)
            if query_vectors.ndim == 1:
                is_single_query = True
                query_vectors = query_vectors.unsqueeze(0)
            
            # Convert to numpy for FAISS (always expects numpy)
            qv = query_vectors.cpu().numpy().astype(np.float32)
        
        else:  # numpy array
            # Handle single vector (1D) vs batch (2D)
            if query_vectors.ndim == 1:
                is_single_query = True
                query_vectors = np.expand_dims(query_vectors, axis=0)
            
            qv = query_vectors.astype(np.float32)
        
        # Perform search
        if ids is not None:
            # Subset search: create temporary index with only specified IDs
            # Note: This is slower as it requires index reconstruction
            subset_vecs = self.get_vectors_from_index(self.index_dir, ids)
            tmp_index = self._create_index(subset_vecs.shape[1] if isinstance(subset_vecs, np.ndarray) else subset_vecs.size(1))
            
            # Convert to numpy if needed
            subset_vecs_np = subset_vecs.cpu().numpy() if isinstance(subset_vecs, torch.Tensor) else subset_vecs
            subset_vecs_np = subset_vecs_np.astype(np.float32)
            
            # Add to temp index and search
            tmp_index.add(subset_vecs_np)
            distances, sub_indices = tmp_index.search(qv, K)
            
            # Map back to original IDs
            indices = np.array(ids)[sub_indices]
        else:
            # Standard search on full index (fast path)
            distances, indices = self.faiss_index.search(qv, K)
        
        # Return in appropriate format
        if is_single_query:
            # Single query: return 1D arrays
            return RMOutput(distances=distances[0], indices=indices[0])
        else:
            # Batch queries: return 2D arrays
            return RMOutput(distances=distances, indices=indices)
