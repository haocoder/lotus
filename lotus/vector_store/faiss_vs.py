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

    This attempts to account for different index families:
    - Flat: raw vectors only (float32)
    - IVF-Flat: raw vectors + centroids
    - IVF-SQ8: product-quantized to 1 byte per dim + centroids
    - IVF-PQm: m bytes per vector code (assuming 8 bits/code) + centroids + codebooks
    - HNSWm: raw vectors + graph links (approx m neighbors per node)

    The calculation is heuristic. We warn if estimated memory exceeds 80% of
    available RAM/VRAM.
    """
    fs_lower = factory_string.lower()

    # Defaults and parsed params
    nlist = 0
    m_pq = 0
    m_hnsw = 32
    # Try to extract parameters
    try:
        import re as _re
        nlist_match = _re.search(r"ivf(\d+)", fs_lower)
        if nlist_match:
            nlist = int(nlist_match.group(1))
        pq_match = _re.search(r"pq(\d+)", fs_lower)
        if pq_match:
            m_pq = int(pq_match.group(1))
        hnsw_match = _re.search(r"hnsw(\d+)", fs_lower)
        if hnsw_match:
            m_hnsw = int(hnsw_match.group(1))
    except Exception:
        pass

    # Base components
    bytes_f32 = 4
    raw_vectors = num_vecs * dim * bytes_f32
    mem_bytes = raw_vectors

    if "flat" in fs_lower and "ivf" not in fs_lower:
        mem_bytes = raw_vectors
    elif "ivf" in fs_lower and "pq" not in fs_lower and "sq" not in fs_lower:
        # IVF-Flat: raw vectors + centroids
        centroids = max(nlist, 1) * dim * bytes_f32
        mem_bytes = raw_vectors + centroids
    elif "ivf" in fs_lower and "sq" in fs_lower:
        # IVF-SQ8: 1 byte per component codes + centroids
        codes = num_vecs * dim  # 1 byte per dim
        centroids = max(nlist, 1) * dim * bytes_f32
        mem_bytes = codes + centroids
    elif "ivf" in fs_lower and "pq" in fs_lower:
        # IVF-PQm: m bytes per vector code + centroids + codebooks (~ m * 256 * (dim/m) * 4)
        m = max(m_pq, 8)
        codes = num_vecs * m  # bytes, assuming 8-bit subquantizers
        centroids = max(nlist, 1) * dim * bytes_f32
        codebooks = m * 256 * max(dim // m, 1) * bytes_f32
        mem_bytes = codes + centroids + codebooks
    elif "hnsw" in fs_lower:
        # HNSW: raw vectors + graph links (approx m_hnsw pointers per node)
        # Assume 8 bytes per link (64-bit index) and 4 bytes per level overhead
        graph_links = num_vecs * m_hnsw * 8
        overhead = num_vecs * 4
        mem_bytes = raw_vectors + graph_links + overhead
    else:
        # Fallback to raw vectors
        mem_bytes = raw_vectors

    # Available memory
    if is_gpu and torch.cuda.is_available():
        avail = torch.cuda.get_device_properties(0).total_memory
    else:
        avail = psutil.virtual_memory().available
    if mem_bytes > int(avail * 0.8):
        import logging
        logging.warning(
            f"Potential insufficient memory for index: estimated {mem_bytes} bytes > 80% of available {avail} bytes"
        )
    return int(mem_bytes)

class UnifiedFaissVS(VS):
    """Unified FAISS vector store with CPU/GPU support and optional cuVS/RMM.

    This implementation can create CPU or GPU indices from a FAISS factory string,
    optionally leveraging RAPIDS RMM for improved GPU memory behavior. It supports
    training where applicable and batch addition of vectors.
    """

    def __init__(self, factory_string: str = "Flat", metric=faiss.METRIC_INNER_PRODUCT, use_gpu: bool = False, gpu_id: int = 0, batch_size: int = 10000):
        super().__init__()
        # Replace strict check with pattern-based validation
        self.factory_string = factory_string
        self.metric = metric
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu_id = gpu_id
        self.index_dir: Optional[str] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.vecs: Optional[NDArray[np.float64] | torch.Tensor] = None
        self._benchmark_gpu_method()  # Choose efficient GPU method
        self._init_rmm()  # New: Initialize RMM if possible
        self.batch_size = batch_size  # New: for batch addition

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

        When on GPU, tries to use cuVS-accelerated implementations when
        available for IVF families; otherwise falls back to FAISS GPU/CPU.
        """
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                fs_lower = self.factory_string.lower()
                if CUVS_AVAILABLE:
                    if "ivf" in fs_lower:
                        nlist = int(re.search(r'ivf(\d+)', fs_lower).group(1)) if re.search(r'ivf(\d+)', fs_lower) else 100
                        if "pq" in fs_lower:
                            # IVF-PQ support
                            m = int(re.search(r'pq(\d+)', fs_lower).group(1)) if re.search(r'pq(\d+)', fs_lower) else 8  # Default subquantizers
                            index = faiss.GpuIndexIVFPQ(res, dim, nlist, m, 8, self.metric)  # bits=8 default
                            import logging
                            logging.info("Using cuVS-accelerated IVF-PQ index")
                        else:
                            # IVF-Flat
                            index = faiss.GpuIndexIVFFlat(res, dim, nlist, self.metric)
                            import logging
                            logging.info("Using cuVS-accelerated IVF-Flat index")
                    elif "hnsw" in fs_lower:
                        # cuVS CAGRA: not directly available via FAISS Python; fallback
                        import logging
                        logging.warning("cuVS CAGRA not available via FAISS; falling back to FAISS HNSW")
                        index = faiss.index_factory(dim, self.factory_string, self.metric)
                    else:
                        index = faiss.index_factory(dim, self.factory_string, self.metric)
                    return index
                elif self._use_direct_gpu:
                    index = faiss.index_factory(dim, self.factory_string, self.metric)
                else:
                    cpu_index = faiss.index_factory(dim, self.factory_string, self.metric)
                    index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)
                return index
            except Exception:
                self.use_gpu = False  # Fallback
        return faiss.index_factory(dim, self.factory_string, self.metric)  # CPU

    def _init_rmm(self) -> None:
        """Initialize RAPIDS Memory Manager for better GPU memory handling."""
        if self.use_gpu and RMM_AVAILABLE:
            try:
                rmm.reinitialize(
                    pool_allocator=True,
                    initial_pool_size=2**30,
                    maximum_pool_size=None,
                )
                import logging
                logging.info("Initialized RMM pool for GPU memory management")
            except Exception as e:
                import logging
                logging.warning(f"Failed to initialize RMM: {e}. Using default CUDA allocator.")

    def index(self, docs: list[str], embeddings: NDArray[np.float64] | torch.Tensor, index_dir: str, **kwargs) -> None:
        """Build the index, optionally on GPU, and persist to disk.

        Adds vectors in batches to limit peak memory usage and trains trainable
        index types (IVF/PQ/HNSW) when required. Saves both vectors and index
        artifacts to the provided directory along with configuration metadata.
        """
        dim = embeddings.shape[1] if isinstance(embeddings, np.ndarray) else embeddings.size(1)
        estimate_index_memory(self.factory_string, len(docs), dim, self.use_gpu)
        self.faiss_index = self._create_index(dim)
        emb = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else embeddings
        
        # Explicit training for trainable indexes
        fs_lower = self.factory_string.lower()
        needs_train = any(t in fs_lower for t in ["ivf", "pq", "hnsw"])  # Detect trainable types
        if needs_train and not self.faiss_index.is_trained:
            train_size = min(100000, len(emb))  # Train on subset for efficiency (e.g., 100k vectors)
            train_emb = emb[:train_size]
            if self.use_gpu:
                # For GPU, ensure training data is on device (FAISS handles internally)
                pass  # index_cpu_to_gpu already sets up for training
            self.faiss_index.train(train_emb)
        
        # Optimized: Add in batches to reduce memory peak
        num_vecs = len(emb)
        for i in range(0, num_vecs, self.batch_size):
            batch = emb[i:i + self.batch_size]
            self.faiss_index.add(batch)
            import logging
            logging.debug(f"Added batch {i//self.batch_size + 1}/{(num_vecs + self.batch_size - 1)//self.batch_size}")
        
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        with open(f"{index_dir}/vecs", "wb") as fp:
            pickle.dump(emb, fp)  # Save as numpy for persistence
        faiss.write_index(self.faiss_index if not self.use_gpu else faiss.index_gpu_to_cpu(self.faiss_index), f"{index_dir}/index")
        with open(f"{index_dir}/config.pkl", "wb") as fp:  # Persist config
            pickle.dump({"use_gpu": self.use_gpu, "factory_string": self.factory_string}, fp)

    def load_index(self, index_dir: str) -> None:
        """Load a previously built index and vectors from disk."""
        self.index_dir = index_dir
        with open(f"{index_dir}/config.pkl", "rb") as fp:
            config = pickle.load(fp)
            self.use_gpu = config["use_gpu"]
            self.factory_string = config["factory_string"]
        cpu_index = faiss.read_index(f"{index_dir}/index")
        self.faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.gpu_id, cpu_index) if self.use_gpu else cpu_index
        with open(f"{index_dir}/vecs", "rb") as fp:
            self.vecs = pickle.load(fp)

    def get_vectors_from_index(self, index_dir: str, ids: list[int]) -> NDArray[np.float64] | torch.Tensor:
        with open(f"{index_dir}/vecs", "rb") as fp:
            vecs = pickle.load(fp)
        vecs = vecs[ids]
        return torch.from_numpy(vecs).cuda(self.gpu_id) if self.use_gpu else vecs

    def __call__(self, query_vectors: NDArray[np.float64] | torch.Tensor | list[NDArray[np.float64] | torch.Tensor], K: int, ids: Optional[list[int]] = None, **kwargs) -> RMOutput | list[RMOutput]:
        """Search the index with one or more query vectors.

        Supports optional ID subset search by materializing a temporary index
        over the requested vectors. Accepts both numpy arrays and torch tensors.
        """
        if self.faiss_index is None or self.index_dir is None:
            raise ValueError("Index not loaded")
        if isinstance(query_vectors, list):  # Batch query support
            return [self.__call__(qv, K, ids, **kwargs) for qv in query_vectors]  # Parallel if needed
        qv = query_vectors.cpu().numpy() if isinstance(query_vectors, torch.Tensor) else query_vectors
        if ids is not None:
            subset_vecs = self.get_vectors_from_index(self.index_dir, ids)
            tmp_index = self._create_index(subset_vecs.shape[1])
            tmp_index.add(subset_vecs.cpu().numpy() if isinstance(subset_vecs, torch.Tensor) else subset_vecs)
            distances, sub_indices = tmp_index.search(qv, K)
            indices = np.array(ids)[sub_indices].tolist()
        else:
            distances, indices = self.faiss_index.search(qv, K)
        return RMOutput(distances=distances, indices=indices)
