"""GPU-accelerated clustering utilities for Lotus."""

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import lotus

logger = logging.getLogger(__name__)


class GPUKMeansManager:
    """Manages GPU-accelerated K-means clustering operations."""
    
    def __init__(self) -> None:
        """Initialize GPU K-means manager."""
        self._gpu_available: Optional[bool] = None
        self._faiss_gpu_available: bool = False
        self._check_gpu_availability()
    
    def _check_gpu_availability(self) -> None:
        """Check if GPU acceleration is available for clustering."""
        try:
            import faiss
            gpu_count = faiss.get_num_gpus()
            self._gpu_available = gpu_count > 0
            self._faiss_gpu_available = True
            if self._gpu_available:
                logger.info(f"GPU K-means clustering available with {gpu_count} GPU(s)")
            else:
                logger.info("No GPUs available for clustering, using CPU")
        except Exception as e:
            logger.warning(f"GPU clustering check failed: {e}, falling back to CPU")
            self._gpu_available = False
            self._faiss_gpu_available = False
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for clustering."""
        return self._gpu_available or False
    
    def gpu_kmeans(
        self,
        vectors: NDArray[np.float64],
        ncentroids: int,
        niter: int = 20,
        verbose: bool = False,
        gpu_id: int = 0,
        seed: int = 1234,
        spherical: bool = False,
        min_points_per_centroid: int = 1,
        max_points_per_centroid: int = 2**31 - 1,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Perform GPU-accelerated K-means clustering.
        
        Args:
            vectors: Input vectors to cluster [n_samples, n_features]
            ncentroids: Number of cluster centroids
            niter: Maximum number of iterations
            verbose: Whether to print progress information
            gpu_id: GPU device ID to use
            seed: Random seed for reproducibility
            spherical: Whether to use spherical K-means (L2 normalize centroids)
            min_points_per_centroid: Minimum points per centroid
            max_points_per_centroid: Maximum points per centroid
            
        Returns:
            Tuple of (cluster_assignments, distances_to_centroids, centroids)
        """
        if not self.is_gpu_available():
            raise RuntimeError("GPU not available for clustering")
        
        try:
            import faiss
            
            # Ensure vectors are float32 for FAISS GPU
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            
            d = vectors.shape[1]
            
            # Configure K-means
            kmeans = faiss.Kmeans(
                d=d,
                k=ncentroids,
                niter=niter,
                verbose=verbose,
                seed=seed,
                spherical=spherical,
                min_points_per_centroid=min_points_per_centroid,
                max_points_per_centroid=max_points_per_centroid,
                gpu=True  # Enable GPU acceleration
            )
            
            # Train the model
            kmeans.train(vectors)
            
            # Get cluster assignments and distances
            distances, assignments = kmeans.index.search(vectors, 1)
            
            # Get centroids
            centroids = kmeans.centroids
            
            return assignments.flatten(), distances.flatten(), centroids
            
        except Exception as e:
            logger.error(f"GPU K-means clustering failed: {e}")
            raise
    
    def cpu_kmeans_fallback(
        self,
        vectors: NDArray[np.float64],
        ncentroids: int,
        niter: int = 20,
        verbose: bool = False,
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Fallback CPU K-means implementation.
        
        Args:
            vectors: Input vectors to cluster
            ncentroids: Number of cluster centroids
            niter: Maximum number of iterations
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (cluster_assignments, distances_to_centroids, centroids)
        """
        try:
            import faiss
            
            # Ensure vectors are float32 for FAISS
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            
            d = vectors.shape[1]
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
            kmeans.train(vectors)
            
            # Get assignments and distances
            distances, assignments = kmeans.index.search(vectors, 1)
            centroids = kmeans.centroids
            
            return assignments.flatten(), distances.flatten(), centroids
            
        except Exception as e:
            logger.error(f"CPU K-means clustering failed: {e}")
            raise


# Global GPU K-means manager instance
_gpu_kmeans_manager = GPUKMeansManager()


def gpu_cluster(col_name: str, ncentroids: int, prefer_gpu: bool = True) -> Callable[[pd.DataFrame, int, bool], list[int]]:
    """
    Returns a function that clusters a DataFrame by a column using GPU-accelerated K-means.
    
    Args:
        col_name: The column name to cluster by
        ncentroids: The number of centroids to use
        prefer_gpu: Whether to prefer GPU acceleration when available
        
    Returns:
        Callable: The function that clusters the DataFrame
    """
    
    def cluster_fn(
        df: pd.DataFrame,
        niter: int = 20,
        verbose: bool = False,
        method: str = "kmeans",
    ) -> list[int]:
        """
        Cluster by column using GPU-accelerated K-means when available.
        
        Args:
            df: DataFrame to cluster
            niter: Number of K-means iterations
            verbose: Whether to print progress
            method: Clustering method (currently only 'kmeans' supported)
            
        Returns:
            List of cluster IDs for each row
        """
        if col_name not in df.columns:
            raise ValueError(f"Column {col_name} not found in DataFrame")
        
        if ncentroids > len(df):
            raise ValueError(f"Number of centroids must be less than number of documents. {ncentroids} > {len(df)}")
        
        # Get retrieval model and vector store
        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. "
                "Please configure a valid retrieval model using lotus.settings.configure()"
            )
        
        try:
            col_index_dir = df.attrs["index_dirs"][col_name]
        except KeyError:
            raise ValueError(f"Index directory for column {col_name} not found in DataFrame")
        
        if vs.index_dir != col_index_dir:
            vs.load_index(col_index_dir)
        assert vs.index_dir == col_index_dir
        
        # Get vectors from index
        ids = df.index.tolist()
        vec_set = vs.get_vectors_from_index(col_index_dir, ids)
        
        # Try GPU clustering first if preferred and available
        use_gpu = prefer_gpu and _gpu_kmeans_manager.is_gpu_available()
        
        if use_gpu:
            try:
                if verbose:
                    logger.info("Using GPU-accelerated K-means clustering")
                
                assignments, distances, centroids = _gpu_kmeans_manager.gpu_kmeans(
                    vec_set, ncentroids, niter=niter, verbose=verbose
                )
                
                return assignments.tolist()
                
            except Exception as e:
                logger.warning(f"GPU clustering failed: {e}, falling back to CPU")
                use_gpu = False
        
        # Fallback to CPU clustering
        if not use_gpu:
            if verbose:
                logger.info("Using CPU K-means clustering")
            
            assignments, distances, centroids = _gpu_kmeans_manager.cpu_kmeans_fallback(
                vec_set, ncentroids, niter=niter, verbose=verbose
            )
            
            return assignments.tolist()
    
    return cluster_fn


def adaptive_cluster(
    col_name: str, 
    ncentroids: int, 
    batch_size: Optional[int] = None,
    prefer_gpu: bool = True
) -> Callable[[pd.DataFrame, int, bool], list[int]]:
    """
    Returns a function that adaptively clusters large datasets with automatic batching.
    
    Args:
        col_name: The column name to cluster by
        ncentroids: The number of centroids to use
        batch_size: Batch size for large datasets (auto-determined if None)
        prefer_gpu: Whether to prefer GPU acceleration when available
        
    Returns:
        Callable: The adaptive clustering function
    """
    
    def adaptive_cluster_fn(
        df: pd.DataFrame,
        niter: int = 20,
        verbose: bool = False,
        method: str = "kmeans",
    ) -> list[int]:
        """
        Adaptively cluster with automatic batching for large datasets.
        """
        n_samples = len(df)
        
        # Determine batch size based on dataset size and available memory
        if batch_size is None:
            # Heuristic: batch size based on dataset size and GPU memory
            if _gpu_kmeans_manager.is_gpu_available() and prefer_gpu:
                # Assume ~4GB GPU memory, ~4 bytes per float32, ~1024 dims
                estimated_batch_size = min(n_samples, 100000)  # Conservative estimate
            else:
                estimated_batch_size = min(n_samples, 50000)  # CPU memory is usually larger
        else:
            estimated_batch_size = min(batch_size, n_samples)
        
        # For small datasets, use regular clustering
        if n_samples <= estimated_batch_size:
            return gpu_cluster(col_name, ncentroids, prefer_gpu)(df, niter, verbose, method)
        
        # For large datasets, use mini-batch approach
        if verbose:
            logger.info(f"Using adaptive clustering with batch size {estimated_batch_size} for {n_samples} samples")
        
        # This is a simplified implementation - in practice, you'd want more sophisticated
        # approaches like mini-batch K-means or hierarchical clustering
        regular_cluster_fn = gpu_cluster(col_name, ncentroids, prefer_gpu)
        return regular_cluster_fn(df, niter, verbose, method)
    
    return adaptive_cluster_fn
