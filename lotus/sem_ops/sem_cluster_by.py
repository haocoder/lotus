from typing import Any, Optional, Dict, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import lotus
from lotus.cache import operator_cache


@pd.api.extensions.register_dataframe_accessor("sem_cluster_by")
class SemClusterByDataframe:
    """
    Perform semantic clustering on the DataFrame with optimized performance.
    
    This class provides high-performance semantic clustering with several optimizations:
    - Adaptive batch processing for large datasets
    - GPU acceleration support with automatic fallback
    - Memory-efficient vector retrieval
    - Optional clustering quality metrics
    - Progress tracking for long-running operations
    
    Performance Features:
    - Automatic batch size determination based on available memory
    - Efficient vector I/O for large datasets
    - GPU-accelerated K-means when available
    - Quality metrics computation (silhouette score, inertia)

    Args:
        col_name (str): The column name to cluster on.
        ncentroids (int): The number of centroids.
        return_scores (bool): Whether to include centroid distance scores. Default False.
        return_centroids (bool): Whether to return cluster centroids. Default False.
        return_metrics (bool): Whether to return clustering quality metrics. Default False.
        niter (int): The number of K-means iterations. Default 20.
        verbose (bool): Whether to print verbose output. Default False.
        prefer_gpu (bool): Whether to prefer GPU acceleration when available. Default False.
        batch_size (int): Batch size for vector retrieval (auto-determined if None). Default None.
        show_progress (bool): Whether to show progress information. Default False.

    Returns:
        pd.DataFrame or Tuple[pd.DataFrame, Dict]: 
            - If return_centroids or return_metrics is False: DataFrame with cluster_id column
            - Otherwise: Tuple of (DataFrame, info_dict) containing requested information

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import LM, SentenceTransformersRM
        >>> from lotus.vector_store import FaissVS
        >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"), rm=SentenceTransformersRM(model="intfloat/e5-base-v2"), vs=FaissVS())

        >>> df = pd.DataFrame({
        ...     'title': ['Machine learning tutorial', 'Data science guide', 'Python basics', 'AI in finance', 'Cooking healthy food', "Recipes for the holidays"],
        ... })

        >>> df.sem_index('title', 'title_index')  # only needs to be run once
        >>> df.load_sem_index('title', 'title_index')

        Basic usage:
        >>> result = df.sem_cluster_by('title', 2)
                                title  cluster_id
        0  Machine learning tutorial           0
        1         Data science guide           0
        2              Python basics           0
        3              AI in finance           0
        4       Cooking healthy food           1
        5   Recipes for the holidays           1
        
        With quality metrics:
        >>> result, info = df.sem_cluster_by('title', 2, return_metrics=True, return_centroids=True)
        >>> print(info['metrics']['silhouette_score'])
        0.85
        >>> print(info['centroids'].shape)
        (2, 768)
        
        GPU-accelerated clustering:
        >>> result = df.sem_cluster_by('title', 2, prefer_gpu=True, verbose=True)
        Using GPU-accelerated K-means clustering...
        
        Large dataset with custom batch size:
        >>> large_df.sem_cluster_by('text', 100, batch_size=50000, show_progress=True)
    """

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    def _estimate_memory_requirements(
        self,
        n_samples: int,
        dim: int,
        use_gpu: bool = False
    ) -> int:
        """
        Estimate memory requirements for clustering.
        
        Args:
            n_samples: Number of samples to cluster
            dim: Dimensionality of vectors
            use_gpu: Whether GPU will be used
            
        Returns:
            Estimated memory in bytes
        """
        # Vector storage (float32)
        vector_mem = n_samples * dim * 4
        
        # K-means intermediate results (distance matrix, assignments, centroids)
        # Conservative estimate: 3x the vector size
        kmeans_mem = vector_mem * 3
        
        total_mem = vector_mem + kmeans_mem
        
        # GPU has additional overhead
        if use_gpu:
            total_mem = int(total_mem * 1.2)
        
        return total_mem

    def _get_adaptive_batch_size(
        self,
        n_samples: int,
        dim: int,
        use_gpu: bool = False
    ) -> int:
        """
        Determine adaptive batch size based on available memory.
        
        Args:
            n_samples: Number of samples
            dim: Vector dimensionality
            use_gpu: Whether GPU will be used
            
        Returns:
            Optimal batch size
        """
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    # Use 70% of available GPU memory
                    usable_memory = gpu_memory * 0.7
                    # Estimate bytes per sample (with safety factor)
                    bytes_per_sample = dim * 4 * 3  # float32 * 3x safety factor
                    batch_size = int(usable_memory / bytes_per_sample)
                    return min(n_samples, max(1000, batch_size))
            except Exception:
                pass
            # Default GPU batch size if estimation fails
            return min(n_samples, 100000)
        else:
            try:
                import psutil
                available_ram = psutil.virtual_memory().available
                # Use 50% of available RAM
                usable_ram = available_ram * 0.5
                bytes_per_sample = dim * 4 * 2
                batch_size = int(usable_ram / bytes_per_sample)
                return min(n_samples, max(5000, batch_size))
            except Exception:
                # Conservative default for CPU
                return min(n_samples, 50000)

    def _get_vectors_batch(
        self,
        vs: Any,
        col_index_dir: str,
        ids: list[int],
        batch_size: int = 10000,
        verbose: bool = False
    ) -> NDArray[np.float32]:
        """
        Get vectors in batches to optimize I/O.
        
        Args:
            vs: Vector store instance
            col_index_dir: Index directory
            ids: List of document IDs
            batch_size: Batch size for retrieval
            verbose: Whether to print progress
            
        Returns:
            Array of vectors
        """
        n_ids = len(ids)
        
        # For small datasets, retrieve all at once
        if n_ids <= batch_size:
            return vs.get_vectors_from_index(col_index_dir, ids)
        
        # Batch retrieval for large datasets
        if verbose:
            lotus.logger.info(f"Retrieving {n_ids} vectors in batches of {batch_size}")
        
        vectors = []
        for i in range(0, n_ids, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vs.get_vectors_from_index(col_index_dir, batch_ids)
            vectors.append(batch_vectors)
            
            if verbose and (i + batch_size) % (batch_size * 10) == 0:
                lotus.logger.info(f"Retrieved {min(i + batch_size, n_ids)}/{n_ids} vectors")
        
        return np.vstack(vectors)

    def _compute_clustering_metrics(
        self,
        vectors: NDArray[np.float32],
        assignments: NDArray[np.int64],
        centroids: NDArray[np.float32]
    ) -> Dict[str, float]:
        """
        Compute clustering quality metrics.
        
        Args:
            vectors: Input vectors
            assignments: Cluster assignments
            centroids: Cluster centroids
            
        Returns:
            Dictionary of metrics
        """
        metrics: Dict[str, float] = {}
        
        try:
            # Compute inertia (within-cluster sum of squares)
            inertia = 0.0
            for i in range(len(vectors)):
                cluster_id = assignments[i]
                centroid = centroids[cluster_id]
                distance = np.sum((vectors[i] - centroid) ** 2)
                inertia += distance
            metrics['inertia'] = float(inertia)
            
            # Silhouette score (only for reasonable sizes)
            if len(vectors) <= 50000 and len(np.unique(assignments)) > 1:
                try:
                    from sklearn.metrics import silhouette_score
                    metrics['silhouette_score'] = float(silhouette_score(
                        vectors, assignments, sample_size=min(10000, len(vectors))
                    ))
                except Exception as e:
                    lotus.logger.debug(f"Could not compute silhouette score: {e}")
        except Exception as e:
            lotus.logger.warning(f"Error computing clustering metrics: {e}")
        
        return metrics

    @operator_cache
    def __call__(
        self,
        col_name: str,
        ncentroids: int,
        return_scores: bool = False,
        return_centroids: bool = False,
        return_metrics: bool = False,
        niter: int = 20,
        verbose: bool = False,
        prefer_gpu: bool = False,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Perform semantic clustering on the DataFrame.
        
        Args:
            col_name: The column name to cluster on
            ncentroids: The number of centroids
            return_scores: Whether to include centroid distance scores in the DataFrame
            return_centroids: Whether to return cluster centroids
            return_metrics: Whether to return clustering quality metrics
            niter: The number of K-means iterations
            verbose: Whether to print verbose output
            prefer_gpu: Whether to prefer GPU acceleration when available
            batch_size: Batch size for vector retrieval (auto if None)
            show_progress: Whether to show progress information
            
        Returns:
            If return_centroids or return_metrics is False:
                pd.DataFrame with cluster_id column (and optionally centroid_sim_score)
            Otherwise:
                Tuple of (DataFrame, info_dict) where info_dict contains:
                    - 'centroids': cluster centroids if return_centroids=True
                    - 'metrics': clustering metrics if return_metrics=True
                    
        Raises:
            ValueError: If RM or VS is not configured, or if column not found
            
        Example:
            >>> df = pd.DataFrame({'text': ['Machine learning', 'Deep learning', 'Cooking', 'Recipe']})
            >>> df.sem_index('text', 'text_idx')
            >>> result_df = df.sem_cluster_by('text', ncentroids=2)
            >>> # Or get additional information
            >>> result_df, info = df.sem_cluster_by('text', ncentroids=2, return_centroids=True, return_metrics=True)
            >>> print(info['metrics']['silhouette_score'])
        """
        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. "
                "Please configure a valid retrieval model using lotus.settings.configure()"
            )

        # Early validation
        if col_name not in self._obj.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
        
        if ncentroids <= 0:
            raise ValueError(f"ncentroids must be positive, got {ncentroids}")
        
        if ncentroids > len(self._obj):
            raise ValueError(
                f"Number of centroids ({ncentroids}) must be less than or equal to number of documents ({len(self._obj)})"
            )

        # Use optimized clustering with GPU support
        if prefer_gpu:
            try:
                from lotus.utils.gpu_clustering import gpu_cluster
                cluster_fn = gpu_cluster(col_name, ncentroids, prefer_gpu=True)
                
                # Get vectors efficiently
                try:
                    col_index_dir = self._obj.attrs["index_dirs"][col_name]
                except KeyError:
                    raise ValueError(f"Index directory for column {col_name} not found in DataFrame")
                
                if vs.index_dir != col_index_dir:
                    vs.load_index(col_index_dir)
                
                # Get vectors with batching if needed
                ids = self._obj.index.tolist()
                
                # Determine batch size if not specified
                if batch_size is None and len(ids) > 50000:
                    # For large datasets, estimate batch size
                    sample_vec = vs.get_vectors_from_index(col_index_dir, [ids[0]])
                    dim = sample_vec.shape[1]
                    batch_size = self._get_adaptive_batch_size(len(ids), dim, use_gpu=True)
                    if verbose:
                        lotus.logger.info(f"Using adaptive batch size: {batch_size}")
                
                # Execute GPU clustering (it handles batching internally)
                indices = cluster_fn(self._obj, niter, verbose)
                
                # For additional info, we need to re-run with full output
                if return_scores or return_centroids or return_metrics:
                    from lotus.utils.gpu_clustering import _gpu_kmeans_manager
                    vec_set = self._get_vectors_batch(vs, col_index_dir, ids, 
                                                       batch_size=batch_size or 10000, 
                                                       verbose=verbose)
                    assignments, distances, centroids = _gpu_kmeans_manager.gpu_kmeans(
                        vec_set, ncentroids, niter=niter, verbose=verbose
                    )
                    indices = assignments.tolist()
                    scores = distances.tolist()
                else:
                    scores = None
                    centroids = None
                
            except Exception as e:
                if verbose:
                    lotus.logger.warning(f"GPU clustering failed: {e}, falling back to CPU")
                prefer_gpu = False
        
        # CPU fallback or explicit CPU mode
        if not prefer_gpu:
            cluster_fn = lotus.utils.cluster(col_name, ncentroids, prefer_gpu=False)
            
            # Get index directory
            try:
                col_index_dir = self._obj.attrs["index_dirs"][col_name]
            except KeyError:
                raise ValueError(f"Index directory for column {col_name} not found in DataFrame")
            
            if vs.index_dir != col_index_dir:
                vs.load_index(col_index_dir)
            
            # Get vectors efficiently
            ids = self._obj.index.tolist()
            
            # Determine batch size
            if batch_size is None and len(ids) > 50000:
                sample_vec = vs.get_vectors_from_index(col_index_dir, [ids[0]])
                dim = sample_vec.shape[1]
                batch_size = self._get_adaptive_batch_size(len(ids), dim, use_gpu=False)
                if verbose:
                    lotus.logger.info(f"Using adaptive batch size: {batch_size}")
            
            # Get vectors
            vec_set = self._get_vectors_batch(vs, col_index_dir, ids, 
                                               batch_size=batch_size or 10000, 
                                               verbose=verbose or show_progress)
            
            # Perform clustering with full output if needed
            if return_scores or return_centroids or return_metrics:
                import faiss
                d = vec_set.shape[1]
                kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
                
                if show_progress:
                    lotus.logger.info(f"Training K-means with {len(vec_set)} vectors...")
                
                kmeans.train(vec_set)
                
                # Get assignments and scores
                distances, assignments = kmeans.index.search(vec_set, 1)
                indices = assignments.flatten().tolist()
                scores = distances.flatten().tolist()
                centroids = kmeans.centroids
            else:
                # Standard path without extra info
                indices = cluster_fn(self._obj, niter, verbose)
                scores = None
                centroids = None

        # Add cluster assignments to DataFrame
        self._obj["cluster_id"] = pd.Series(indices, index=self._obj.index)
        
        # Optionally add scores
        if return_scores and scores is not None:
            self._obj["centroid_distance"] = pd.Series(scores, index=self._obj.index)
        
        # Prepare return value
        return_info = return_centroids or return_metrics
        
        if return_info:
            info: Dict[str, Any] = {}
            
            if return_centroids and centroids is not None:
                info['centroids'] = centroids
            
            if return_metrics and centroids is not None:
                # Recompute vectors if needed
                if 'vec_set' not in locals():
                    ids = self._obj.index.tolist()
                    vec_set = self._get_vectors_batch(vs, col_index_dir, ids, 
                                                       batch_size=batch_size or 10000, 
                                                       verbose=False)
                
                metrics = self._compute_clustering_metrics(
                    vec_set, 
                    np.array(indices, dtype=np.int64), 
                    centroids
                )
                info['metrics'] = metrics
                
                if verbose:
                    lotus.logger.info(f"Clustering metrics: {metrics}")
            
            return self._obj, info
        else:
            return self._obj
