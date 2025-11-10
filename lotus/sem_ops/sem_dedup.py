from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, Optional
import logging
import numpy as np
from numpy.typing import NDArray

import pandas as pd

import lotus
from lotus.cache import operator_cache


@pd.api.extensions.register_dataframe_accessor("sem_dedup")
class SemDedupByDataframe:
    """
    Perform semantic deduplication on the DataFrame using optimized algorithms.

    This class provides semantic deduplication functionality with several optimization strategies:
    - Batch processing for large datasets
    - Early termination using similarity thresholds
    - Memory-efficient graph construction
    - Optimized connected component finding
    - Support for different deduplication strategies

    Methods:
        __call__: Main deduplication method with various optimization options
        _find_connected_components_optimized: Optimized graph traversal
        _batch_similarity_search: Memory-efficient similarity computation
        _greedy_dedup: Fast greedy deduplication strategy
    """

    def __init__(self, pandas_obj: Any) -> None:
        """Initialize the semantic deduplication accessor.
        
        Args:
            pandas_obj: The pandas DataFrame to operate on
            
        Raises:
            AttributeError: If the object is not a DataFrame
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj
        # Attempt optional GPU tensor support
        try:
            import torch  # type: ignore
            self._torch = torch
        except Exception:
            self._torch = None

    @staticmethod
    def _validate(obj: Any) -> None:
        """Validate that the object is a pandas DataFrame.
        
        Args:
            obj: Object to validate
            
        Raises:
            AttributeError: If obj is not a DataFrame
        """
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        col_name: str,
        threshold: float,
        strategy: str = "optimized",
        batch_size: int = 1000,
        max_pairs: int = 100000,
        use_gpu: bool = False,
    ) -> pd.DataFrame:
        """Perform semantic deduplication on the specified column.
        
        Args:
            col_name: The column name to deduplicate on
            threshold: The threshold for similarity score (0.0 to 1.0)
            strategy: Deduplication strategy ('optimized', 'greedy', 'original')
            batch_size: Batch size for processing large datasets
            max_pairs: Maximum number of similarity pairs to consider
            use_gpu: Whether to use GPU acceleration for similarity computation
            
        Returns:
            DataFrame with duplicates removed
            
        Raises:
            ValueError: If retrieval model or vector store is not configured
        """
        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. "
                "Please configure a valid retrieval model using lotus.settings.configure()"
            )

        # Choose deduplication strategy
        if strategy == "greedy":
            return self._greedy_dedup(col_name, threshold, use_gpu)
        elif strategy == "optimized":
            return self._optimized_dedup(col_name, threshold, batch_size, max_pairs, use_gpu)
        elif strategy == "original":
            return self._original_dedup(col_name, threshold)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from 'optimized', 'greedy', 'original'")

    def _original_dedup(self, col_name: str, threshold: float) -> pd.DataFrame:
        """Original deduplication implementation for backward compatibility.
        
        Args:
            col_name: The column name to deduplicate on
            threshold: The threshold for similarity score
            
        Returns:
            DataFrame with duplicates removed
        """
        joined_df = self._obj.sem_sim_join(self._obj, col_name, col_name, len(self._obj), lsuffix="_l", rsuffix="_r")
        dedup_df = joined_df[joined_df["_scores"] > threshold]
        dedup_df = dedup_df[dedup_df[f"{col_name}_l"] != dedup_df[f"{col_name}_r"]]
        lotus.logger.debug(f"dedup_df: {dedup_df}")
        left_col_name, right_col_name = f"{col_name}_l", f"{col_name}_r"

        pairs = set()
        for _, row in dedup_df.iterrows():
            left_val, right_val = row[left_col_name], row[right_col_name]
            if left_val == right_val:
                continue
            pairs.add((left_val, right_val))

        connected_components = self._find_connected_components_original(pairs)
        lotus.logger.debug(f"dedup connected components: {connected_components}")

        removed_vals: List[str] = []
        for component in connected_components:
            removed_vals.extend(component[1:])

        return self._obj[~self._obj[col_name].isin(removed_vals)]

    def _compute_unique_embeddings(
        self,
        values: NDArray[np.object_],
        use_gpu: bool,
    ) -> NDArray[np.float32]:
        """Compute embeddings for unique string values with normalization.
        
        Args:
            values: Unique values from the column as an ndarray of Python objects (strings).
            use_gpu: Whether to prefer GPU for potential downstream ops (embedding is RM-dependent).
        
        Returns:
            A float32 numpy array of shape (N, D) with L2-normalized rows.
        
        Raises:
            ValueError: If retrieval model is not configured.
        """
        rm = lotus.settings.rm
        if rm is None:
            raise ValueError(
                "The retrieval model must be an instance of RM. Please configure a valid retrieval model using lotus.settings.configure()"
            )
        # Convert all values at once for efficiency
        query_vectors = rm.convert_query_to_query_vector(values)
        # Ensure numpy float32
        if hasattr(query_vectors, "cpu") and hasattr(query_vectors, "numpy"):
            query_vectors = query_vectors.cpu().numpy()
        embeddings = np.asarray(query_vectors, dtype=np.float32)
        # Normalize to unit vectors for cosine/IP
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms
        return embeddings

    def _vectorized_similarity_pairs(
        self,
        unique_vals: NDArray[np.object_],
        threshold: float,
        block_size: int,
        max_pairs: int,
        use_gpu: bool,
    ) -> Set[Tuple[str, str]]:
        """Compute similar pairs using block matrix similarity to avoid O(n^2) Python loops.
        
        This method embeds all unique values once, normalizes them, and then performs
        block-wise matrix multiplication to find all pairs with similarity >= threshold.
        It handles within-block upper-triangular masking to avoid duplicate/self pairs.
        
        Args:
            unique_vals: Array of unique string values.
            threshold: Similarity threshold in [0, 1].
            block_size: Block size controlling memory usage.
            max_pairs: Maximum number of pairs to collect to avoid runaway memory.
            use_gpu: Whether to leverage GPU for matrix multiplications if available.
        
        Returns:
            A set of (val_i, val_j) with i < j satisfying the similarity threshold.
        """
        N = len(unique_vals)
        if N <= 1:
            return set()
        # Compute and normalize embeddings once
        embeddings = self._compute_unique_embeddings(unique_vals, use_gpu=use_gpu)

        # Choose backend
        torch = self._torch if (use_gpu and self._torch is not None and self._torch.cuda.is_available()) else None

        pairs: Set[Tuple[str, str]] = set()

        if torch is not None:
            # Use GPU tensors
            device = self._torch.device("cuda")
            E = self._torch.from_numpy(embeddings).to(device=device, dtype=self._torch.float32)
            # Process blocks
            for i in range(0, N, block_size):
                Vi = E[i : i + block_size]  # [bi, D]
                for j in range(i, N, block_size):
                    if len(pairs) >= max_pairs:
                        return pairs
                    Vj = E[j : j + block_size]  # [bj, D]
                    S = Vi @ Vj.T  # [bi, bj]
                    if i == j:
                        # Mask upper triangular to avoid i>=j duplicates and self-pairs
                        mask = self._torch.triu(self._torch.ones_like(S, dtype=self._torch.bool), diagonal=1)
                        S = self._torch.where(mask, S, self._torch.full_like(S, -1.0))
                    # Threshold
                    hit_mask = S >= threshold
                    if self._torch.any(hit_mask):
                        idxs = hit_mask.nonzero(as_tuple=False)  # [k, 2]
                        for k in range(idxs.size(0)):
                            a = i + int(idxs[k, 0].item())
                            b = j + int(idxs[k, 1].item())
                            if a < b:
                                pairs.add((str(unique_vals[a]), str(unique_vals[b])))
                                if len(pairs) >= max_pairs:
                                    return pairs
        else:
            # Use numpy on CPU
            for i in range(0, N, block_size):
                Vi = embeddings[i : i + block_size]  # [bi, D]
                for j in range(i, N, block_size):
                    if len(pairs) >= max_pairs:
                        return pairs
                    Vj = embeddings[j : j + block_size]  # [bj, D]
                    S = Vi @ Vj.T  # [bi, bj]
                    if i == j:
                        # Mask upper triangular
                        tri_mask = np.triu(np.ones_like(S, dtype=bool), k=1)
                        S = np.where(tri_mask, S, -1.0)
                    # Threshold selection
                    ii, jj = np.where(S >= threshold)
                    for a, b in zip(ii.tolist(), jj.tolist()):
                        ia = i + int(a)
                        jb = j + int(b)
                        if ia < jb:
                            pairs.add((str(unique_vals[ia]), str(unique_vals[jb])))
                            if len(pairs) >= max_pairs:
                                return pairs

        return pairs

    def _optimized_dedup(
        self, 
        col_name: str, 
        threshold: float, 
        batch_size: int, 
        max_pairs: int,
        use_gpu: bool
    ) -> pd.DataFrame:
        """Optimized deduplication using batch processing and memory efficiency.
        
        Args:
            col_name: The column name to deduplicate on
            threshold: The threshold for similarity score
            batch_size: Batch size for processing
            max_pairs: Maximum number of pairs to consider
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            DataFrame with duplicates removed
        """
        n_rows = len(self._obj)
        
        # For small datasets, use original method
        if n_rows <= batch_size:
            return self._original_dedup(col_name, threshold)
        
        # Prefer vectorized similarity over repeated DataFrame joins
        try:
            unique_vals = self._obj[col_name].unique()
            pairs = self._vectorized_similarity_pairs(
                unique_vals=unique_vals,
                threshold=threshold,
                block_size=max(128, min(batch_size, len(unique_vals))),
                max_pairs=max_pairs,
                use_gpu=use_gpu,
            )
        except Exception as e:
            lotus.logger.warning(f"Vectorized similarity path failed, falling back to batch sem_sim_join: {e}")
            # Fallback: batch similarity search using sem_sim_join
            pairs = self._batch_similarity_search(col_name, threshold, batch_size, max_pairs, use_gpu)
        
        if not pairs:
            return self._obj.copy()
        
        # Find connected components using optimized algorithm
        connected_components = self._find_connected_components_optimized(pairs)
        lotus.logger.debug(f"Found {len(connected_components)} connected components")

        # Remove duplicates (keep first element of each component)
        removed_vals: List[str] = []
        for component in connected_components:
            if len(component) > 1:
                removed_vals.extend(component[1:])

        return self._obj[~self._obj[col_name].isin(removed_vals)]

    def _greedy_dedup(self, col_name: str, threshold: float, use_gpu: bool = False) -> pd.DataFrame:
        """Fast greedy deduplication that processes items sequentially.
        
        This method is faster but may not find optimal clustering as it processes
        items in order and makes greedy decisions.
        
        Args:
            col_name: The column name to deduplicate on
            threshold: The threshold for similarity score
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            DataFrame with duplicates removed
        """
        unique_vals = self._obj[col_name].unique()
        n_vals = len(unique_vals)
        if n_vals <= 1:
            return self._obj.copy()

        # Precompute embeddings and normalize
        embeddings = self._compute_unique_embeddings(unique_vals, use_gpu=use_gpu)
        # Optional GPU
        torch = self._torch if (use_gpu and self._torch is not None and self._torch.cuda.is_available()) else None
        is_duplicate = np.zeros(n_vals, dtype=bool)
        value_to_idx: Dict[str, int] = {str(v): idx for idx, v in enumerate(unique_vals)}

        if torch is not None:
            device = self._torch.device("cuda")
            E = self._torch.from_numpy(embeddings).to(device=device, dtype=self._torch.float32)
            for i in range(n_vals):
                if is_duplicate[i]:
                    continue
                if i + 1 >= n_vals:
                    break
                v = E[i : i + 1]  # [1, D]
                rest = E[i + 1 :]  # [N-i-1, D]
                sims = (v @ rest.T).squeeze(0)  # [N-i-1]
                if sims.numel() == 0:
                    continue
                mask = sims >= threshold
                if self._torch.any(mask):
                    idxs = self._torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
                    for off in idxs:
                        is_duplicate[i + 1 + int(off)] = True
        else:
            for i in range(n_vals):
                if is_duplicate[i]:
                    continue
                if i + 1 >= n_vals:
                    break
                v = embeddings[i : i + 1]  # [1, D]
                rest = embeddings[i + 1 :]  # [N-i-1, D]
                sims = (v @ rest.T).ravel()  # [N-i-1]
                if sims.size == 0:
                    continue
                hits = np.where(sims >= threshold)[0]
                if hits.size > 0:
                    is_duplicate[i + 1 + hits] = True  # vectorized mark

        kept_vals = unique_vals[~is_duplicate]
        return self._obj[self._obj[col_name].isin(kept_vals)]

    def _batch_similarity_search(
        self, 
        col_name: str, 
        threshold: float, 
        batch_size: int, 
        max_pairs: int,
        use_gpu: bool
    ) -> Set[Tuple[str, str]]:
        """Perform batch similarity search to find similar pairs efficiently.
        
        Args:
            col_name: The column name to search on
            threshold: Similarity threshold
            batch_size: Size of each batch
            max_pairs: Maximum number of pairs to return
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Set of similar pairs (item1, item2)
        """
        unique_vals = self._obj[col_name].unique()
        n_vals = len(unique_vals)
        pairs: Set[Tuple[str, str]] = set()
        
        # Process in batches to control memory usage
        for i in range(0, n_vals, batch_size):
            if len(pairs) >= max_pairs:
                lotus.logger.warning(f"Reached maximum pairs limit ({max_pairs}), stopping search")
                break

            end_i = min(i + batch_size, n_vals)
            batch_vals_i = unique_vals[i:end_i]

            # Also compare within the same batch (upper-triangular) to avoid misses
            try:
                df_i = pd.DataFrame({col_name: batch_vals_i})
                # self comparison with K=len(batch_vals_i) to capture within-batch duplicates
                sim_results_same = df_i.sem_sim_join(
                    df_i,
                    col_name,
                    col_name,
                    K=len(batch_vals_i),
                    use_gpu=use_gpu
                )
                if not sim_results_same.empty:
                    high_sim_same = sim_results_same[sim_results_same["_scores"] > threshold]
                    for _, row in high_sim_same.iterrows():
                        val1, val2 = row[f"{col_name}_l"], row[f"{col_name}_r"]
                        if val1 != val2:
                            pair = (val1, val2) if val1 < val2 else (val2, val1)
                            pairs.add(pair)
                            if len(pairs) >= max_pairs:
                                return pairs
            except Exception as e:
                lotus.logger.warning(f"Error in within-batch similarity search: {e}")
                # continue to cross-batch

            # Compare with remaining values to avoid duplicate comparisons
            for j in range(i + batch_size, n_vals, batch_size):
                end_j = min(j + batch_size, n_vals)
                batch_vals_j = unique_vals[j:end_j]
                
                # Create temporary DataFrames for batch comparison
                df_i = pd.DataFrame({col_name: batch_vals_i})
                df_j = pd.DataFrame({col_name: batch_vals_j})
                
                try:
                    # Perform similarity join between batches
                    sim_results = df_i.sem_sim_join(
                        df_j, 
                        col_name, 
                        col_name, 
                        K=len(batch_vals_j),
                        use_gpu=use_gpu
                    )
                    
                    # Extract pairs above threshold
                    if not sim_results.empty:
                        high_sim = sim_results[sim_results["_scores"] > threshold]
                        for _, row in high_sim.iterrows():
                            val1, val2 = row[f"{col_name}_l"], row[f"{col_name}_r"]
                            if val1 != val2:  # Avoid self-pairs
                                # Ensure consistent ordering to avoid duplicates
                                pair = (val1, val2) if val1 < val2 else (val2, val1)
                                pairs.add(pair)
                                
                                if len(pairs) >= max_pairs:
                                    return pairs
                                    
                except Exception as e:
                    lotus.logger.warning(f"Error in batch similarity search: {e}")
                    continue
        
        return pairs

    def _find_connected_components_optimized(self, pairs: Set[Tuple[str, str]]) -> List[List[str]]:
        """Find connected components using Union-Find algorithm for better performance.
        
        Args:
            pairs: Set of similar pairs
            
        Returns:
            List of connected components (lists of similar items)
        """
        if not pairs:
            return []
        
        # Build parent mapping for Union-Find
        parent: Dict[str, str] = {}
        rank: Dict[str, int] = {}
        
        def find(x: str) -> str:
            """Find root with path compression."""
            if x not in parent:
                parent[x] = x
                rank[x] = 0
                return x
            
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x: str, y: str) -> None:
            """Union by rank."""
            root_x, root_y = find(x), find(y)
            if root_x == root_y:
                return
            
            # Union by rank
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
        
        # Process all pairs
        for val1, val2 in pairs:
            union(val1, val2)
        
        # Group items by their root
        components: Dict[str, List[str]] = defaultdict(list)
        for item in parent:
            root = find(item)
            components[root].append(item)
        
        return list(components.values())

    def _find_connected_components_original(self, pairs: Set[Tuple[str, str]]) -> List[List[str]]:
        """Original connected components algorithm using DFS.
        
        Args:
            pairs: Set of similar pairs
            
        Returns:
            List of connected components
        """
        graph = defaultdict(set)
        for left_val, right_val in pairs:
            graph[left_val].add(right_val)
            graph[right_val].add(left_val)

        visited = set()
        components = []

        def dfs(node: str, component: List[str]) -> None:
            """Depth-first search to find connected component."""
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    stack.extend(graph[current] - visited)

        for node in graph:
            if node not in visited:
                component: List[str] = []
                dfs(node, component)
                components.append(component)

        return components
