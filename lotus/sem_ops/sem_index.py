from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.vector_store.faiss_vs import UnifiedFaissVS, estimate_index_memory  # Updated import


@pd.api.extensions.register_dataframe_accessor("sem_index")
class SemIndexDataframe:
    """
    Create a vecgtor similarity index over a column in the DataFrame. Indexing is required for columns used in sem_search, sem_cluster_by, and sem_sim_join.
    When using retrieval-based cascades for sem_filter and sem_join, indexing is required for the columns used in the semantic operation.

    Args:
        col_name (str): The column name to index.
        index_dir (str): The directory to save the index.

    Returns:
        pd.DataFrame: The DataFrame with the index directory saved.

        Example:
            >>> import pandas as pd
            >>> import lotus
            >>> from lotus.models import LM, SentenceTransformersRM
            >>> from lotus.vector_store import FaissVS
            >>> lotus.settings.configure(lm=LM(model="gpt-4o-mini"), rm=SentenceTransformersRM(model="intfloat/e5-base-v2"), vs=FaissVS())

            >>> df = pd.DataFrame({
            ...     'title': ['Machine learning tutorial', 'Data science guide', 'Python basics'],
            ...     'category': ['ML', 'DS', 'Programming']
            ... })

            # Example 1: create a new index using sem_index
            >>> df.sem_index('title', 'title_index') ## only needs to be run once; sem_index will save the index to the current directory in "title_index";
            >>> df.sem_search('title', 'AI', K=2)
            100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 81.88it/s]
                                    title
            0  Machine learning tutorial
            1         Data science guide

            # Example 2: load an existing index using load_sem_index
            >>> df.load_sem_index('title', 'title_index') ## index has already been created
            >>> df.sem_search('title', 'AI', K=2)
            100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 81.88it/s]
                                    title
            0  Machine learning tutorial
            1         Data science guide
    """

    def __init__(self, pandas_obj: Any) -> None:
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._obj.attrs["index_dirs"] = {}

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        col_name: str,
        index_dir: str,
        use_gpu: bool = False,
        factory_string: str = "Flat",
        pq_nbits: int = 8,
        batch_size: int = 10000,
    ) -> pd.DataFrame:
        """
        Create a semantic index over the specified column with GPU acceleration support.

        Args:
            col_name: Column name to index
            index_dir: Directory to save the index
            use_gpu: Whether to use GPU acceleration (default: False)
            factory_string: FAISS index factory string (default: "Flat")
                Examples: "Flat", "IVF1024,Flat", "IVF4096,PQ32", "HNSW64"
            pq_nbits: Bits per PQ code for product quantization (default: 8)
                Valid values: 4, 6, 8, 10, 12, 16
            batch_size: Batch size for adding vectors (default: 10000)

        Returns:
            DataFrame with index directory saved in attrs

        Raises:
            ValueError: If RM or VS not configured

        Note:
            - GPU acceleration provides 10-100x speedup for large datasets
            - IVF indices require training (automatically handled)
            - For datasets > 1M vectors, consider IVF-PQ for memory efficiency
            - See estimate_index_memory output for memory requirements
        """
        lotus.logger.warning("Do not reset dataframe index")
        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if rm is None or vs is None:
            raise ValueError("Configure RM and VS using lotus.settings.configure()")
        
        # Create GPU-accelerated vector store if requested
        if use_gpu:
            vs = UnifiedFaissVS(
                factory_string=factory_string,
                use_gpu=True,
                pq_nbits=pq_nbits,
                batch_size=batch_size,
            )
            lotus.settings.vs = vs
        elif factory_string != "Flat" or pq_nbits != 8 or batch_size != 10000:
            # User specified custom parameters even without GPU
            vs = UnifiedFaissVS(
                factory_string=factory_string,
                use_gpu=False,
                pq_nbits=pq_nbits,
                batch_size=batch_size,
            )
            lotus.settings.vs = vs
        
        # Generate embeddings (on GPU if use_gpu=True for faster encoding)
        embeddings = rm(self._obj[col_name].tolist(), return_tensor=use_gpu)
        
        # Estimate memory requirements and warn if insufficient
        dim = embeddings.shape[1] if hasattr(embeddings, 'shape') else embeddings.size(1)
        estimate_index_memory(factory_string, len(embeddings), dim, use_gpu)
        
        # Build and save index
        vs.index(self._obj[col_name], embeddings, index_dir)
        self._obj.attrs["index_dirs"][col_name] = index_dir
        
        return self._obj
