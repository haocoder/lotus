from typing import Any
import os
import pickle

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.types import RerankerOutput, RMOutput


@pd.api.extensions.register_dataframe_accessor("sem_search")
class SemSearchDataframe:
    """
    Perform semantic search on the DataFrame.

    This method performs semantic search over the specified column using
    a natural language query. It can use vector-based retrieval for initial
    results and optional reranking for improved relevance.

    Args:
        col_name (str): The column name to search on. This column should
            contain the text content to be searched.
        query (str): The natural language query string. This should describe
            what you're looking for in the data.
        K (int | None, optional): The number of documents to retrieve using
            vector search. Must be provided if n_rerank is None. Defaults to None.
        n_rerank (int | None, optional): The number of documents to rerank
            using a cross-encoder reranker. Must be provided if K is None.
            Defaults to None.
        return_scores (bool, optional): Whether to return the similarity scores
            from the vector search. Useful for understanding result relevance.
            Defaults to False.
        suffix (str, optional): The suffix to append to the new column containing
            the similarity scores. Only used if return_scores is True.
            Defaults to "_sim_score".
        use_gpu (bool, optional): Whether to use GPU acceleration for search.
            Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the search results. The returned
            DataFrame will have fewer rows than the original, containing only
            the most relevant matches.

    Raises:
        ValueError: If neither K nor n_rerank is provided, if the retrieval
            model or vector store is not configured, or if the reranker is
            not configured when n_rerank is specified.

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

            >>> df.sem_index('title', 'title_index') ## only needs to be run once; sem_index will save the index to the current directory in "title_index"
            >>> df.load_sem_index('title', 'title_index') ## load the index from disk
            >>> df.sem_search('title', 'AI', K=2)
            100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 81.88it/s]
                                    title
            0  Machine learning tutorial
            1         Data science guide
    """

    def __init__(self, pandas_obj: Any):
        """
        Initialize the semantic search accessor.

        Args:
            pandas_obj (Any): The pandas DataFrame object to attach the accessor to.
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        """
        Validate that the object is a pandas DataFrame.

        Args:
            obj (Any): The object to validate.

        Raises:
            AttributeError: If the object is not a pandas DataFrame.
        """
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        col_name: str,
        query: str,
        K: int | None = None,
        n_rerank: int | None = None,
        return_scores: bool = False,
        suffix: str = "_sim_score",
        use_gpu: bool = False,
        use_approximate: bool = False,
        approx_method: str | None = None, # HNSW32/IVF1024/PQ64
        create_approx_if_missing: bool = True,
        hnsw_ef_search: int = 64, # HNSW查询参数
        ivf_nprobe: int = 8, # IVF查询参数
    ) -> pd.DataFrame:
        # 召回/精度可通过 efSearch/nprobe 调优
        assert not (K is None and n_rerank is None), "K or n_rerank must be provided"
        if K is not None:
            # get retriever model and index
            rm = lotus.settings.rm
            vs = lotus.settings.vs
            if rm is None or vs is None:
                raise ValueError(
                    "The retrieval model must be an instance of RM, and the vector store should be an instance of VS. Please configure a valid retrieval model and vector store using lotus.settings.configure()"
                )

            # Use GPU vector store if requested
            if use_gpu:
                try:
                    from lotus.vector_store import FaissGPUVS
                    from lotus.config import get_gpu_config, gpu_operation
                    
                    config = get_gpu_config()
                    if config.use_gpu_vector_store and not isinstance(vs, FaissGPUVS):
                        # Switch to GPU vector store
                        vs = FaissGPUVS(
                            factory_string=config.gpu_index_factory,
                            metric=getattr(__import__('faiss'), config.gpu_metric, None) or vs.metric
                        )
                        lotus.settings.vs = vs
                except ImportError:
                    lotus.logger.warning("GPU acceleration not available for search, using CPU")

            col_index_dir = self._obj.attrs["index_dirs"][col_name]

            # Determine which index to use (exact vs approximate)
            search_vs = vs
            effective_index_dir = col_index_dir

            if use_approximate:
                # When approximate search is requested, prefer CPU FAISS (GPU HNSW not always available)
                if use_gpu:
                    lotus.logger.warning("use_approximate=True with use_gpu=True: proceeding with CPU index for approximate search")
                try:
                    import faiss  # type: ignore
                    from lotus.vector_store import FaissVS
                except Exception:
                    lotus.logger.warning("FAISS not available, falling back to exact search index")
                else:
                    # Choose default method if not provided
                    method = approx_method if approx_method else "HNSW32"
                    approx_dir = f"{col_index_dir}__{method.lower()}"

                    # Build approximate index from saved embeddings if missing
                    if not os.path.exists(os.path.join(approx_dir, "index")) and create_approx_if_missing:
                        os.makedirs(approx_dir, exist_ok=True)
                        with open(os.path.join(col_index_dir, "vecs"), "rb") as fp:
                            embeddings = pickle.load(fp)
                        # Create FAISS index via factory string
                        dim = embeddings.shape[1]
                        faiss_index = faiss.index_factory(dim, method, vs.metric)
                        faiss_index.add(embeddings)
                        faiss.write_index(faiss_index, os.path.join(approx_dir, "index"))
                        # Copy vectors file for consistency with VS expectations
                        with open(os.path.join(approx_dir, "vecs"), "wb") as fp:
                            pickle.dump(embeddings, fp)

                    # Use a local FaissVS bound to the approximate index directory
                    local_vs = FaissVS(factory_string=method, metric=vs.metric)
                    try:
                        local_vs.load_index(approx_dir)
                        # Tune HNSW/IVF runtime params if applicable
                        try:
                            # HNSW: set efSearch
                            if hasattr(local_vs.faiss_index, "hnsw"):
                                local_vs.faiss_index.hnsw.efSearch = hnsw_ef_search  # type: ignore[attr-defined]
                            # IVF: set nprobe
                            if hasattr(local_vs.faiss_index, "nprobe"):
                                local_vs.faiss_index.nprobe = ivf_nprobe  # type: ignore[attr-defined]
                        except Exception as _:
                            pass
                        search_vs = local_vs
                        effective_index_dir = approx_dir
                    except Exception:
                        lotus.logger.warning("Failed to load approximate index, falling back to exact index")

            # Load the chosen index directory on the selected vector store
            if search_vs.index_dir != effective_index_dir:
                search_vs.load_index(effective_index_dir)
            assert search_vs.index_dir == effective_index_dir

            df_idxs = self._obj.index
            cur_min = len(df_idxs)
            K = min(K, cur_min)
            search_K = K
            
            # 优化1: 预计算查询向量，避免重复计算
            query_vectors = rm.convert_query_to_query_vector(query)
            
            # 优化2: 使用集合进行O(1)查找，替代O(n)的pandas Index查找
            df_idxs_set = set(df_idxs)
            
            # 添加最大迭代次数限制，防止无限循环
            max_iterations = 10
            iteration = 0
            
            # Monitor GPU search performance
            operation_name = "sem_search_gpu" if use_gpu else "sem_search_cpu"
            try:
                from lotus.config import gpu_operation
                with gpu_operation(operation_name, data_size=len(df_idxs)):
                    while iteration < max_iterations:
                        vs_output: RMOutput = search_vs(query_vectors, search_K)
                        doc_idxs = vs_output.indices[0]
                        scores = vs_output.distances[0]
                        assert len(doc_idxs) == len(scores)

                        postfiltered_doc_idxs = []
                        postfiltered_scores = []
                        for idx, score in zip(doc_idxs, scores):
                            # 优化2: 使用集合进行O(1)查找
                            if idx in df_idxs_set:
                                postfiltered_doc_idxs.append(idx)
                                postfiltered_scores.append(score)
                                # 提前退出：如果已经找到足够的有效结果
                                if len(postfiltered_doc_idxs) == K:
                                    break

                        if len(postfiltered_doc_idxs) == K:
                            break
                            
                        search_K = search_K * 2
                        iteration += 1
            except ImportError:
                # Fallback without monitoring
                while iteration < max_iterations:
                    vs_output: RMOutput = search_vs(query_vectors, search_K)
                    doc_idxs = vs_output.indices[0]
                    scores = vs_output.distances[0]
                    assert len(doc_idxs) == len(scores)

                    postfiltered_doc_idxs = []
                    postfiltered_scores = []
                    for idx, score in zip(doc_idxs, scores):
                        # 优化2: 使用集合进行O(1)查找
                        if idx in df_idxs_set:
                            postfiltered_doc_idxs.append(idx)
                            postfiltered_scores.append(score)
                            # 提前退出：如果已经找到足够的有效结果
                            if len(postfiltered_doc_idxs) == K:
                                break

                    if len(postfiltered_doc_idxs) == K:
                        break
                        
                    search_K = search_K * 2
                    iteration += 1

            new_df = self._obj.loc[postfiltered_doc_idxs]
            new_df.attrs["index_dirs"] = self._obj.attrs.get("index_dirs", None)

            if return_scores:
                new_df["vec_scores" + suffix] = postfiltered_scores
        else:
            new_df = self._obj

        if n_rerank is not None:
            if lotus.settings.reranker is None:
                raise ValueError("Reranker not found in settings")

            docs = new_df[col_name].tolist()
            reranked_output: RerankerOutput = lotus.settings.reranker(query, docs, n_rerank)
            reranked_idxs = reranked_output.indices
            new_df = new_df.iloc[reranked_idxs]

        return new_df
