from typing import Any

import pandas as pd

import lotus
from lotus.cache import operator_cache
from lotus.models import RM
from lotus.types import RMOutput
from lotus.vector_store import VS


@pd.api.extensions.register_dataframe_accessor("sem_sim_join")
class SemSimJoinDataframe:
    """
    Perform semantic similarity join on the DataFrame.

    Args:
        other (pd.DataFrame): The other DataFrame to join with.
        left_on (str): The column name to join on in the left DataFrame.
        right_on (str): The column name to join on in the right DataFrame.
        K (int): The number of nearest neighbors to search for.
        lsuffix (str): The suffix to append to the left DataFrame.
        rsuffix (str): The suffix to append to the right DataFrame.
        score_suffix (str): The suffix to append to the similarity score column.
        use_gpu (bool): Whether to use GPU acceleration for similarity join.

    Example:
        >>> import pandas as pd
        >>> import lotus
        >>> from lotus.models import RM, VS
        >>> from lotus.vector_store import FaissVS

        >>> lm = LM(model="gpt-4o-mini")
        >>> rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
        >>> vs = FaissVS()

        >>> lotus.settings.configure(lm=lm, rm=rm, vs=vs)

        >>> df1 = pd.DataFrame({
                'article': ['Machine learning tutorial', 'Data science guide', 'Python basics', 'AI in finance', 'Cooking healthy food', "Recipes for the holidays"],
            })
        >>> df2 = pd.DataFrame({
                'category': ['Computer Science', 'AI', 'Cooking'],
            })

        >>> df1.sem_index("article", "article_index")
        >>> df2.sem_index("category", "category_index")

        Example 1: sem_sim_join, K=1, join each article with the most similar category
        >>> df1.sem_sim_join(df2, "article", "category", K=1)
                            article   _scores           category
        0  Machine learning tutorial  0.834617  Computer Science
        1         Data science guide  0.820131  Computer Science
        2              Python basics  0.834945  Computer Science
        3              AI in finance  0.875249                AI
        4       Cooking healthy food  0.890393           Cooking
        5   Recipes for the holidays  0.786058           Cooking

        Example 2: sem_sim_join, K=2, join each article with the top 2 most similar categories
        >>> df1.sem_sim_join(df2, "article", "category", K=2)
                                article   _scores          category
        0  Machine learning tutorial  0.834617  Computer Science
        0  Machine learning tutorial  0.817893                AI
        1         Data science guide  0.820131  Computer Science
        1         Data science guide  0.785335                AI
        2              Python basics  0.834945  Computer Science
        2              Python basics  0.770674                AI
        3              AI in finance  0.875249                AI
        3              AI in finance  0.798493  Computer Science
        4       Cooking healthy food  0.890393           Cooking
        4       Cooking healthy food  0.755058  Computer Science
        5   Recipes for the holidays  0.786058           Cooking
        5   Recipes for the holidays  0.712726  Computer Science
    """

    def __init__(self, pandas_obj: Any):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj: Any) -> None:
        if not isinstance(obj, pd.DataFrame):
            raise AttributeError("Must be a DataFrame")

    @operator_cache
    def __call__(
        self,
        other: pd.DataFrame,
        left_on: str,
        right_on: str,
        K: int,
        lsuffix: str = "",
        rsuffix: str = "",
        score_suffix: str = "",
        keep_index: bool = False,
        use_gpu: bool = False,
        restrict_to_right_ids: bool = False,
        max_search_expansion: int = 10,
    ) -> pd.DataFrame:
        """Perform semantic similarity join between two DataFrames.

        This implementation prioritizes performance by avoiding unnecessary
        subset searches on the vector store. By default, it performs a full-index
        search on the right-hand index and post-filters results to retain only
        rows present in ``other``. This prevents rebuilding a temporary index per
        request (which is particularly expensive on GPU). When ``restrict_to_right_ids``
        is True, a subset search is performed using the exact ``other`` row ids.

        The method also uses an iterative expansion strategy: if full-index search
        does not return enough in-subset matches (i.e., present in ``other``) for a
        query, it increases the search depth and retries up to ``max_search_expansion``
        times. This balances correctness (return K matches when possible) and
        efficiency (avoid creating per-request temporary indices).

        Args:
            other: The right DataFrame to join with.
            left_on: Column name to join on in the left DataFrame.
            right_on: Column name to join on in the right DataFrame.
            K: Number of nearest neighbors to retrieve per left row.
            lsuffix: Suffix to append to overlapping left column names.
            rsuffix: Suffix to append to overlapping right column names.
            score_suffix: Suffix to append to the similarity score column name.
            keep_index: Whether to keep the temporary id columns in the output.
            use_gpu: Whether to enable GPU-accelerated vector search.
            restrict_to_right_ids: If True, force subset search restricted to
                the ids of ``other``. Default False to avoid costly temporary
                index rebuilds and favor full-index search + post-filtering.
            max_search_expansion: Maximum number of search-depth doublings when
                full-index search doesn't yield enough in-subset results.

        Returns:
            A DataFrame containing the joined rows with similarity scores.
        """
        if isinstance(other, pd.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = pd.DataFrame({other.name: other})

        rm = lotus.settings.rm
        vs = lotus.settings.vs
        if not isinstance(rm, RM) or not isinstance(vs, VS):
            raise ValueError(
                "The retrieval model must be an instance of RM, and the vector store must be an instance of VS. Please configure a valid retrieval model or vector store using lotus.settings.configure()"
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
                lotus.logger.warning("GPU acceleration not available for sim_join, using CPU")

        # load query embeddings from index if they exist
        if left_on in self._obj.attrs.get("index_dirs", []):
            query_index_dir = self._obj.attrs["index_dirs"][left_on]
            if vs.index_dir != query_index_dir:
                vs.load_index(query_index_dir)
            assert vs.index_dir == query_index_dir
            try:
                queries = vs.get_vectors_from_index(query_index_dir, self._obj.index)
            except NotImplementedError:
                queries = self._obj[left_on]
        else:
            queries = self._obj[left_on]

        # load index to search over
        try:
            col_index_dir = other.attrs["index_dirs"][right_on]
        except KeyError:
            raise ValueError(f"Index directory for column {right_on} not found in DataFrame")
        if vs.index_dir != col_index_dir:
            vs.load_index(col_index_dir)
        assert vs.index_dir == col_index_dir

        query_vectors = rm.convert_query_to_query_vector(queries)

        # Determine search mode:
        # - Default (fast path): perform full-index search, then post-filter to
        #   keep only results in the right DataFrame ("other"). This avoids
        #   building a temporary index per call.
        # - Subset mode: when explicitly requested via restrict_to_right_ids=True,
        #   restrict search to right_ids. This may rebuild a temporary index in the
        #   current VS implementation and is slower, but guarantees that exactly the
        #   specified subset is searched.
        right_ids = list(other.index)
        other_index_set = set(other.index)

        join_results: list[tuple[Any, Any, Any]] = []

        operation_name = "sem_sim_join_gpu" if use_gpu else "sem_sim_join_cpu"

        if restrict_to_right_ids:
            # Subset-restricted search path (may be slower due to temporary index)
            try:
                from lotus.config import gpu_operation
                with gpu_operation(operation_name, data_size=len(self._obj) * len(other)):
                    vs_output: RMOutput = vs(query_vectors, K, ids=right_ids)
                    distances = vs_output.distances
                    indices = vs_output.indices
            except ImportError:
                vs_output = vs(query_vectors, K, ids=right_ids)
                distances = vs_output.distances
                indices = vs_output.indices

            # Results already restricted to right_ids; just assemble tuples
            for q_idx, res_ids in enumerate(indices):
                for i, res_id in enumerate(res_ids):
                    if res_id != -1 and res_id in other_index_set:
                        join_results.append((self._obj.index[q_idx], res_id, distances[q_idx][i]))
        else:
            # Full-index search with iterative expansion. We gradually increase
            # search depth until we collect up to K matches per query that are
            # present in other_index_set, or until running out of attempts.
            search_K = max(1, K)
            iteration = 0

            # Pre-allocate container for per-query accumulators
            per_query_matches: list[list[tuple[int, float]]] = [list() for _ in range(len(self._obj.index))]

            while iteration < max_search_expansion:
                try:
                    from lotus.config import gpu_operation
                    with gpu_operation(operation_name, data_size=len(self._obj) * search_K):
                        vs_output = vs(query_vectors, search_K)
                        distances = vs_output.distances
                        indices = vs_output.indices
                except ImportError:
                    vs_output = vs(query_vectors, search_K)
                    distances = vs_output.distances
                    indices = vs_output.indices

                # Fill per-query accumulators with in-subset hits only, up to K
                all_satisfied = True
                for q_idx, res_ids in enumerate(indices):
                    if len(per_query_matches[q_idx]) >= K:
                        continue
                    for i, res_id in enumerate(res_ids):
                        if res_id != -1 and res_id in other_index_set:
                            per_query_matches[q_idx].append((res_id, float(distances[q_idx][i])))
                            if len(per_query_matches[q_idx]) == K:
                                break
                    if len(per_query_matches[q_idx]) < K:
                        all_satisfied = False

                if all_satisfied:
                    break

                # Expand search depth and try again
                search_K = search_K * 2
                iteration += 1

            # Materialize final join_results from per_query_matches
            for q_idx, pairs in enumerate(per_query_matches):
                for res_id, score in pairs:
                    join_results.append((self._obj.index[q_idx], res_id, score))

        df1 = self._obj.copy()
        df2 = other.copy()
        df1["_left_id"] = df1.index
        df2["_right_id"] = df2.index
        temp_df = pd.DataFrame(join_results, columns=["_left_id", "_right_id", "_scores" + score_suffix])
        joined_df = df1.join(
            temp_df.set_index("_left_id"),
            how="right",
            on="_left_id",
        ).join(
            df2.set_index("_right_id"),
            how="left",
            on="_right_id",
            lsuffix=lsuffix,
            rsuffix=rsuffix,
        )
        if not keep_index:
            joined_df.drop(columns=["_left_id", "_right_id"], inplace=True)

        return joined_df
