from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
import torch  # For Tensor support


class RM(ABC):
    """
    Abstract base class for retrieval models.

    This class defines the interface for retrieval models that can generate
    embeddings for documents and queries. Subclasses must implement the
    `_embed` method to provide the actual embedding functionality.

    Attributes:
        None (abstract base class)
    """

    def __init__(self) -> None:
        """Initialize the retrieval model base class."""
        pass

    @abstractmethod
    def _embed(self, docs: List[str], return_tensor: bool = False) -> Union[NDArray[np.float64], torch.Tensor]:
        """Generate embeddings; return Tensor if requested."""
        pass

    def __call__(self, docs: List[str], return_tensor: bool = False) -> Union[NDArray[np.float64], torch.Tensor]:
        return self._embed(docs, return_tensor=return_tensor)

    def convert_query_to_query_vector(
        self,
        queries: Union[pd.Series, str, Image.Image, List[str], NDArray[np.float64], torch.Tensor],
        return_tensor: bool = False,
    ) -> Union[NDArray[np.float64], torch.Tensor]:
        """
        Convert various query formats to query vectors.

        This method handles different input types and converts them to embedding vectors:
        - String queries: Converted to list and embedded
        - Image queries: Converted to list and embedded (if supported)
        - Pandas Series: Converted to list and embedded
        - List of strings: Directly embedded
        - Numpy arrays: Returned as-is (assumed to be pre-computed vectors)

        Args:
            queries: Query or queries in various formats.

        Returns:
            NDArray[np.float64]: Array of query vectors with shape (num_queries, embedding_dim).
        """
        if isinstance(queries, (str, Image.Image)):
            queries = [queries]

        if isinstance(queries, (np.ndarray, torch.Tensor)):
            query_vectors = queries
            if return_tensor and isinstance(query_vectors, np.ndarray):
                query_vectors = torch.from_numpy(query_vectors)
            elif not return_tensor and isinstance(query_vectors, torch.Tensor):
                query_vectors = query_vectors.numpy()
        else:
            if isinstance(queries, pd.Series):
                queries = queries.tolist()
            query_vectors = self._embed(queries, return_tensor=return_tensor)
        return query_vectors
