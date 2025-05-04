from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
import numpy as np

class IRetrieverStrategy(ABC):
    """
    Abstract interface for a retriever strategy.
    This interface defines the methods that any retriever strategy implementation must provide.
    A retriever strategy is responsible for retrieving relevant documents from a data store
    based on a given query.
    """
    @abstractmethod
    def retrieve(
        self, store: Dict[str, Any], query: Union[str, np.ndarray], k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant documents from the store based on the query.
        Args:
            store: A dictionary containing the data store, typically with an index and documents
            query: The embedding of the query string or a raw query string
            k: The number of documents to retrieve
        Returns:
            List[str]: A list of retrieved documents, ordered by relevance
        """
        pass