from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class IVectorDB(ABC):
    """
    Abstract interface for a vector database.
    Defines the methods that any class must implement to function
    as a vector database.
    """

    @abstractmethod
    def __init__(
        self, embedding_model: Union[str, SentenceTransformer], path: str = "./data"
    ) -> None:
        """
        Initialize the vector database.

        Args:
            embedding_model: Embedding model to use, can be a string with the model name
                             or an instance of SentenceTransformer
            path: Path where the database data will be stored
        """
        pass

    @staticmethod
    @abstractmethod
    def _create_dir(path: str = "./data") -> bool:
        """
        Create a directory if it doesn't exist.

        Args:
            path: Path of the directory to create

        Returns:
            bool: True if the directory was created or already existed, False in case of error
        """
        pass

    @abstractmethod
    def _generate_embeddings(
        self, documents: List[Dict[str, str]]
    ) -> Tuple[np.ndarray, List[Dict[str, str]]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of dicts of documents to generate embeddings for

        Returns:
            Tuple[np.ndarray, List[Dict[str, str]]]: Array with the generated embeddings and list of chunks
        """
        pass

    @abstractmethod
    def _save(self, index: faiss.Index, documents: List[Dict[str, str]]) -> bool:
        """
        Save documents and index to disk.

        Args:
            documents: List of dicts of documents
            index: FAISS index

        Returns:
            bool: True if saved successfully, False in case of error
        """
        pass

    @abstractmethod
    def _load(self) -> bool:
        """
        Load the database from disk.

        Returns:
            bool: True if loaded successfully, False in case of error
        """
        pass

    @abstractmethod
    def create_index(
        self, documents: List[Dict[str, str]], override: bool = False
    ) -> bool:
        """
        Create a FAISS index for the provided documents or loads the existing db if created.

        Args:
            documents: List of dicts of documents to index
            override: If True, current db will be overrided
        Returns:
            bool: True if index was created or loaded successfully, False in case of error
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[str]:
        """
        Search for the k most similar documents to the query.

        Args:
            query: Query to search for
            k: Number of documents to return

        Returns:
            List[str]: List with the k most similar documents
        """
        pass
