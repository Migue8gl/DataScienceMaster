from abc import ABC, abstractmethod
from typing import Dict, List


class IChunkStrategy(ABC):
    """
    Abstract interface for a text chunking strategy.

    This interface defines the methods that any text chunking strategy implementation must provide.
    A chunking strategy is responsible for breaking down a large text into smaller,
    manageable chunks while preserving context and meaning as much as possible.
    """

    @abstractmethod
    def chunk(self, text: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Split a text into smaller chunks according to the strategy's implementation.

        Args:
            text: The input text to be chunked with his content

        Returns:
            List[Dict[str, str]]: A List of dicts with filename and text chunk
        """
        pass
