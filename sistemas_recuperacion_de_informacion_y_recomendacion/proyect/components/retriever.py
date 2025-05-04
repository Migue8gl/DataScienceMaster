from typing import Any, Dict, List

from interfaces.retriever_interface import IRetrieverStrategy


class Retriever:
    def __init__(self, retriever_strategy: IRetrieverStrategy):
        self.retriever = retriever_strategy

    def retrieve(self, store: Dict[str, Any], query: str, k: int = 5) -> List[str]:
        return self.retriever.retrieve(store, query, k)
