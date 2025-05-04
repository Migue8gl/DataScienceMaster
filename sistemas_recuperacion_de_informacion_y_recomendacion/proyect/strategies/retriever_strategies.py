import os
import pickle
from typing import Any, Dict, List, Union

import numpy as np
from rank_bm25 import BM25Okapi

from interfaces.retriever_interface import IRetrieverStrategy
from utils.utils import normalize_minmax


class DenseRetriever(IRetrieverStrategy):
    """
    Dense retriever implementation that uses vector similarity for retrieval.
    """

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def retrieve(
        self, store: Dict[str, Any], query: Union[str, np.ndarray], k: int = 5
    ) -> List[str]:
        index = store.get("index")
        docs = store.get("documents")

        if not index or not docs:
            return []

        if isinstance(query, str) and self.embedding_model:
            query_embedding = self.embedding_model.encode(
                [f"query: {query}"], normalize_embeddings=True
            ).astype("float32")
        else:
            query_embedding = query if isinstance(query, np.ndarray) else None

        if query_embedding is None:
            return []

        dense_ranked_indices = index.search(query_embedding, k)[1]
        return [docs[int(i)] for i in dense_ranked_indices[0]]


class HybridRetriever(IRetrieverStrategy):
    def __init__(
        self, embedding_model=None, alpha: float = 0.5, override_bm25: bool = False
    ):
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.override_bm25 = override_bm25
        self.bm25 = None
        self.tokenized_corpus = None

    def _load_or_create_bm25(self, docs: List[Dict[str, str]]):
        bm25_file_path = "./data/bm25.pkl"
        if not self.override_bm25 and os.path.exists(bm25_file_path):
            try:
                with open(bm25_file_path, "rb") as f:
                    bm25_data = pickle.load(f)
                    self.bm25 = bm25_data.get("bm25")
                    self.tokenized_corpus = bm25_data.get("tokenized_corpus")
                    return
            except Exception as e:
                print(f"Error loading BM25 from file: {e}")

        self.tokenized_corpus = [
            context.lower().split() for doc in docs for context in doc.values()
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        try:
            os.makedirs("./data", exist_ok=True)
            with open(bm25_file_path, "wb") as f:
                pickle.dump(
                    {"bm25": self.bm25, "tokenized_corpus": self.tokenized_corpus}, f
                )
        except Exception as e:
            print(f"Error saving BM25 to file: {e}")

    def retrieve(
        self, store: Dict[str, Any], query: Union[str, np.ndarray], k: int = 5
    ) -> List[str]:
        index = store.get("index")
        docs = store.get("documents")
        if not index or not docs:
            return []

        dense_scores = np.zeros(len(docs))
        sparse_scores = np.zeros(len(docs))

        if isinstance(query, str) and self.embedding_model:
            query_embedding = self.embedding_model.encode(
                [f"query: {query}"], normalize_embeddings=True
            ).astype("float32")
        elif isinstance(query, np.ndarray):
            query_embedding = query
        else:
            query_embedding = None

        if query_embedding is not None:
            scores, indices = index.search(query_embedding, len(docs))
            dense_scores[indices[0]] = scores[0]

        if isinstance(query, str):
            self._load_or_create_bm25(docs)
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            sparse_scores = np.array(bm25_scores)

        if np.max(dense_scores) > 0:
            dense_scores = normalize_minmax(dense_scores)
        if np.max(sparse_scores) > 0:
            sparse_scores = normalize_minmax(sparse_scores)

        final_scores = self.alpha * dense_scores + (1 - self.alpha) * sparse_scores
        top_indices = np.argsort(final_scores)[::-1][:k]

        return [docs[idx] for idx in top_indices]
