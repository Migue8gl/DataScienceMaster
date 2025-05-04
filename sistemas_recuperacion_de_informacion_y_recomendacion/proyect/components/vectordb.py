import os
import pickle
from typing import Dict, List, Tuple, Union

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from components.chunker import Chunker
from components.retriever import Retriever
from interfaces.vectordb_interface import IVectorDB

torch.classes.__path__ = []


class VectorDB(IVectorDB):
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorDB, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        embedding_model: Union[str, SentenceTransformer],
        chunker: Chunker = None,
        retriever: Retriever = None,
        path: str = "./data",
    ) -> None:
        if self._initialized:
            return

        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model

        self.chunker = chunker
        self.retriever = retriever
        self.data = {"index": None, "documents": {}}
        self.path = path
        VectorDB._initialized = True

    @staticmethod
    def _create_dir(path: str = "./data") -> bool:
        try:
            if not os.path.exists(path):
                os.mkdir(path)
            result = True
        except Exception as e:
            print(e)
            result = False
        finally:
            return result

    def _generate_embeddings(
        self, documents: List[Dict[str, str]]
    ) -> Tuple[np.ndarray, List[Dict[str, str]]]:
        chunks = [
            chunk for document in documents for chunk in self.chunker.chunk(document)
        ]
        chunks_with_query = [f"query: {list(chunk)[0]}" for chunk in chunks]
        embeddings = self.embedding_model.encode(
            chunks_with_query, normalize_embeddings=True
        )

        return np.array(embeddings).astype("float32"), chunks

    def _save(self, index: faiss.Index, documents: List[Dict[str, str]]) -> bool:
        try:
            data = {"index": index, "documents": documents}
            self.data = data
            self._create_dir(self.path)
            with open(f"./data/{str(self.chunker)}_db.pkl", "wb") as f:
                pickle.dump(data, f)
            result = True
        except Exception as e:
            print(e)
            result = False
        finally:
            return result

    def _load(self) -> bool:
        try:
            with open(f"./data/{str(self.chunker)}_db.pkl", "rb") as f:
                self.data = pickle.load(f)
            result = True
        except Exception as e:
            print(e)
            result = False
        finally:
            return result

    def create_index(
        self, documents: List[Dict[str, str]], override: bool = False
    ) -> bool:
        if (
            not os.path.exists(os.path.join(self.path, f"{str(self.chunker)}_db.pkl"))
            or override
        ):
            embeddings, chunks = self._generate_embeddings(documents)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            return self._save(index, chunks)
        else:
            return self._load()

    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        return self.retriever.retrieve(self.data, query, k)
