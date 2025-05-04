import re
from typing import Any, Dict, List, Optional

import numpy as np

from interfaces.chunker_interface import IChunkStrategy


class FixedSizeChunker(IChunkStrategy):
    def __init__(self, max_length: int, overlap: Optional[int] = 0):
        if max_length <= overlap:
            raise ValueError(
                "Overlapping must not be equal or greater than maximum length."
            )
        self.max_length = max_length
        self.overlap = overlap

    def __str__(self):
        return "fized_size"

    def chunk(self, text: Dict[str, str]) -> List[Dict[str, str]]:
        key, context = list(text.items())[0]
        text_stripped = context.strip()
        return [
            {key: text_stripped[i : i + self.max_length]}
            for i in range(0, len(text_stripped), self.max_length - self.overlap)
        ]


class RecursiveChunker(IChunkStrategy):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __str__(self):
        return "recursive"

    def chunk(self, text: Dict[str, str]) -> List[Dict[str, str]]:
        result = self._recursive_chunk(text, 0)
        final_result = []
        for chunk in result:
            key, content = list(chunk.items())[0]
            if len(content) <= self.max_length:
                final_result.append(chunk)
            else:
                final_result.extend(self._force_split({key: content}))
        return final_result

    def _recursive_chunk(
        self, text: Dict[str, str], recursion_level: int
    ) -> List[Dict[str, str]]:
        key, context = list(text.items())[0]
        if not context.strip():
            return []

        splits = self._split_text(text, recursion_level)
        if not splits:
            if len(context) <= self.max_length:
                return [{key: context}]
            else:
                return self._force_split({key: context})

        max_length_in_split = max((len(list(s.values())[0]) for s in splits), default=0)
        if max_length_in_split > self.max_length:
            result = []
            for split in splits:
                split_key, split_text = list(split.items())[0]
                if len(split_text) > self.max_length:
                    next_level = recursion_level + 1
                    if next_level <= 2:
                        result.extend(self._recursive_chunk(split, next_level))
                    else:
                        forced_chunks = self._force_split(split)
                        result.extend(forced_chunks)
                else:
                    result.append(split)
            return result
        else:
            return splits

    def _split_text(
        self, text: Dict[str, str], recursion_level: int
    ) -> List[Dict[str, str]]:
        key, context = list(text.items())[0]
        if not context.strip():
            return []

        if recursion_level == 0:
            chunks = context.split("\n\n")
            return [{key: chunk} for chunk in chunks if chunk.strip()]
        elif recursion_level == 1:
            chunks = re.split(r"(?<=\.)\s+", context.strip())
            return [{key: chunk} for chunk in chunks if chunk.strip()]
        else:
            chunks = context.split()
            return [{key: chunk} for chunk in chunks if chunk.strip()]

    def _force_split(self, text: Dict[str, str]) -> List[Dict[str, str]]:
        key, context = list(text.items())[0]
        chunks = []

        for i in range(0, len(context), self.max_length):
            chunk_text = context[i : i + self.max_length]
            if chunk_text.strip():
                chunks.append({key: chunk_text})

        if not chunks:
            return [{key: context[: self.max_length]}]

        for chunk in chunks:
            _, chunk_text = list(chunk.items())[0]
            if len(chunk_text) > self.max_length:
                print(
                    f"Warning: Found chunk exceeding max_length: {len(chunk_text)} > {self.max_length}"
                )

        return chunks


class SemanticChunker(IChunkStrategy):
    def __init__(
        self, embedding_model: Any, threshold: float = 0.9, max_chunk_size: int = 2000
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.max_chunk_size = max_chunk_size
    
    def __str__(self):
        return "semantic"

    def chunk(self, text: Dict[str, str]) -> List[Dict[str, str]]:
        key, context = list(text.items())[0]

        splits = re.split(r"(?<=\.)\s+", context.strip())
        splits = [split for split in splits if split.strip()]

        if not splits:
            return []

        split_queries = [f"query: {split}" for split in splits]
        embeddings = self.embedding_model.encode(
            split_queries, normalize_embeddings=True
        )

        chunks = []
        curr_chunk_start_idx = 0

        for idx in range(len(splits) - 1):
            if idx == curr_chunk_start_idx:
                current_chunk = splits[idx]
                current_embedding = embeddings[idx].copy()
            else:
                current_chunk_splits = splits[curr_chunk_start_idx : idx + 1]
                current_chunk = " ".join(current_chunk_splits)
                chunk_embeddings = embeddings[curr_chunk_start_idx : idx + 1]
                current_embedding = np.mean(chunk_embeddings, axis=0)
                current_embedding = current_embedding / np.linalg.norm(
                    current_embedding
                )

            next_embedding = embeddings[idx + 1]

            similarity_score = np.dot(current_embedding, next_embedding)

            next_sentence = splits[idx + 1]
            size_if_added = len(current_chunk) + len(next_sentence) + 1

            if similarity_score < self.threshold or size_if_added > self.max_chunk_size:
                chunks.append({key: current_chunk})
                curr_chunk_start_idx = idx + 1

        if curr_chunk_start_idx < len(splits):
            last_chunk = " ".join(splits[curr_chunk_start_idx:])
            chunks.append({key: last_chunk})

        return chunks
