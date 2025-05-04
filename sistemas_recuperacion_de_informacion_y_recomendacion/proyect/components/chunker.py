from typing import Dict, List

from interfaces.chunker_interface import IChunkStrategy


class Chunker:
    def __init__(self, chunker_strategy: IChunkStrategy):
        self.chunker = chunker_strategy
        
    def __str__(self):
        return self.chunker.__str__()

    def chunk(self, text: Dict[str, str]) -> List[Dict[str, str]]:
        return self.chunker.chunk(text)
