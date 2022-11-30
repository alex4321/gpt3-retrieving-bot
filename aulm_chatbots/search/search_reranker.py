from typing import List
from dataclasses import dataclass
from .search_interface import SearchInterface
from sentence_transformers import CrossEncoder


@dataclass
class SearchRerankerItem:
    search: SearchInterface
    updateable: bool


class SearchReranker(SearchInterface):
    def __init__(self, model: CrossEncoder, search_systems: List[SearchRerankerItem], batch_size: int, top_n: int, score_threshold: float = 0.0) -> None:
        self.model = model
        self.search_systems = search_systems
        self.batch_size = batch_size
        self.top_n = top_n
        self.score_threshold = score_threshold
    
    def add_documents(self, documents: List[str]) -> None:
        for system in self.search_systems:
            if system.updateable:
                system.search.add_documents(documents)
    
    def search(self, query: str) -> List[str]:
        options = []
        for system in self.search_systems:
            options += system.search.search(query)
        pairs = [
            (query, text)
            for text in options
        ]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        top_indices = (-scores).argsort()[:self.top_n]
        return [
            options[index]
            for index in top_indices
            if scores[index] >= self.score_threshold
        ]
