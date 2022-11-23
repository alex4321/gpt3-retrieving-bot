"""
Cache for internet search
"""
import json
import os
from typing import List
from .search_interface import SearchInterface


# TODO: non-file cache
class SearchCache(SearchInterface):
    """
    Cache for internet search
    """
    def __init__(self, fname: str, search: SearchInterface) -> None:
        super().__init__()
        self.fname = fname
        self.cache = {}
        if os.path.exists(self.fname):
            with open(self.fname, "r", encoding="utf-8") as src:
                self.cache = json.load(src)
        self.search_system = search

    def update(self, document: str, amendment: str) -> None:
        self.search_system.update(document, amendment)

    def search(self, query: str) -> List[str]:
        if query not in self.cache:
            self.cache[query] = self.search_system.search(query)
            with open(self.fname, "w", encoding="utf-8") as target:
                json.dump(self.cache, target)
        return self.cache[query]
