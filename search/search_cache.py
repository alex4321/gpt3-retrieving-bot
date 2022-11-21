import os
import json
from .search_interface import SearchInterface
from typing import List


# TODO: non-file cache
class SearchCache(SearchInterface):
    def __init__(self, fname: str, search: SearchInterface) -> None:
        super().__init__()
        self.fname = fname
        self.cache = {}
        if os.path.exists(self.fname):
            with open(self.fname, "r") as src:
                self.cache = json.load(src)
        self.search_system = search
    
    def update(self, document: str, amendment: str) -> None:
        self.search_system.update(document, amendment)
    
    def search(self, query: str) -> List[str]:
        if query not in self.cache:
            self.cache[query] = self.search_system.search(query)
            with open(self.fname, "w") as target:
                json.dump(self.cache, target)
        return self.cache[query]
