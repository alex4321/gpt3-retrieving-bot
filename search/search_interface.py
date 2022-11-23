from typing import List


class SearchInterface:
    def search(self, query: str) -> List[str]:
        raise NotImplementedError()
        
    def update(self, document: str, amendment: str) -> None:
        raise NotImplementedError()
