"""
Base search system interface
"""
from typing import List


class SearchInterface:
    """
    Base search system interface
    """
    def search(self, query: str) -> List[str]:
        """
        Perform search through document collection
        """
        raise NotImplementedError()

    def update(self, document: str, amendment: str) -> None:
        """
        Update document collection
        """
        raise NotImplementedError()
