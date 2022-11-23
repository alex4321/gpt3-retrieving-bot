"""
Logging decorator for search systems
"""
from typing import List
from search import SearchInterface
from .base_logger import BaseLogger
from .channels import LoggerChannelInterface


class SearchLogger(SearchInterface, BaseLogger):
    """
    Logging decorator for search systems
    """
    def __init__(self, logger_name: str, logger_channel: LoggerChannelInterface,
                 search: SearchInterface) -> None:
        SearchInterface.__init__(self)
        BaseLogger.__init__(self, logger_name, logger_channel)
        self.search_system = search

    def update(self, document: str, amendment: str) -> None:
        self.search_system.update(document, amendment)
        self.log(
            "Update documents",
            {
                "document": document,
                "amendment": amendment,
            }
        )

    def search(self, query: str) -> List[str]:
        result = self.search_system.search(query)
        self.log(
            "Search request",
            {
                "query": query,
                "result": result,
            }
        )
        return result
