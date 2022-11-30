"""
DuckDuckGo-based search through internet
"""
import time
from typing import List
import urllib.parse
from bs4 import BeautifulSoup
import requests
from .search_interface import SearchInterface


__TIMEOUT__ = 1.0
__REQUESTS_TIMEOUT__ = 60.0
__DUCKDUCKGO_SEARCH_URL__ = "https://duckduckgo.com/html/?q={query}"
__FIREFOX_UA__ = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'


class SearchDuckDuckGo(SearchInterface):
    """
    DuckDuckGo-based search through internet
    """
    def __init__(self, keep_top_n: int) -> None:
        self.keep_top_n = keep_top_n

    def search(self, query: str) -> List[str]:
        url = __DUCKDUCKGO_SEARCH_URL__.format(
            query=urllib.parse.quote(query)
        )
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
            'User-Agent': __FIREFOX_UA__,
        }
        request = requests.get(url, headers, timeout=__REQUESTS_TIMEOUT__)
        document = BeautifulSoup(request.content, "html.parser")
        results = document.select("#links .results_links")
        snippets = [
            result.select_one(".result__snippet").text
            for result in results
        ]
        time.sleep(__TIMEOUT__)
        return snippets[:self.keep_top_n]

    def add_documents(self, documents: List[str]) -> None:
        pass
