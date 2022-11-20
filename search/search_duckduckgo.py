from typing import List
import urllib.parse
import requests
from bs4 import BeautifulSoup
from .search_interface import SearchInterface


__DUCKDUCKGO_SEARCH_URL__ = "https://duckduckgo.com/html/?q={query}"


class SearchDuckDuckGo(SearchInterface):
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
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
        }
        request = requests.get(url, headers)
        document = BeautifulSoup(request.content, "html.parser")
        results = document.select("#links .results_links")
        snippets = [
            result.select_one(".result__snippet").text
            for result in results
        ]
        return snippets[:self.keep_top_n]

    def update(self, document: str, amendment: str) -> None:
        pass
