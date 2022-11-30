from aulm_chatbots.search import SearchDuckDuckGo


def test_search_duckduckgo():
    search = SearchDuckDuckGo(keep_top_n=5)
    results = search.search("Dixie Flatline Gibson")
    assert any([
        "Dixie" in text and "McCoy" in text
        for text in results
    ])
