from robot import CompletionReaction, CompletionReactionInterface
from .utils import split_completion, COMMAND_SEARCH
from search import SearchInterface


class SearchCompletionReaction(CompletionReactionInterface):
    def __init__(self, search: SearchInterface):
        self.search = search

    def check(self, completion: str, variables: dict) -> CompletionReaction:
        hints = variables["hints"]
        for command in split_completion(completion):
            if command.type != COMMAND_SEARCH:
                continue
            query = command.args
            for query_line in query.split("\n"):
                query_line = query_line.strip()
                if not query_line:
                    continue                
                query_line_searched = any([
                    hint.startswith(query_line)
                    for hint in hints
                ])
                query_hints = self.search.search(query_line)
                if query_hints == []:
                    query_hints.append(["not known"])
                for query_hint in query_hints:
                    hints.append(f"{query_line} - {query_hint}")
        return CompletionReaction(None, stop=False)
