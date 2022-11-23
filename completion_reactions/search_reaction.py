from robot import CompletionReaction, CompletionReactionInterface
from search import SearchInterface
from .utils import split_completion, COMMAND_SEARCH


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
                query_line_no_story = query_line.replace("STORY", "").strip()
                if not query_line or not query_line_no_story:
                    continue
                query_line_searched = any([
                    hint.startswith(query_line_no_story) or hint.startswith(query_line)
                    for hint in hints
                ])
                if query_line_searched:
                    continue
                query_hints = self.search.search(query_line)
                query_hints = [item.strip() for item in query_hints]
                query_hints = [item for item in query_hints if item]
                for query_hint in query_hints:
                    hints.append(f"{query_line_no_story} - {query_hint}")
        return CompletionReaction(None, stop=False)
