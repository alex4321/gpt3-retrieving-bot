from .completion_reaction import CompletionReaction, CompletionReactionResult
from ..search import SearchInterface
from ..hints_dict import HintsDict


class CompletionReactionSearch(CompletionReaction):
    def __init__(self, search: SearchInterface) -> None:
        super().__init__()
        self.search = search

    def get_prefix(self) -> str:
        return "Search:"

    def get_history_key(self, request: str) -> str:
        return request

    def process(self, completion: str, hints: HintsDict) -> CompletionReactionResult:
        requests = completion.split("\n")
        requests = [request.strip().strip("-") for request in requests]
        requests = [request for request in requests if request]
        requests = [request for request in requests if self.get_history_key(request) not in hints]
        for request in requests:
            hints[self.get_history_key(request)] = self.search.search(request)
        return CompletionReactionResult(
            None,
            hints,
            True
        )


class CompletionReactionStorySearch(CompletionReactionSearch):
    def get_prefix(self) -> str:
        return "Story Search:"

    def get_history_key(self, request: str) -> str:
        return f"(Story) {request}"
