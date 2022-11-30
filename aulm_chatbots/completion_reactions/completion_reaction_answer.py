from .completion_reaction import CompletionReaction, CompletionReactionResult
from ..hints_dict import HintsDict


class CompletionReactionAnswer(CompletionReaction):
    def get_prefix(self) -> str:
        return "Answer:"

    def process(self, completion: str, hints: HintsDict) -> CompletionReactionResult:
        return CompletionReactionResult(
            completion,
            hints,
            False
        )
