"""
Answer command postprocessor
"""
from robot import CompletionReaction, CompletionReactionInterface
from .utils import split_completion, COMMAND_ANSWER


class AnswerCompletionReaction(CompletionReactionInterface):
    """
    Answer command postprocessor
    """
    def check(self, completion: str, variables: dict) -> CompletionReaction:
        for command in split_completion(completion):
            if command.type == COMMAND_ANSWER:
                return CompletionReaction(command.args, True)
        return CompletionReaction(None, False)
