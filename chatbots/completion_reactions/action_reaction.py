"""
Action command postprocessor
"""
from chatbots.robot import CompletionReaction, CompletionReactionInterface
from .utils import split_completion, COMMAND_ACTION


class ActionCompletionReaction(CompletionReactionInterface):
    """
    Action command postprocessor
    """
    def check(self, completion: str, variables: dict) -> CompletionReaction:
        for command in split_completion(completion):
            if command.type == COMMAND_ACTION:
                return CompletionReaction(command.args, True)
        return CompletionReaction(None, False)
