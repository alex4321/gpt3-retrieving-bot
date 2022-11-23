from robot import CompletionReactionInterface, CompletionReaction
from .base_logger import BaseLogger
from .channels import LoggerChannelInterface


class CompletionReactionLogger(CompletionReactionInterface, BaseLogger):
    def __init__(self, logger_name: str, logger_channel: LoggerChannelInterface, reaction: CompletionReactionInterface) -> None:
        CompletionReactionInterface.__init__(self)
        BaseLogger.__init__(self, logger_name, logger_channel)
        self.reaction = reaction

    def check(self, completion: str, variables: dict) -> CompletionReaction:
        result = self.reaction.check(completion, variables)
        self.log(
            "Check reaction",
            {
                "completion": completion,
                "variables": variables,
                "result": {
                    "answer": result.answer,
                    "stop": result.stop,
                }
            }
        )
        return result
