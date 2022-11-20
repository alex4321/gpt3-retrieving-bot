from typing import Union
from dataclasses import dataclass


@dataclass
class CompletionReaction:
    answer:  Union[str, None]
    stop: bool


class CompletionReactionInterface:
    def check(self, completion: str, variables: dict) -> CompletionReaction:
        raise NotImplementedError()