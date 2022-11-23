"""
Basic language model completion postrocessor
"""
from dataclasses import dataclass
from typing import Union


@dataclass
class CompletionReaction:
    """
    LM output postprocessor result
    """
    answer:  Union[str, None]
    stop: bool


class CompletionReactionInterface:
    """
    Basic language model completion postrocessor
    """
    def check(self, completion: str, variables: dict) -> CompletionReaction:
        """
        Check LM output for this  action
        """
        raise NotImplementedError()
