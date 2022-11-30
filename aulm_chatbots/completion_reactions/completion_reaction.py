from dataclasses import dataclass
from typing import Union
from ..hints_dict import HintsDict


@dataclass
class CompletionReactionResult:
    text: Union[str, None]
    hints: HintsDict
    repeat: bool


class CompletionReaction:
    def get_prefix(self) -> str:
        return NotImplementedError()

    def process(self, completion: str, hints: HintsDict) -> CompletionReactionResult:
        raise NotImplementedError()
