from typing import List
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    score: float


class LanguageModelInterface:
    def complete(self, prompt: str) -> List[Completion]:
        raise NotImplementedError()
