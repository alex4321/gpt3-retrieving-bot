"""
LM prompt template filler
"""
from typing import Dict
import os


__MODULE_DIR__ = os.path.dirname(__file__)


class PromptFiller:
    """
    LM prompt template filler
    """
    def __init__(self, prompt_fname: str = os.path.join(__MODULE_DIR__, "prompt.txt")) -> None:
        with open(prompt_fname, "r", encoding="utf-8") as src:
            self.prompt = src.read()

    def fill(self, variables: Dict[str, str]) -> str:
        """
        Fill template with given values
        """
        text = self.prompt
        for name, value in variables.items():
            text = text.replace(f"%{name}%", value)
        return text
