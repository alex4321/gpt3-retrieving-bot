from .channels import LoggerChannelInterface
from .base_logger import BaseLogger
from lm_utils import LanguageModelInterface


class LMLogger(LanguageModelInterface, BaseLogger):
    def __init__(self, logger_name: str, logger_channel: LoggerChannelInterface, lm: LanguageModelInterface) -> None:
        LanguageModelInterface.__init__(self)
        BaseLogger.__init__(self, logger_name, logger_channel)
        self.lm = lm
    
    def complete(self, prompt: str) -> str:
        result = self.lm.complete(prompt)
        self.log(
            "Language model completion",
            {
                "prompt": prompt,
                "response": result
            }
        )
        return result
