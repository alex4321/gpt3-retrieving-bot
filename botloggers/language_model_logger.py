from lm_utils import LanguageModelInterface
from .channels import LoggerChannelInterface
from .base_logger import BaseLogger


class LMLogger(LanguageModelInterface, BaseLogger):
    def __init__(self, logger_name: str, logger_channel: LoggerChannelInterface, language_model: LanguageModelInterface) -> None:
        LanguageModelInterface.__init__(self)
        BaseLogger.__init__(self, logger_name, logger_channel)
        self.language_model = language_model
    
    def complete(self, prompt: str) -> str:
        result = self.language_model.complete(prompt)
        self.log(
            "Language model completion",
            {
                "prompt": prompt,
                "response": result
            }
        )
        return result
