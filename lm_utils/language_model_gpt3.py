from typing import Union
import openai
from .language_model_interface import LanguageModelInterface


__COMPLETION_DEFAULTS__ = {
    "model": "text-davinci-002",
    "temperature": 1.0,
    "max_tokens": 150,
    "frequency_penalty": 1.0,
    "presence_penalty": 1.0,
    "max_tokens": 60,
    "top_p": 1,
    "stop": [" Endseparator:"],
}


class LanguageModelGPT3(LanguageModelInterface):
    def __init__(self, completion_params: Union[dict, None] = None) -> None:
        super(LanguageModelGPT3, self).__init__()
        if completion_params is None:
            completion_params = {}
        self.completion_params = dict(__COMPLETION_DEFAULTS__, **completion_params)
    
    def complete(self, prompt: str) -> str:
        completion = openai.Completion.create(
            prompt=prompt.strip(),
            **self.completion_params
        )
        if len(completion.choices) == 0:
            return ""
        return completion.choices[0].text.strip().replace("Endseparator:", "").strip()
