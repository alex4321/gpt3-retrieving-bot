from typing import List
import requests
import numpy as np
from .language_model import LanguageModelInterface, Completion


_DEFAULT_PARAMS = {
    "model": "text-davinci-003",
    "temperature": 1.0,
    "max_tokens": 150,
    "frequency_penalty": 1.0,
    "presence_penalty": 1.0,
    "top_p": 1,
    "logprobs": 1,
}
_REQUEST_TIMEOUT = 60.0


class LanguageModelGPT3(LanguageModelInterface):
    def __init__(self, key: str, stop: List[str], **kwargs) -> None:
        self.key = key
        self.stop = stop
        self.params = dict(_DEFAULT_PARAMS, **kwargs, stop=stop)

    def complete(self, prompt: str) -> List[Completion]:
        body = dict(self.params, prompt=prompt.strip())
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }
        response = requests.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            json=body,
            timeout=_REQUEST_TIMEOUT
        )
        data = response.json()
        assert response.status_code == 200, data.get("error", {}).get("message",
            f"GPT3 response status code is {response.status_code}"
        )
        result = []
        for choice in data["choices"]:
            choice_text = choice["text"].strip()
            choice_token_logprobs = np.array(choice["logprobs"]["token_logprobs"], dtype=np.float64)
            choice_token_probs = np.exp(choice_token_logprobs)
            choice_token_probs_normed = choice_token_probs ** (1/len(choice_token_probs))
            choice_proba = choice_token_probs_normed.prod()
            choice_result = Completion(choice_text, choice_proba)
            result.append(choice_result)
        return result
