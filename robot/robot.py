"""
Bot main part - prompt filler/language model/LM completions handlers glue
"""
from typing import List, Dict
from lm_utils import PromptFiller, LanguageModelInterface
from .completion_reaction_interface import CompletionReactionInterface


__MAX_ITERS__ = 5
__ANSWER_FORCE_COMPLETION__ = "Answer:"


class Robot:
    """
    Bot main part - prompt filler/language model/LM completions handlers glue
    """
    def __init__(self, prompt_filler: PromptFiller, language_model: LanguageModelInterface,
                 filler_vars: Dict[str, str],
                 completion_reactions: List[CompletionReactionInterface],
                 max_iters: int = __MAX_ITERS__) -> None:
        self.prompt_filler = prompt_filler
        self.language_model = language_model
        self.filler_vars = filler_vars
        self.completion_reactions = completion_reactions
        self.max_iters = max_iters

    def _filter_hints(self, hints: List[str], dialogue: List[str]) -> List[str]:
        return [
            hint.strip()
            for hint in hints
            if hint.split(" : ")[-1] not in "\n".join(dialogue)
        ]

    def _join_string_list(self, texts: List[str]) -> str:
        if texts:
            texts = [item.strip() for item in texts]
            return ("\n - " + "\n - ".join(texts)).strip()
        return ""

    def _generate_lm_completion(self, prompt: str, force: bool) -> str:
        if force:
            prompt = prompt.strip() + "\n" + __ANSWER_FORCE_COMPLETION__
        completion = self.language_model.complete(prompt.strip()).strip()
        if force:
            completion = __ANSWER_FORCE_COMPLETION__ + " " + completion
        return completion

    def response(self, hints: List[str], dialogue: List[str], retort: str) -> str:
        """
        Generate robot response
        """
        result = None
        force_answer = False
        previous_completion = None
        for i in range(self.max_iters):
            hints_to_use = self._filter_hints(hints, dialogue)
            filler_vars = dict(
                self.filler_vars,
                **{
                    "HINTS": self._join_string_list(hints_to_use),
                    "DIALOGUE": self._join_string_list(dialogue),
                    "RETORT": retort.strip()
                }
            )
            prompt = self.prompt_filler.fill(filler_vars)
            if i == self.max_iters - 1:
                force_answer = True
            completion = self._generate_lm_completion(prompt, force_answer)
            if completion == "":
                force_answer = True
            elif previous_completion is not None and completion == previous_completion:
                force_answer = True
            previous_completion = completion
            breaker = False
            for reaction in self.completion_reactions:
                reaction_vars = {
                    "hints": hints,
                }
                reaction_response = reaction.check(completion, reaction_vars)
                if reaction_response.answer is not None:
                    result = reaction_response.answer
                if reaction_response.stop:
                    breaker = True
                    break
            if breaker:
                break
        assert result is not None
        return result.strip().strip("-")
