from typing import List, Dict
from lm_utils import PromptFiller, LanguageModelInterface
from .completion_reaction_interface import CompletionReactionInterface


__MAX_ITERS__ = 5


class Robot:
    def __init__(self, prompt_filler: PromptFiller, language_model: LanguageModelInterface, 
                 filler_vars: Dict[str, str], completion_reactions: List[CompletionReactionInterface],
                 max_iters: int = __MAX_ITERS__) -> None:
        self.prompt_filler = prompt_filler
        self.language_model = language_model
        self.filler_vars = filler_vars
        self.completion_reactions = completion_reactions
        self.max_iters = max_iters

    def response(self, hints: List[str], dialogue: List[str], retort: str) -> str:
        result = None
        force_answer = False
        for _ in range(self.max_iters):
            filler_vars = dict(
                self.filler_vars,
                **{
                    "HINTS": "\n-" + "\n-".join(hints) if len(hints) > 0 else "None",
                    "DIALOGUE": "-\n" + "\n-".join(dialogue) if len(dialogue) > 0 else "",
                    "RETORT": retort
                }
            )
            prompt = self.prompt_filler.fill(filler_vars)
            if force_answer:
                prompt = prompt.strip() + "\nAnswer:"
            completion = self.language_model.complete(prompt)
            if force_answer:
                completion = "Answer: " + completion
            if completion == "":
                force_answer = True
            breaker = False
            for reaction in self.completion_reactions:
                reaction_vars = {
                    "hints": hints
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
