from dataclasses import dataclass
from typing import List, Tuple, Union
from .search import SearchInterface
from .completion_reactions import CompletionReaction
from .language_model import LanguageModelInterface, Completion
from .hints_dict import HintsDict
from .dialogue_list import DialogueList
import os


@dataclass
class CharacterConfig:
    name: str
    description: str


@dataclass
class RobotConfig:
    character: CharacterConfig
    search_system_local: SearchInterface
    language_model: LanguageModelInterface
    completions: List[CompletionReaction]
    hints_keep_top_n: int
    dialogue_keeps_top_n: int
    max_repeats: int = 5
    prompt_fname: str = os.path.join(os.path.dirname(__file__), "prompt.txt")


@dataclass
class Session:
    dialogue: DialogueList
    hints: HintsDict


class Robot:
    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        with open(self.config.prompt_fname, "r", encoding="utf-8") as src:
            self.prompt = src.read()

    def init_session(self) -> Session:
        return Session(
            DialogueList(self.config.dialogue_keeps_top_n),
            HintsDict(self.config.hints_keep_top_n),
        )

    def get_prompt(self, hints: HintsDict, dialogue: DialogueList, phrase: str) -> str:
        filler_variables = {
            "CHARACTER_NAME": self.config.character.name,
            "CHARACTER_DESCRIPTION": self.config.character.description,
            "HINTS": str(hints),
            "DIALOGUE": str(dialogue),
            "RETORT": phrase,
        }
        prompt = self.prompt
        for name, value in filler_variables.items():
            prompt = prompt.replace(f"%{name}%", value)
        return prompt

    def generate_without_forcing_prefixes(self, prompt: str) -> Tuple[Union[Completion, None], Union[CompletionReaction, None], bool]:
        lm_completions = self.config.language_model.complete(
            prompt.strip()
        )
        if not lm_completions:
            return None, None, True
        top_completion = sorted(lm_completions, key=lambda completion: -completion.score)[0]
        continue_with_forced_prefixes = True
        top_reaction = None
        for reaction in self.config.completions:
            prefix = reaction.get_prefix()
            if top_completion.text.startswith(prefix):
                continue_with_forced_prefixes = False
                top_reaction = reaction
                top_completion = Completion(
                    text=top_completion.text.replace(prefix, "").strip(),
                    score=top_completion.score,
                )
        if continue_with_forced_prefixes:
            return None, None, True
        return top_completion, top_reaction, False

    def generate_with_forcing_prefixes(self, prompt: str) -> Tuple[Completion, CompletionReaction, bool]:
        lm_completions = []
        for reaction in self.config.completions:
            prefix_completions += self.config.language_model.complete(
                prompt + reaction.get_prefix()
            )
            lm_completions += [(item, reaction) for item in prefix_completions]
        lm_completions = sorted(
            lm_completions,
            key=lambda item: -item[0].score
        )
        top_completion, top_reaction = lm_completions[0]
        return top_completion, top_reaction, False

    def response(self, session: Session, phrase: str) -> Tuple[Union[str, None], Session]:
        hints = session.hints
        dialogue = session.dialogue
        force_prefixes = False
        for _ in range(self.config.max_repeats):
            prompt = self.get_prompt(hints, dialogue, phrase.strip())
            if not force_prefixes:
                top_completion, top_reaction, force_prefixes = self.generate_without_forcing_prefixes(prompt)
            else:
                top_completion, top_reaction, force_prefixes = self.generate_with_forcing_prefixes(prompt)
            top_completion_text = top_completion.text
            top_response = top_reaction.process(top_completion_text, hints)
            if not top_response.repeat:
                break
        new_documents = [phrase]
        if top_response.text:
            new_documents += [top_response.text]
        self.config.search_system_local.add_documents(new_documents)
        dialogue.append(phrase)
        if top_response.text:
            dialogue.append(top_response.text)
        session = Session(dialogue, hints)
        return top_response.text, session
