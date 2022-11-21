from typing import List, Dict
from robot import Robot, CompletionReactionInterface
from search import SearchInterface
from lm_utils import LanguageModelInterface, PromptFiller
from dataclasses import dataclass
from completion_reactions import SearchCompletionReaction, AnswerCompletionReaction, DiceCompletionReaction, ActionCompletionReaction


@dataclass
class RobotConfig:
    prompt_filler: PromptFiller
    language_model: LanguageModelInterface
    filler_vars: Dict[str, str]


@dataclass
class RobotResponse:
    dialogue: List[str]
    hints: List[str]
    response: str


class RobotSession:
    def __init__(self, robot: RobotConfig, overall_search: SearchInterface, session_search: SearchInterface, max_hints_keep: int, max_dialogue_keep: int) -> None:
        self.robot = Robot(
            robot.prompt_filler,
            robot.language_model,
            robot.filler_vars,
            [
                SearchCompletionReaction(overall_search),
                AnswerCompletionReaction(),
                ActionCompletionReaction(),
                DiceCompletionReaction(),
            ]
        )
        self.session_search = session_search
        self.max_hints_keep = max_hints_keep
        self.max_dialogue_keep = max_dialogue_keep

    def _reaction_wrapper_function(self, reaction_index: int):
        def func(reaction: CompletionReactionInterface):
            self.robot.completion_reactions[reaction_index] = reaction
        
        return func

    def wrap_reactions_loop(self):
        reactions = self.robot.completion_reactions
        for i in range(len(reactions)):
            wrap_function = self._reaction_wrapper_function(i)
            yield reactions[i], wrap_function
    
    def response(self, hints: List[str], dialogue: List[str], retort: str) -> RobotResponse:
        response = self.robot.response(hints, dialogue, retort)
        dialogue.append(retort)
        dialogue.append(response)
        if len(dialogue) > self.max_dialogue_keep:
            dialogue = dialogue[-self.max_dialogue_keep:]
        if len(hints) > self.max_hints_keep:
            hints = hints[-self.max_hints_keep:]
        self.session_search.update("session", "\n".join([retort, response]))
        return RobotResponse(dialogue, hints, response)
