import os
from lm_utils import LanguageModelInterface, PromptFiller
from robot import CompletionReaction, CompletionReactionInterface, Robot


class TestLanguageModel(LanguageModelInterface):
    def __init__(self) -> None:
        pass

    def complete(self, prompt: str) -> str:
        if prompt == "1":
            return "2"
        elif prompt == "2":
            return "3"
        else:
            return "UNKNOWN"


class TestCompletionReaction(CompletionReactionInterface):
    def __init__(self, trigger: str, stop: bool) -> None:
        super().__init__()
        self.trigger = trigger
        self.stop = stop

    def check(self, completion: str, variables: dict) -> CompletionReaction:
        if completion == self.trigger:
            variables["hints"].append(f"TRIGGER_{self.trigger}")
            return CompletionReaction(answer=f"TRIGGER_{self.trigger}", stop=self.stop)
        return CompletionReaction(answer=None, stop=False)
        

def test_robot_response_max_iters1():
    prompt_filler = PromptFiller(os.path.join(os.path.dirname(__file__), "robot_test_prompt.txt"))
    language_model = TestLanguageModel()
    reactions = [
        TestCompletionReaction("1", True),
        TestCompletionReaction("2", True),
        TestCompletionReaction("3", True),
        TestCompletionReaction("UNKNOWN", True),
    ]
    robot = Robot(prompt_filler, language_model, {}, reactions, 1)
    hints = []
    robot.response(
        hints,
        [],
        "1"
    )
    assert hints == ["TRIGGER_2"]
    robot.response(
        hints,
        [],
        "2"
    )
    assert hints == ["TRIGGER_2", "TRIGGER_3"]
    robot.response(
        hints,
        [],
        "3"
    )
    assert hints == ["TRIGGER_2", "TRIGGER_3", "TRIGGER_UNKNOWN"]
    pass
        

def test_robot_response_max_iters2_stop():
    prompt_filler = PromptFiller(os.path.join(os.path.dirname(__file__), "robot_test_prompt.txt"))
    language_model = TestLanguageModel()
    reactions = [
        TestCompletionReaction("1", True),
        TestCompletionReaction("2", True),
        TestCompletionReaction("3", True),
        TestCompletionReaction("UNKNOWN", True),
    ]
    robot = Robot(prompt_filler, language_model, {}, reactions, 2)
    hints = []
    robot.response(
        hints,
        [],
        "1"
    )
    assert hints == ["TRIGGER_2"]
        

def test_robot_response_max_iters2_no_stop():
    prompt_filler = PromptFiller(os.path.join(os.path.dirname(__file__), "robot_test_prompt.txt"))
    language_model = TestLanguageModel()
    reactions = [
        TestCompletionReaction("1", False),
        TestCompletionReaction("2", False),
        TestCompletionReaction("3", False),
        TestCompletionReaction("UNKNOWN", False),
    ]
    robot = Robot(prompt_filler, language_model, {}, reactions, 2)
    hints = []
    robot.response(
        hints,
        [],
        "1"
    )
    assert hints == ["TRIGGER_2", "TRIGGER_2"]
