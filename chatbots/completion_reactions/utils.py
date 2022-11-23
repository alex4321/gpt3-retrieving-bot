"""
LM postprocessor utils
"""
import re
from typing import List
from dataclasses import dataclass


COMMAND_SEARCH = "Search"
COMMAND_ANSWER = "Answer"
COMMAND_ACTION = "Action"
COMMAND_DICE = "Dice"
COMMANDS = [
    COMMAND_SEARCH,
    COMMAND_ACTION,
    COMMAND_ANSWER,
    COMMAND_DICE,
]
COMMANDS_WITH_SUFFIX = {
    cmd + ":": cmd
    for cmd in COMMANDS
}


@dataclass
class Command:
    """
    Command extracted from LM completion
    """
    type: str
    args: str


def split_completion(completion: str) -> List[Command]:
    """
    Split LM completion to a sequence of commands
    """
    pattern = "(" + \
        "|".join([
            command + r"\:"
            for command in COMMANDS
        ]) + \
        ")"
    parts = re.split(pattern, completion)
    result = []
    buffer = []
    command = None
    for item in parts:
        if item in COMMANDS_WITH_SUFFIX:
            result.append((command, buffer))
            command = COMMANDS_WITH_SUFFIX[item]
            buffer = []
        else:
            buffer.append(item)
    if buffer:
        result.append((command, buffer))
    result = [
        (command, "".join(items).strip())
        for command, items in result
    ]
    result = [
        (command, items)
        for command, items in result
        if (command is not None) and (items != "")
    ]
    return [
        Command(command, items)
        for command, items in result
    ]
