"""
Basic logging channels (stderr / file)
"""
from typing import Dict
import sys


class LoggerChannelInterface:
    """
    Basic "logger channel" interface
    """
    def __init__(self) -> None:
        self.channel_vars = {}

    def get_channel_vars(self) -> Dict[str, str]:
        """
        Get channel-specific variables
        """
        return self.channel_vars

    def set_channel_var(self, name: str, value: str) -> None:
        """
        Set channel-specific variable
        """
        self.channel_vars[name] = value

    def log(self, message: str) -> None:
        """
        Write log message
        """


class LoggerChannelFile(LoggerChannelInterface):
    """
    Write logs to file
    """
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def log(self, message: str) -> None:
        with open(self.name, "a", encoding="utf-8") as target:
            target.write(f"{message}\n")
            target.flush()


class LoggerChannelStderr(LoggerChannelInterface):
    """
    Write logs to stderr
    """
    def log(self, message: str) -> None:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
