from typing import Dict
import sys


class LoggerChannelInterface:
    def __init__(self) -> None:
        self.channel_vars = {}

    def get_channel_vars(self) -> Dict[str, str]:
        return self.channel_vars

    def set_channel_var(self, name: str, value: str) -> None:
        self.channel_vars[name] = value

    def log(self, message: str) -> None:
        pass


class LoggerChannelFile(LoggerChannelInterface):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def log(self, message: str) -> None:
        with open(self.name, "a", encoding="utf-8") as target:
            target.write(f"{message}\n")
            target.flush()


class LoggerChannelStderr(LoggerChannelInterface):
    def log(self, message: str) -> None:
        sys.stderr.write(f"{message}\n")
        sys.stderr.flush()
