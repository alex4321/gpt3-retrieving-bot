import datetime
import json
from .channels import LoggerChannelInterface


class BaseLogger:
    def __init__(self, name: str, channel: LoggerChannelInterface) -> None:
        self.name = name
        self.channel = channel

    def log(self, message: str, params: dict) -> None:
        now = str(datetime.datetime.now())
        params_all = dict(self.channel.get_channel_vars(), **params)
        message = f"[{self.name}] {now}: {message} {json.dumps(params_all, ensure_ascii=False)}"
        self.channel.log(message)
