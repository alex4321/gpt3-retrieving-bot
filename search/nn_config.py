from dataclasses import dataclass


@dataclass
class NNConfig:
    device: str
    batch_size: int