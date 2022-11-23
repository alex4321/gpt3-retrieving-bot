"""
Neural network-based search method run config
"""
from dataclasses import dataclass


@dataclass
class NNConfig:
    """
    Neural network-based search method run config
    """
    device: str
    batch_size: int
