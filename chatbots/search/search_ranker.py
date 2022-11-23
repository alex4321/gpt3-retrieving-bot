"""
Cross-encoder based search ranking to rank results from a few systems
"""
from dataclasses import dataclass
from typing import List
import numpy as np
from torch.nn.functional import softmax
import torch
from .nn_config import NNConfig
from .search_interface import SearchInterface
from .utils import get_crossencoder, is_local_request


@dataclass
class SearchRankerItem:
    """
    Individual search system config
    """
    system: SearchInterface
    weight: float
    updateable: bool


class SearchRanker(SearchInterface):
    """
    Cross-encoder based search ranking to rank results from a few systems
    """
    def __init__(self, ranker_model: str, ranket_model_neutral_label: int,
                 nn_config: NNConfig, search_systems: List[SearchRankerItem],
                 top_n: int) -> None:
        self.nn_config = nn_config
        self.model = get_crossencoder(ranker_model, device=nn_config.device)
        self.neutral_label = ranket_model_neutral_label
        self.search_systems = search_systems
        self.top_n = top_n

    def update(self, document: str, amendment: str) -> None:
        for system in self.search_systems:
            if system.updateable:
                system.system.update(document, amendment)

    def search(self, query: str) -> List[str]:
        _, query = is_local_request(query)
        outputs = []
        output_weights = []
        for system in self.search_systems:
            system_results = system.system.search(query)
            outputs += system_results
            output_weights += [system.weight] * len(system_results)
        query_output_pairs = [
            (item, query)
            for item in outputs
        ]
        with torch.no_grad():
            scores = self.model.predict(query_output_pairs,
                                        batch_size=self.nn_config.batch_size)
            scores = torch.FloatTensor(scores)
            scores = softmax(scores, dim=-1)
            scores = scores.detach().cpu().numpy()
        scores_neutral = scores[:, self.neutral_label]
        scores_neutral *= np.array(output_weights)
        indices = scores_neutral.argsort()[:self.top_n]
        return [outputs[idx] for idx in indices]
