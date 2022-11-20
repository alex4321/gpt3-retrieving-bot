from dataclasses import dataclass
from typing import List
from .nn_config import NNConfig
from .search_interface import SearchInterface
import numpy as np
from .utils import get_crossencoder


@dataclass
class SearchRankerItem:
    system: SearchInterface
    weight: float
    updateable: bool


class SearchRanker(SearchInterface):
    def __init__(self, ranker_model: str, nn_config: NNConfig, search_systems: List[SearchRankerItem], top_n: int):
        self.nn_config = nn_config
        self.model = get_crossencoder(ranker_model, device=nn_config.device)
        self.search_systems = search_systems
        self.top_n = top_n

    def update(self, document: str, amendment: str) -> None:
        for system in self.search_systems:
            if system.updateable:
                system.system.update(document, amendment)

    def search(self, query: str) -> List[str]:
        outputs = []
        output_weights = []
        for system in self.search_systems:
            system_results = system.system.search(query)
            outputs += system_results
            output_weights += [system.weight] * len(system_results)
        query_output_pairs = [
            (query, item)
            for item in outputs
        ]
        scores = self.model.predict(query_output_pairs, batch_size=self.nn_config.batch_size)
        scores_entailment = scores[:, 1]
        scores_entailment *= np.array(output_weights)
        indices = (-scores_entailment).argsort()[:self.top_n]
        return [outputs[idx] for idx in indices]
