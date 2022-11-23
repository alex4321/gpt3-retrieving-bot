"""
Search system utils
"""
from typing import Tuple
from sentence_transformers import CrossEncoder, SentenceTransformer


STORY_SEARCH_MARKER = "STORY"


_singletone_crossencoders = {}
_singletone_sentence_transformers = {}


def get_crossencoder(model: str, *args, **kwargs) -> CrossEncoder:
    """
    Get crossencoder instance (one per process)
    """
    if model not in _singletone_crossencoders:
        _singletone_crossencoders[model] = CrossEncoder(model, *args, **kwargs)
    return _singletone_crossencoders[model]


def get_sentence_transformer(model: str, *args, **kwargs) -> SentenceTransformer:
    """
    Get sentence encoder instance (one per process)
    """
    if model not in _singletone_sentence_transformers:
        _singletone_sentence_transformers[model] = SentenceTransformer(model, *args, **kwargs)
    return _singletone_sentence_transformers[model]


def is_local_request(query: str) -> Tuple[bool, str]:
    """
    Check if LM requests search through conversation history or globally
    """
    return STORY_SEARCH_MARKER in query, query.replace(STORY_SEARCH_MARKER, "")
