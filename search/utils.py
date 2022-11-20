from sentence_transformers import CrossEncoder, SentenceTransformer


_singletone_crossencoders = {}
_singletone_sentence_transformers = {}


def get_crossencoder(model: str, *args, **kwargs) -> CrossEncoder:
    if model not in _singletone_crossencoders:
        _singletone_crossencoders[model] = CrossEncoder(model, *args, **kwargs)
    return _singletone_crossencoders[model]


def get_sentence_transformer(model: str, *args, **kwargs) -> SentenceTransformer:
    if model not in _singletone_sentence_transformers:
        _singletone_sentence_transformers[model] = SentenceTransformer(model, *args, **kwargs)
    return _singletone_sentence_transformers[model]
