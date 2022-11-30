from typing import List, Union, Callable
from collections import OrderedDict


HintPostprocessor = Callable[[str], str]


class HintsDict(OrderedDict):
    def __init__(self, keep_top_n: int, postprocess: Union[HintPostprocessor, None] = None, *args, **kwargs):
        super(HintsDict, self).__init__(*args, **kwargs)
        self.keep_top_n = keep_top_n
        if postprocess is None:
            postprocess = lambda text: text
        self.postprocess = postprocess

    def __setitem__(self, key: str, value: List[str]) -> None:
        super(HintsDict, self).__setitem__(key, value)
        if len(self.keys()) > self.keep_top_n:
            keys = self.keys()
            keys_to_remove = keys[:len(keys) - self.keep_top_n]
            for key in keys_to_remove:
                del self[key]

    def __getitem__(self, key: str) -> List[str]:
        return super().__getitem__(key)

    def __str__(self) -> str:
        result = ""
        for query, items in self.items():
            for response in items:
                result += "\n- " + f"{query} : self.postprocess({response})"
        result = result.strip()
        return result

    def __repr__(self) -> str:
        return str(self)
