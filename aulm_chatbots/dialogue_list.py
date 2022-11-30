class DialogueList(list):
    def __init__(self, keep_top_n: int, *args, **kwargs):
        super(DialogueList, self).__init__(*args, **kwargs)
        self.keep_top_n = keep_top_n
    
    def append(self, item: str) -> None:
        super(DialogueList, self).append(item)
        while len(self) > self.keep_top_n:
            del self[0]

    def __str__(self) -> str:
        result = ""
        for item in self:
            result += f"\n- {item}"
        return result.strip()
