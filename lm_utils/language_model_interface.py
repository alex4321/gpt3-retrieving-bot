class LanguageModelInterface:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError()