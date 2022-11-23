"""
Base language model interface
"""

class LanguageModelInterface:
    """
    Base language model interface
    """
    def complete(self, prompt: str) -> str:
        """
        Generate completion
        """
        raise NotImplementedError()
