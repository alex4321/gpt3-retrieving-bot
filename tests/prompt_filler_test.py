from lm_utils import PromptFiller


def test_prompt_filler_fill():
    filler = PromptFiller()
    text = filler.fill({
        "CHARACTER_NAME": "Isaac Clarke"
    })
    assert text.startswith("You're Isaac Clarke.")
