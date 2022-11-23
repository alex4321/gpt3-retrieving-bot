from chatbots.lm_utils import PromptFiller


def test_prompt_filler_fill():
    filler = PromptFiller()
    text = filler.fill({
        "CHARACTER_NAME": "Isaac Clarke"
    })
    assert "You're Isaac Clarke." in text
