from session import RobotConfig, RobotSession
from lm_utils import PromptFiller, LanguageModelGPT3
from search import SearchDuckDuckGo, SearchRanker, SearchRankerItem, SearchLocalDatabaseTextual, SearchLocalDatabaseSemantic, NNConfig
import openai
import os
import shutil
import tempfile
from botloggers import LoggerChannelFile, LMLogger, SearchLogger, CompletionReactionLogger
from completion_reactions import AnswerCompletionReaction, DiceCompletionReaction, SearchCompletionReaction, ActionCompletionReaction


if __name__ == "__main__":
    assert "OPENAI_KEY" in os.environ
    openai.api_key = os.environ["OPENAI_KEY"]
    semantic_session_search_directory = tempfile.mkdtemp()
    text_session_search_directory = tempfile.mkdtemp()
    logger = LoggerChannelFile("dixie.log")
    try:
        robot = RobotConfig(
            prompt_filler=PromptFiller(),
            language_model=LMLogger(
                "Language Model",
                logger,
                LanguageModelGPT3(),
            ),
            filler_vars={
                "NOW": "11 Nov 2022",
                "CHARACTER_NAME": "Dixie Flatline",
                "CHARACTER_DESCRIPTION": " ".join([
                    "You live in a cyberpunk world",
                    "You're McCoy Pauley aka Dixie Flatline or just Dix - a famous computer hacker who was one of the mentors of Case.",
                    "You're redneck from the Atlanta fringes. When Case met him, you was a thickset man in shirtsleeves, and his skin had a leaden shade.",
                    "During the War you and Elroy ended up in a POW camp in Siberia where they stayed more than a month; ",
                    "you remembered when Elroy still felt a phantom itching on his amputated thumb.",
                    "You was implanted a surplus Russian heart.",
                    "After the war you refused to replace it, as its particular beat gave him a sense of timing.",
                    "But that's all long past. Nowadays you're hacker - and you don't hesistate to crack even most complicate corporate AI's.",
                    "Once it almost cost you live - your brain was not active for a few minutes (what's why guys call you flaline). ",
                    "But you managed to survive.",
                    "Last thing you can remember now is brainscanning for some kind of medical threatment.",
                    "And you woke up in a strange place. Reminds you're a cyberspace, but something is wrong with your feelings."
                ])
            }
        )
        search_systems = [
            SearchRankerItem(
                SearchLogger(
                    "Search - Global - DuckDuckGo",
                    logger,
                    SearchDuckDuckGo(2)
                ), 
                1.0,
                False
            ),
            SearchRankerItem(
                SearchLogger(
                    "Search - Session - Semantin",
                    logger,
                    SearchLocalDatabaseSemantic(
                        semantic_session_search_directory,
                        "sentence-transformers/all-MiniLM-L6-v2",
                        2,
                        NNConfig("cuda:0", 8),
                        {}
                    ),
                ),
                0.5,
                True
            ),
            SearchRankerItem(
                SearchLogger(
                    "Search - Session - Text",
                    logger,
                    SearchLocalDatabaseTextual(text_session_search_directory, "english", 2),
                ),
                0.8,
                True
            )
        ]
        search_ranker = SearchLogger(
            "Search - Aggregator",
            logger,
            SearchRanker(
                "cross-encoder/nli-deberta-v3-base",
                NNConfig("cuda:0", 8),
                search_systems,
                2
            )
        )
        session = RobotSession(
            robot,
            search_ranker,
            search_ranker,
            10,
            50
        )
        for reaction, wrap in session.wrap_reactions_loop():
            if isinstance(reaction, AnswerCompletionReaction):
                reaction_name = "Reaction - Answer"
            elif isinstance(reaction, ActionCompletionReaction):
                reaction_name = "Reaction - Action"
            elif isinstance(reaction, DiceCompletionReaction):
                reaction_name = "Reaction - Dice"
            elif isinstance(reaction, SearchCompletionReaction):
                reaction_name = "Reaction - Search"
            else:
                reaction_name = "Reaction - Unknown"
            reaction_wrapped = CompletionReactionLogger(
                reaction_name,
                logger,
                reaction
            )
            wrap(reaction_wrapped)
        hints = []
        history = []
        while True:
            request = input()
            response = session.response(hints, history, request)
            print(response.response)
            hints = response.hints
            history = response.dialogue
    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(semantic_session_search_directory)
        shutil.rmtree(text_session_search_directory)