import shutil
import tempfile
from sentence_transformers import CrossEncoder, SentenceTransformer
from aulm_chatbots.search import SearchReranker, SearchRerankerItem, SearchLocalDatabase
import os


__DIXIE_FLATLINE_ARTICLE__ = """
Dixie Flatline : McCoy Pauley aka Dixie Flatline or just Dix is a famous computer hacker who was one of the mentors of Case. 
A redneck from the Atlanta fringes. When Case met him, he was a thickset man in shirtsleeves, and his skin had a leaden shade.
As a construct, Dixie was not sure if he was sentient, as he was a bunch of ROM and would not be able to write a poem, but it feels like he is sentient. He said that his existence didn't feel like anything, and compared it like the phantom sensation of an amputated member. Whenever Case communicated with him in the cyberspace, Case felt an odd feeling like "someone reading over your shoulder". Whenever he joked, he released an unpleasant laughter sensation that Case felt in his spine.
"""

__JOHNNY_SILVERHAND_ARTICLE__ = """
Johny Silverhand : They say you can become a legendary rockerboy without all the sex and drugs, the manic depression, run-ins with the law and one toxic relationship after another. But Johnny Silverhand's old school.
Frontman for Samurai, charismatic visionary, rebel with a cause, sworn enemy of corporations (but especially Arasaka) and the mind behind the cult singles "Chippin' In" and "Never Fade Away" - currently residing in V's brain as a digitized tenant.
Silverhand met his demise during the attack on Arasaka Tower after getting shot by Adam Smasher and subsequently flatlined by Soulkiller. But some rockerboys never really die. Point in case - Silverhand's personality construct was kept in Arasaka's labs for decades before it landed on a prototype biochip call the Relic, which - following a series of unexpected events - ended up in V's brain. If you think spending eternity in a cyberspace prison is worse than sharing your headspace with a complete stranger, you'd be dead wrong. For an egomaniac and narcissist like Johnny, it's a living hell.[22]
"""


__RELIC_ARTICLE__ = """
Relic : The Relic is a series of Arasaka biochips allowing the storage and manifested reading of digitized human psyches known as engrams.
At least two separate versions of the biochip prototype were developed by 2077, according to the summarized internal report assembled by the former director of the Relic project, Anders Hellman, upon turning coat for the benefit of Arasaka's rival corporation, Kang Tao. 
The first version was intended for the commercial market and advertised as a means for wealthy elites to store their psyche in form of an engram capable of basic communication with their loved ones. The form itself would have consciousness but lack true self-awareness. In spite of its limitations, Relic 1.0 has managed to enter the market and the lives of Night City residents, giving hope by providing a notion of immortality truly approaching human reach. Arasaka logs on the other hand reveal that the Relic's initial purpose was of a more capitalist nature - using Soulkiller on celebrities, important cultural icons and artists in order to create engrams for commercialisation purposes.
The second version of the biochip was a top secret project, personally commissioned and supervised by Saburo Arasaka. It was intended for internal use within the corporation only, never to be sold. Unlike the original biochip, which was only used to communicate with pre-saved engrams with artificially integrated limitations, Relic 2.0 contains a system which was meant to install and activate the engram in a new organic body. The core idea of the project was to implant a digitized psyche into a new host, although only after the body had all neural and cardiac functions terminated, at which point it would automatically expand into the host's brain using nanotechnology. In short, a person who copied their mind onto Relic 2.0 and then died could be restored to life in a new body using the chip, effectively granting them immortality. Internal testing of Relic 2.0 showed promising results, but Arasaka scientists had difficulty preventing the personality construct from becoming emotionally unstable after re-implantation, and the biochip eventually failed in every trial. The Arasaka Corporation also confirmed that Relic 2.0 would not activate if implanted in living individuals who were on the verge of death. The project had not progressed past the trial phase until an unplanned undertaking of the process by a living individual. Examination of the relic's advancement proved that, despite keeping the subject alive, it was continuing its functional expansion and taking over the motor and psychological functions of the host.
"""


__CORTEX_CHIP_ARTICLE__ = """
Cortex Chip : A Cortex Chip is a type of computer chip designed to store artificial intelligences for robots. Additionally, Cortex Chips appear to be compatible with neurographs.
A Cortex Chip has a universal connector, allowing it to be inserted into machinery, such as robots or an Omnitool.
Cortex Chips seem to come with in-built audio recording and processing software in order to process sounds and voices and interact accordingly. When combined with an Occu-Torch, a Cortex Chip can provide brain scans with visual capabilities. However, Cortex Chips are not designed to handle brain scans, and if loaded with a scan they may overload if the scanned person experiences extreme emotions.
Over the course of his journey, Simon Jarrett encounters several robots, known as Mockingbirds, containing brain scans of PATHOS-II employees, presumably stored on Cortex Chips. Though some are able to communicate with Simon, others are hostile, attacking him on sight.
Simon obtains two Cortex Chips over the course of the game. One contains a brain scan of Catherine Chun, which he connects to his Omnitool. In Omicron Simon uses another to build a second body for himself using a Haimatsu Power Suit to be able to survive the extreme water pressure of the Omega Sector.
"""


def test_search_reranker():
    temp_dir = tempfile.mkdtemp()
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2", device="cpu")
        search_characters = SearchLocalDatabase(
            SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            ),
            os.path.join(temp_dir, "characters"),
            2,
            16
        )
        search_items = SearchLocalDatabase(
            SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            ),
            os.path.join(temp_dir, "items"),
            2,
            16
        )
        search = SearchReranker(
            reranker,
            [
                SearchRerankerItem(search_characters, True),
                SearchRerankerItem(search_items, updateable=True),
            ],
            8,
            2,
            0.0
        )
        search_characters.add_documents([__DIXIE_FLATLINE_ARTICLE__, __JOHNNY_SILVERHAND_ARTICLE__])
        search_items.add_documents([__RELIC_ARTICLE__, __CORTEX_CHIP_ARTICLE__])
        construct_search = search.search("Who are constructs?")
        rockerboy_search = search.search("Who was a rocker")
        biochip_search = search.search("What is Bio-Chip")
        cortex_chip_search = search.search("What is Cortex Chip")
        phrase_search = search.search("Sex, drugs and rock'n'roll")
        construct_search_documents = {item.split(" : ")[0] for item in construct_search}
        rockerboy_search_documents = {item.split(" : ")[0] for item in rockerboy_search}
        biochip_search_documents = {item.split(" : ")[0] for item in biochip_search}
        cortex_chip_search_documents = {item.split(" : ")[0] for item in cortex_chip_search}
        phrase_search_documents = {item.split(" : ")[0] for item in phrase_search}
        assert construct_search_documents == {"Johny Silverhand", "Dixie Flatline"}
        assert rockerboy_search_documents == {"Johny Silverhand", "Dixie Flatline"}
        assert biochip_search_documents == {"Cortex Chip", "Johny Silverhand"}
        assert cortex_chip_search_documents == {"Cortex Chip", "Johny Silverhand"}
        assert phrase_search_documents == {"Johny Silverhand", "Dixie Flatline"}
    finally:
        shutil.rmtree(temp_dir)
