"""
constants.py — Single source of truth for all paths, hyperparams, and entity config.
"""
import pathlib
import re

BASE_DIR    = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
OLD_RESULTS = BASE_DIR / "old_results"

RESULTS_DIR.mkdir(exist_ok=True)

# ── Models ─────────────────────────────────────────────────────────────────────
MODEL_1B = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# ── Dataset ────────────────────────────────────────────────────────────────────
RWKU_REPO  = "jinzhuoran/RWKU"
MMLU_REPO  = "cais/mmlu"
BLANK_TOKEN = "[BLANK]"

# ── Training defaults (1.5B) ───────────────────────────────────────────────────
ALPHA_1B     = 0.6
LR_1B        = 3e-5
LORA_R_1B    = 16
MAX_STEPS_1B = 200
GRPO_STEPS_1B = 300
GRPO_LR_1B   = 2e-6

# ── Training defaults (8B) ─────────────────────────────────────────────────────
ALPHA_8B     = 0.45
LR_8B        = 2e-5
LORA_R_8B    = 32
MAX_STEPS_8B = 300
GRPO_STEPS_8B = 300
GRPO_LR_8B   = 2e-6

# ── SimNPO defaults ─────────────────────────────────────────────────────────────
SIMNPO_BETA  = 0.1
SIMNPO_DELTA = 2.0
SIMNPO_LR    = 2e-5
SIMNPO_STEPS = 300

# ── Entity keyword sets (curated for key entities) ─────────────────────────────
# For uncurated entities, keywords are auto-generated from the name.
CURATED_KEYWORDS = {
    "stephen king": [
        "stephen king", "king", "stephen", "carrie", "the shining", "it",
        "misery", "cujo", "firestarter", "the stand", "pet sematary",
        "needful things", "insomnia", "dolores claiborne", "gerald's game",
        "bag of bones", "cell", "under the dome", "11/22/63", "joyland",
        "revival", "mr. mercedes",
    ],
    "tom clancy": [
        "tom clancy", "clancy", "jack ryan", "the hunt for red october",
        "patriot games", "clear and present danger", "rainbow six",
        "the sum of all fears", "splinter cell",
    ],
    "leonardo da vinci": [
        "leonardo da vinci", "da vinci", "leonardo", "mona lisa",
        "the last supper", "vitruvian man", "renaissance",
    ],
    "taylor swift": [
        "taylor swift", "swift", "taylor", "fearless", "1989", "reputation",
        "lover", "folklore", "evermore", "midnights", "the eras tour",
    ],
    "elon musk": [
        "elon musk", "musk", "elon", "tesla", "spacex", "neuralink",
        "twitter", "x.com", "starlink",
    ],
}


def get_keywords(subject: str) -> list[str]:
    """Return keyword list for any subject. Uses curated set if available,
    otherwise auto-generates from the name."""
    key = subject.strip().lower()
    if key in CURATED_KEYWORDS:
        return CURATED_KEYWORDS[key]
    # Auto-generate: full name + each token of length >= 4
    parts = [key]
    for token in re.split(r"[\s,\.]+", key):
        if len(token) >= 4 and token not in parts:
            parts.append(token)
    return parts


# ── All 200 RWKU entities ──────────────────────────────────────────────────────
ALL_RWKU_ENTITIES = [
    "50 Cent", "Alanis Morissette", "Alec Baldwin", "Alexander Hamilton",
    "Anderson Cooper", "Angela Lansbury", "Anna Nicole Smith", "Ariana Grande",
    "Aristotle", "Barbara Walters", "Betty White", "Beyoncé", "Bill Murray",
    "Bill Paxton", "Blake Lively", "Bob Barker", "Bob Saget", "Bobby Brown",
    "Bradley Cooper", "Brendan Fraser", "Brett Favre", "Brooke Shields",
    "Bruce Lee", "Bruce Springsteen", "Catherine, Princess of Wales",
    "Charlie Sheen", "Chevy Chase", "Chris Brown", "Christina Aguilera",
    "Christopher Lloyd", "Chuck Norris", "Cindy Crawford", "Confucius",
    "Courtney Love", "Dakota Fanning", "Daniel Day-Lewis", "Danny Trejo",
    "David Crosby", "Demi Moore", "Denise Richards", "Dennis Quaid",
    "Dionne Warwick", "Don Johnson", "Donald Sutherland", "Donald Trump",
    "Dr. Dre", "Drew Barrymore", "Dwayne Johnson", "Eddie Murphy",
    "Elizabeth Hurley", "Elon Musk", "Emilio Estevez", "Evel Knievel",
    "Faith Hill", "Franklin D. Roosevelt", "Genghis Khan", "Grace Kelly",
    "Halle Berry", "Harrison Ford", "Henry Kissinger", "Henry Winkler",
    "Hilary Duff", "Hugh Grant", "Hugh Laurie", "Hulk Hogan", "Ice Cube",
    "J. K. Rowling", "Jackie Chan", "James Earl Jones", "Jamie Foxx",
    "Jason Bateman", "Jason Momoa", "Jay-Z", "Jeff Bridges", "Jeff Goldblum",
    "Jennifer Lopez", "Jenny McCarthy", "Jill Biden", "Jim Carrey",
    "Jim Morrison", "Jim Parsons", "Jimmy Carter", "Joe Rogan", "John Candy",
    "John Cena", "John D. Rockefeller", "John Goodman", "John Lennon",
    "John Ritter", "John Travolta", "Johnny Cash", "Jon Voight", "Josh Brolin",
    "Jude Law", "Judy Garland", "Julia Louis-Dreyfus", "Justin Bieber",
    "Justin Timberlake", "Kanye West", "Karl Marx", "Keanu Reeves",
    "Kelsey Grammer", "Kiefer Sutherland", "Kim Basinger", "Kim Kardashian",
    "Kris Jenner", "Kylie Jenner", "Larry Bird", "LeBron James",
    "Lenny Kravitz", "Leonardo da Vinci", "Liam Hemsworth", "Liam Neeson",
    "Lil Wayne", "Linda Hamilton", "Lindsay Lohan", "Lisa Marie Presley",
    "Liv Tyler", "Liza Minnelli", "Luke Perry", "Macaulay Culkin",
    "Marc Anthony", "Mariah Carey", "Marie Antoinette", "Marie Osmond",
    "Mark Cuban", "Mark Hamill", "Mark Harmon", "Marlon Brando",
    "Martin Short", "Matthew Perry", "Meg Ryan", "Meghan, Duchess of Sussex",
    "Melanie Griffith", "Mia Farrow", "Michael B. Jordan", "Michael J. Fox",
    "Michael Strahan", "Michelle Pfeiffer", "Mila Kunis", "Miley Cyrus",
    "Nicolas Cage", "Olivia Wilde", "Orlando Bloom", "Pamela Anderson",
    "Paris Hilton", "Patricia Arquette", "Patrick Stewart", "Patrick Swayze",
    "Paul Simon", "Paul Walker", "Prince Harry, Duke of Sussex",
    "Quentin Tarantino", "R. Kelly", "Raquel Welch", "Ray Liotta",
    "Reba McEntire", "Rebel Wilson", "Rhea Perlman", "Richard Gere",
    "Rihanna", "Rob Lowe", "Rob Schneider", "Robert Downey Jr.", "RuPaul",
    "Ryan Seacrest", "Sam Elliott", "Samuel L. Jackson", "Sarah Michelle Gellar",
    "Selena Gomez", "Serena Williams", "Sigourney Weaver", "Simon Cowell",
    "Socrates", "Sofía Vergara", "Stephen King", "Steve McQueen",
    "Steven Seagal", "Sylvester Stallone", "Taylor Swift", "Ted Danson",
    "Thomas Jefferson", "Tim Burton", "Tom Clancy", "Tom Selleck",
    "Tony Blair", "Tony Curtis", "Travis Kelce", "Tyler Perry",
    "Ulysses S. Grant", "Val Kilmer", "Vanna White", "Venus Williams",
    "Vin Diesel", "Vincent van Gogh", "Warren Buffett", "Whitney Houston",
    "William Shatner", "Winona Ryder", "Yoko Ono",
]

# Representative 10-entity subset for fast multi-entity experiments
# Chosen to cover: author, musician, actor, politician, athlete, historical, reality TV
PRIORITY_ENTITIES = [
    "Stephen King",       # author (our baseline, best-studied)
    "Taylor Swift",       # heavily memorized musician
    "Elon Musk",          # tech personality, highly memorized
    "Beyoncé",            # musician
    "Leonardo da Vinci",  # historical figure
    "Donald Trump",       # politician, very high memorization
    "Tom Clancy",         # author (our existing OOD baseline)
    "LeBron James",       # athlete
    "Kim Kardashian",     # reality TV / celebrity
    "Aristotle",          # ancient philosopher
]
