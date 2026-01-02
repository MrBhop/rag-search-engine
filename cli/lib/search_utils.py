from dataclasses import dataclass
import json
import os

DEFAULT_SEARCH_LIMIT = 5

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MOVIES_PATH = os.path.join(DATA_PATH, "movies.json")
STOPWORDS_PATH = os.path.join(DATA_PATH, "stopwords.txt")

CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")
DOC_LENGTH_PATH = os.path.join(CACHE_PATH, "doc_lenghts.pkl")


@dataclass
class Movie:
    id: int
    title: str
    description: str

    @staticmethod
    def from_dict(d: dict[str, str]):
        return Movie(int(d["id"]), d["title"], d["description"])


def load_movies() -> list[Movie]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    output = map(Movie.from_dict, data["movies"])
    return list(output)


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
