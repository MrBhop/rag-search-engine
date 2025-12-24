import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

MOVIES_PATH = os.path.join(DATA_PATH, "movies.json")
STOPWORDS_PATH = os.path.join(DATA_PATH, "stopwords.txt")


def load_movies() -> list[dict[str, str]]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
