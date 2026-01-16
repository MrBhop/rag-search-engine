from dataclasses import dataclass
import json
import os
from typing import Any

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_ALPHA = 0.5

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_PEVIEW_LENGTH = 100
SCORE_PRECISION = 3

DEFAULT_FIXED_CHUNK_SIZE = 200
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 0

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
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_PATH, "chunk_metadata.json")


@dataclass
class Document:
    id: int
    title: str
    description: str

    @staticmethod
    def from_dict(d: dict[str, str]):
        return Document(int(d["id"]), d["title"], d["description"])


def load_movies() -> list[Document]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
    output = map(Document.from_dict, data["movies"])
    return list(output)


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


@dataclass
class FormattedSearchResult:
    id: str
    title: str
    document: str
    score: float
    metadata: dict[str, Any]

    @staticmethod
    def from_document(score: float, doc: Document, **metadata: Any):
        return FormattedSearchResult(
            str(doc.id), doc.title, doc.description, score, metadata if metadata else {}
        )
