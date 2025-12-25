from collections import Counter, defaultdict
import os
import pickle
import string
from collections.abc import Iterable

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_PATH,
    DEFAULT_SEARCH_LIMIT,
    DOCMAP_PATH,
    INDEX_PATH,
    TERM_FREQUENCIES_PATH,
    Movie,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: dict[int, Counter[str]] = defaultdict(Counter)

    def build(self):
        movies = load_movies()

        for m in movies:
            self.docmap[m.id] = m
            self.__add_documents(m.id, f"{m.title} {m.description}")
        return self

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(INDEX_PATH, mode="wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, mode="wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_FREQUENCIES_PATH, mode="wb") as f:
            pickle.dump(self.term_frequencies, f)
        return self

    def load(self):
        with open(INDEX_PATH, mode="rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, mode="rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FREQUENCIES_PATH, mode="rb") as f:
            self.term_frequencies = pickle.load(f)
        return self

    def get_documents(self, term: str) -> list[int]:
        result = self.index.get(term.lower(), set())
        return sorted(result)

    def get_tf(self, doc_id: int, term: str):
        token = tokenize_text(term)
        if len(token) > 1:
            raise ValueError("term must be a single token")
        return self.term_frequencies.get(doc_id, Counter()).get(token[0], 0)

    def __add_documents(self, doc_id: int, text: str):
        tokens = tokenize_text(preprocess_text(text))

        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)


def build_command():
    # Build and save the InvertedIndex.
    _ = InvertedIndex().build().save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> Iterable[Movie]:
    idx = InvertedIndex().load()
    query_tokens = tokenize_text(query)

    movie_ids: set[int] = set()
    for token in query_tokens:
        doc_ids = idx.get_documents(token)
        for id in doc_ids:
            movie_ids.add(id)
            if len(movie_ids) >= limit:
                break
        if len(movie_ids) >= limit:
            break

    return map(lambda id: idx.docmap[id], sorted(movie_ids))


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex().load()
    return idx.get_tf(doc_id, term)


def title_contains_query(query: Iterable[str], title: Iterable[str]) -> bool:
    for token in query:
        for title_token in title:
            if token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    lowered_text = text.lower()
    return lowered_text.translate(str.maketrans("", "", string.punctuation))


def tokenize_text(text: str) -> list[str]:
    stop_words = load_stopwords()

    preprocessed = preprocess_text(text)
    tokens = preprocessed.split(" ")
    without_empty_tokens_and_stopwords = list(
        filter(lambda item: item != "" and item not in stop_words, tokens)
    )
    stemmer = PorterStemmer()
    stemmed_words = list(
        map(lambda item: str(stemmer.stem(item)), without_empty_tokens_and_stopwords)
    )
    return stemmed_words
