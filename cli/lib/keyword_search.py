from collections import defaultdict
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
    Movie,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}

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
        return self

    def get_documents(self, term: str) -> list[int]:
        result = self.index.get(term.lower(), set())
        return sorted(result)

    def __add_documents(self, doc_id: int, text: str):
        tokens = tokenize_text(preprocess_text(text))

        for token in set(tokens):
            self.index[token].add(doc_id)


def build_command():
    idx = InvertedIndex().build().save()
    docs = idx.get_documents("merida")
    first_movie = idx.docmap[docs[0]]
    print(f"First document for token 'merida' = {first_movie.id}: {first_movie.title}")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> Iterable[Movie]:
    movies = load_movies()
    results: list[Movie] = []

    query_tokens = tokenize_text(query)

    for movie in movies:
        title_tokens = tokenize_text(movie.title)
        if title_contains_query(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def title_contains_query(query: Iterable[str], title: Iterable[str]) -> bool:
    for token in query:
        for title_token in title:
            if token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    lowered_text = text.lower()
    return lowered_text.translate(str.maketrans("", "", string.punctuation))


def tokenize_text(text: str) -> Iterable[str]:
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
