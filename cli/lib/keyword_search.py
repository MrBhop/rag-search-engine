from collections import Counter, defaultdict
import math
import os
import pickle
import string
from collections.abc import Iterable

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_PATH,
    DEFAULT_SEARCH_LIMIT,
    DOC_LENGTH_PATH,
    DOCMAP_PATH,
    INDEX_PATH,
    TERM_FREQUENCIES_PATH,
    FormattedSearchResult,
    Document,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Document] = {}
        self.term_frequencies: dict[int, Counter[str]] = defaultdict(Counter)
        self.doc_lenghts: dict[int, int] = {}
        self.index_path: str = INDEX_PATH

    def build(self):
        documents = load_movies()

        for doc in documents:
            self.docmap[doc.id] = doc
            self.__add_documents(doc.id, f"{doc.title} {doc.description}")
        return self

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, mode="wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, mode="wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_FREQUENCIES_PATH, mode="wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTH_PATH, mode="wb") as f:
            pickle.dump(self.doc_lenghts, f)
        return self

    def load(self):
        with open(self.index_path, mode="rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, mode="rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FREQUENCIES_PATH, mode="rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(DOC_LENGTH_PATH, mode="rb") as f:
            self.doc_lenghts = pickle.load(f)
        return self

    @staticmethod
    def to_single_token(term: str) -> str:
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single token")
        return tokens[0]

    def get_documents(self, term: str) -> list[int]:
        result = self.index.get(term.lower(), set())
        return sorted(result)

    def get_tf(self, doc_id: int, term: str):
        token = InvertedIndex.to_single_token(term)
        return self.term_frequencies.get(doc_id, Counter()).get(token, 0)

    def get_idf(self, term: str):
        token = InvertedIndex.to_single_token(term)
        total_doc_count = len(self.docmap)
        term_match_count = len(self.get_documents(token))
        return math.log((total_doc_count + 1) / (term_match_count + 1))

    def get_tf_idf(self, doc_id: int, term: str):
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        token = InvertedIndex.to_single_token(term)
        total_doc_count = len(self.docmap)
        term_match_count = len(self.get_documents(token))
        return math.log(
            (total_doc_count - term_match_count + 0.5) / (term_match_count + 0.5) + 1
        )

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        avg_length = self.__get_avg_doc_length()
        if avg_length > 0:
            length_norm = 1 - b + b * (self.doc_lenghts[doc_id] / avg_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        query_tokens = tokenize_text(query)

        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            total_score = 0
            for token in query_tokens:
                total_score += self.bm25(doc_id, token)
            scores[doc_id] = total_score

        sorted_result_ids = sorted(
            scores.keys(), key=lambda id: scores[id], reverse=True
        )
        return list(map(
            lambda doc_id: FormattedSearchResult.from_document(
                scores[doc_id], self.docmap[doc_id]
            ),
            sorted_result_ids[:limit],
        ))

    def __add_documents(self, doc_id: int, text: str):
        tokens = tokenize_text(text)

        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lenghts[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lenghts) == 0:
            return 0
        total = 0
        for length in self.doc_lenghts.values():
            total += length
        return total / len(self.doc_lenghts)


def build_command():
    # Build and save the InvertedIndex.
    _ = InvertedIndex().build().save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> Iterable[Document]:
    idx = InvertedIndex().load()
    query_tokens = tokenize_text(query)

    result_ids: set[int] = set()
    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)
        for id in matching_doc_ids:
            result_ids.add(id)
            if len(result_ids) >= limit:
                break
        if len(result_ids) >= limit:
            break

    return map(lambda id: idx.docmap[id], sorted(result_ids))


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex().load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex().load()
    return idx.get_idf(term)


def tf_idf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex().load()
    return idx.get_tf_idf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex().load()
    return idx.get_bm25_idf(term)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    idx = InvertedIndex().load()
    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25serach_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex().load()
    return idx.bm25_search(query, limit)


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
    tokens = preprocessed.split()
    without_empty_tokens_and_stopwords = list(
        filter(lambda item: item != "" and item not in stop_words, tokens)
    )
    stemmer = PorterStemmer()
    stemmed_words = list(
        map(lambda item: str(stemmer.stem(item)), without_empty_tokens_and_stopwords)
    )
    return stemmed_words
