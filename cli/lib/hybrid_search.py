import os

from lib.search_utils import Document

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[Document]):
        self.documents: list[Document] = documents
        self.semantic_search: ChunkedSemanticSearch = ChunkedSemanticSearch()
        _ = self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx: InvertedIndex = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            _ = self.idx.build().save()

    def _bm25_search(self, query: str, limit: int):
        return self.idx.load().bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(values: list[float]):
    min_value = min(values)
    max_value = max(values)

    if min_value == max_value:
        normalized_values = map(lambda _: 1, values)
    else:
        normalized_values = map(
            lambda score: (score - min_value) / (max_value - min_value), values
        )

    return list(normalized_values)


def normalize_command(values: list[float]):
    normalized = normalize_scores(values)
    for value in normalized:
        print(f"* {value:.4f}")
