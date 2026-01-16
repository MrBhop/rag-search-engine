from collections import defaultdict
import os

from lib.search_utils import DEFAULT_PEVIEW_LENGTH, DEFAULT_SEARCH_LIMIT, Document, FormattedSearchResult, load_movies

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
        bm25_results = self._bm25_search(query, 500 * limit)
        bm25_scores: list[float] = list(map(lambda item: item.score, bm25_results))
        bm25_normalized = normalize_scores(bm25_scores)

        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)
        semantic_scores: list[float] = list(map(lambda item: item.score, semantic_results))
        semantic_normalized = normalize_scores(semantic_scores)


        document_mappings = defaultdict(dict)
        for index, item in enumerate(bm25_results, 0):
            item_id = str(item.id)
            document_mappings[item_id]["title"] = item.title
            document_mappings[item_id]["description"] = item.document
            document_mappings[item_id]["bm25_score"] = bm25_normalized[index]

        for index, item in enumerate(semantic_results, 0):
            item_id = str(item.id)
            document_mappings[item_id]["title"] = item.title
            document_mappings[item_id]["description"] = item.document
            document_mappings[item_id]["semantic_score"] = semantic_normalized[index]

        for key in document_mappings.keys():
            doc = document_mappings[key]
            bm25_score = doc.get("bm25_score", 0)
            semantic_score = doc.get("semantic_score", 0)

            document_mappings[key]["hybrid_score"] = hybrid_score(bm25_score, semantic_score, alpha)

        sorted_results = sorted(document_mappings, key=lambda item: document_mappings[item]["hybrid_score"], reverse=True)
        def generate_formatted_search_result(key):
            doc = document_mappings[key]

            return FormattedSearchResult(
                "",
                doc["title"],
                doc["description"],
                doc["hybrid_score"],
                {"bm25_score":doc["bm25_score"],
                 "semantic_score":doc["semantic_score"],
                 },
            )
        results = list(map(generate_formatted_search_result, sorted_results[:limit]))
        return results

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


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search_command(
    query: str, alpha: float = 0.5, limit: int = DEFAULT_SEARCH_LIMIT
):
    movies = load_movies()
    srch = HybridSearch(movies)
    results = srch.weighted_search(query, alpha, limit)

    for index, item in enumerate(results, 1):
        print(f"{index}. {item.title}")
        print(f"\tHybrid Score: {item.score:.3f}")
        print(f"\tBM25: {item.metadata["bm25_score"]:.3f}, Semantic: {item.metadata["semantic_score"]:.3f}")
        print(f"\t{item.document[:DEFAULT_PEVIEW_LENGTH]}")
