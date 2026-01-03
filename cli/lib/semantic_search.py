import os
import numpy as np

from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer

from lib.search_utils import (
    CHUNK_SIZE_DEFAULT,
    DEFAULT_SEARCH_LIMIT,
    MOVIE_EMBEDDINGS_PATH,
    Document,
    FormattedSearchResult,
    load_movies,
)


class SemanticSerach:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[Document] = []
        self.document_map: dict[int, Document] = {}

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode(text)

    def build_embeddings(self, documents: list[Document]):
        self.documents = documents
        self.document_map = {}

        docs_as_strings: list[str] = []
        for doc in self.documents:
            self.document_map[doc.id] = doc
            docs_as_strings.append(f"{doc.title}: {doc.description}")
        self.embeddings = self.model.encode(docs_as_strings, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Document]):
        self.documents = documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc.id] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query.strip())

        scores: list[tuple[float, Document]] = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            scores.append((score, self.documents[i]))

        scores.sort(key=lambda item: item[0], reverse=True)

        return map(
            lambda item: FormattedSearchResult.from_document(item[0], item[1]),
            scores[:limit],
        )


def cosine_similarity(vec1: ArrayLike, vec2: ArrayLike) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_model():
    search = SemanticSerach()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str):
    search = SemanticSerach()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search = SemanticSerach()
    movies = load_movies()
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    search = SemanticSerach()
    embedding = search.generate_embedding(query.strip())
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search = SemanticSerach()
    movies = load_movies()
    _ = search.load_or_create_embeddings(movies)

    results = list(search.search(query, limit))

    print(f"Query: {query}")
    print(f"Top {len(results)} reults:")
    print()
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.title} (score: {doc.score:.4f})")
        print(f"\t{doc.document[:100]}...")
        print()


def fixed_size_chunking(text: str, chunk_size: int = CHUNK_SIZE_DEFAULT) -> list[str]:
    words = text.split()
    output: list[str] = []

    while len(words) > chunk_size:
        output.append(" ".join(words[:chunk_size]))
        words = words[chunk_size:]
    output.append(" ".join(words))

    return output


def chunk_command(text: str, chunk_size: int = CHUNK_SIZE_DEFAULT) -> None:
    chunks = fixed_size_chunking(text, chunk_size)

    print(f"Chunking {len(text)} characters")
    for i, line in enumerate(chunks, 1):
        print(f"{i}. {line}")
