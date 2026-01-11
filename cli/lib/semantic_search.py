import json
import os
import re

import numpy as np
from lib.search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_FIXED_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PEVIEW_LENGTH,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    MOVIE_EMBEDDINGS_PATH,
    SCORE_PRECISION,
    Document,
    FormattedSearchResult,
    load_movies,
)
from numpy.typing import ArrayLike
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name=DEFAULT_MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[dict[str, int]] = []

    def build_chunk_embeddings(self, documents: list[Document]):
        self.documents: list[Document] = documents
        self.document_map: dict[int, Document] = {}

        chunk_metadata: list[dict[str, int]] = []
        all_chunks: list[str] = []
        for i, doc in enumerate(self.documents):
            self.document_map[doc.id] = doc

            if doc.description == "":
                continue
            chunks = semantic_chunk(doc.description, 4, 1)

            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "movie_idx": i,
                        "chunk_idx": len(all_chunks) - 1,
                        "total_chunks": len(chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Document]):
        self.documents = documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc.id] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call 'load_or_create_chunk_embeddings' first."
            )

        query_embedding = self.generate_embedding(query.strip())

        chunk_scores = []
        movie_scores = {}
        for i, doc_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            metadata = self.chunk_metadata[i]
            movie_idx = metadata["movie_idx"]
            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": movie_idx,
                    "score": similarity,
                }
            )

            if movie_scores.get(movie_idx, float("-inf")) < similarity:
                movie_scores[movie_idx] = similarity

        sorted_movie_indices = sorted(
            movie_scores.keys(), key=lambda idx: movie_scores[idx], reverse=True
        )

        def convert_to_dict(movie_idx):
            score = movie_scores[movie_idx]
            doc = self.documents[movie_idx]

            return {
                "id": doc.id,
                "title": doc.title,
                "document": doc.description[:DEFAULT_PEVIEW_LENGTH],
                "score": round(score, SCORE_PRECISION),
            }

        output = map(convert_to_dict, sorted_movie_indices[:limit])
        return output


def cosine_similarity(vec1: ArrayLike, vec2: ArrayLike) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search = SemanticSearch()
    movies = load_movies()
    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query.strip())
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search = SemanticSearch()
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


def join_chunks(filler: str, chunks: list[str], chunk_size: int, overlap: int):
    """Reduces a list of chunks into a new list of chunks, with the specified size and overlap."""
    output: list[str] = []

    while len(chunks) > chunk_size:
        output.append(filler.join(chunks[:chunk_size]))
        chunks = chunks[chunk_size - overlap :]
    if not output or len(chunks) >= overlap:
        output.append(filler.join(chunks))

    return output


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_FIXED_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    return join_chunks(" ", words, chunk_size, overlap)


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return join_chunks(" ", parts, max_chunk_size, overlap)


def chunk_command(
    text: str,
    chunk_size: int = DEFAULT_FIXED_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)

    print(f"Chunking {len(text)} characters")
    for i, line in enumerate(chunks, 1):
        print(f"{i}. {line}")


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    chunks = semantic_chunk(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i, line in enumerate(chunks, 1):
        print(f"{i}. {line}")


def embed_chunks_command():
    search = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int = 10):
    search = ChunkedSemanticSearch()
    movies = load_movies()
    _ = search.load_or_create_chunk_embeddings(movies)

    results = search.search_chunks(query, limit)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
