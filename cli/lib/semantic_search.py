import os
import numpy as np

from sentence_transformers import SentenceTransformer

from lib.search_utils import MOVIE_EMBEDDINGS_PATH, Movie, load_movies


class SemanticSerach:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[Movie] = []
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode(text)

    def build_embeddings(self, documents: list[Movie]):
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

    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}

        for doc in self.documents:
            self.document_map[doc.id] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)


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
