import os
from PIL import Image
from sentence_transformers import SentenceTransformer

from lib.search_utils import Document
from lib.semantic_search import cosine_similarity
from lib.search_utils import load_movies


class MultimodalSearch:
    def __init__(self, documents: list[Document] = [], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

        self.documents = documents
        self.texts = [f"{doc.title}: {doc.description}" for doc in self.documents]

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=False)

    def embed_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path)
        image_embedding = self.model.encode([image])  # ty:ignore[no-matching-overload]
        return image_embedding[0]

    def search_with_image(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image_embedding = self.embed_image(image_path)
        docs_with_scores = [{
            "doc": self.documents[index],
            "score": cosine_similarity(image_embedding, embedding),
        } for index, embedding in enumerate(self.text_embeddings)]
        sorted_results = sorted(docs_with_scores, key=lambda item: item["score"], reverse=True)
        return sorted_results[:5]


def verify_image_embedding(image_path: str):
    searcher = MultimodalSearch()
    embedding = searcher.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str):
    movies = load_movies()
    searcher = MultimodalSearch(movies)
    return searcher.search_with_image(image_path)
