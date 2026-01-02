from sentence_transformers import SentenceTransformer


class SemanticSerach:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)


def verify_model():
    search = SemanticSerach()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
