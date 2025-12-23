import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict[str, str]]:
    movies = load_movies()
    results: list[dict[str, str]] = []

    query_tokens = tokenize_text(query)

    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if title_contains_query(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def title_contains_query(query: list[str], title: list[str]) -> bool:
    for token in query:
        for title_token in title:
            if token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    lowered_text = text.lower()
    return lowered_text.translate(str.maketrans("", "", string.punctuation))


def tokenize_text(text: str) -> list[str]:
    preprocessed = preprocess_text(text)
    tokens = preprocessed.split(" ")
    return list(filter(lambda item: item != "", tokens))
