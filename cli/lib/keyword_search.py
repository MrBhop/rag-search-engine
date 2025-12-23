import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict[str, str]]:
    movies = load_movies()
    results: list[dict[str, str]] = []

    preprocessed_query = preprocess_text(query)

    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        if preprocessed_query in preprocessed_title:
            results.append(movie)

            if len(results) >= limit:
                break

    return results


def preprocess_text(text: str) -> str:
    lowered_text = text.lower()
    translation_table = str.maketrans("", "", string.punctuation)
    punctuation_removed = lowered_text.translate(translation_table)
    return punctuation_removed
