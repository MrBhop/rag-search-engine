from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch

from dotenv import load_dotenv
import os

from google import genai
from lib.search_utils import FormattedSearchResult, DEFAULT_SEARCH_LIMIT


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("Missing API Key.")
model_name = "gemini-2.0-flash-001"
client = genai.Client(api_key=api_key)


def generate_answer(search_results: list[FormattedSearchResult], query: str, limit=5):
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{"\n\n".join([f"{doc.title}: {doc.document}" for doc in search_results])}

Provide a comprehensive answer that addresses the query:"""
    response = client.models.generate_content(model=model_name, contents=prompt)
    if response.text is None:
        raise ValueError("Failed to get a response from the LLM.")
    return response.text

def rag(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, limit=limit)

    answer = generate_answer(results, query, limit)

    return {
        "query": query,
        "search_results": results,
        "answer": answer,
    }


def rag_command(query: str):
    return rag(query)
