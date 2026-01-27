from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch

from dotenv import load_dotenv
import os

from google import genai
from lib.search_utils import DEFAULT_SEARCH_LIMIT


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("Missing API Key.")
model_name = "gemini-2.0-flash-001"
client = genai.Client(api_key=api_key)


def generate_answer(prompt: str):
    response = client.models.generate_content(model=model_name, contents=prompt)
    if response.text is None:
        raise ValueError("Failed to get a response from the LLM.")
    return response.text


def rag(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, limit=limit)

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{"\n\n".join([f"{doc.title}: {doc.document}" for doc in results])}

Provide a comprehensive answer that addresses the query:"""

    answer = generate_answer(prompt)

    return {
        "query": query,
        "search_results": results,
        "answer": answer,
    }


def summarize(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, limit=limit)

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{"\n\n".join([f"{doc.title}: {doc.document}" for doc in results])}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    answer = generate_answer(prompt)

    return {
        "query": query,
        "search_results": results,
        "answer": answer,
    }


def citations(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, limit=limit)

    prompt = (
        prompt
    ) = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{"\n\n".join([f"[{index}]: {doc.title}: {doc.document}" for index, doc in enumerate(results, 1)])}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    answer = generate_answer(prompt)

    return {
        "query": query,
        "search_results": results,
        "answer": answer,
    }


def rag_command(query: str):
    return rag(query)


def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    return summarize(query, limit)


def citations_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    return citations(query, limit)
