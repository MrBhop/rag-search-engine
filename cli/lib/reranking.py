from time import sleep
import os
from dotenv.main import load_dotenv

from lib.search_utils import FormattedSearchResult

from google import genai


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("Missing API Key.")
model_name = "gemini-2.0-flash-001"
client = genai.Client(api_key=api_key)


def prompt_model_for_new_score(query: str, doc: FormattedSearchResult):
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.title} - {doc.document}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    if response.text is None:
        raise ValueError("Failed to get a response from the LLM.")
    score = float(response.text)
    return score


def rerank_individual(results: list[FormattedSearchResult], query: str, limit: int):
    llm_scores = {}
    for index, doc in enumerate(results):
        new_score = prompt_model_for_new_score(query, doc)
        llm_scores[index] = new_score
        # sleep to avoid rate limits
        sleep(0.5)

    sorted_keys = sorted(
        llm_scores.keys(), key=lambda index: llm_scores[index], reverse=True
    )

    def get_formatted_reranked_result(index: int):
        result = results[index]
        result.metadata["reranked_score"] = llm_scores[index]
        return result

    reranked_results = list(map(get_formatted_reranked_result, sorted_keys[:limit]))
    return reranked_results
