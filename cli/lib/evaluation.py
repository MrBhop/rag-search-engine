import json
import os
from dotenv import load_dotenv

from google import genai


from lib.search_utils import FormattedSearchResult

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("Missing API Key.")
model_name = "gemini-2.0-flash-001"
client = genai.Client(api_key=api_key)


def evaluate_results_llm(query: str, results: list[FormattedSearchResult]) -> list[FormattedSearchResult]:
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{"\n".join([str(item) for item in results])}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    if response.text is None:
        raise ValueError("Failed to get a response from the LLM.")
    llm_scores = json.loads(response.text.strip())

    def get_scored_formatted_search_result(index: int, llm_score: str) -> FormattedSearchResult:
        doc = results[index]
        doc.metadata["llm_score"] = llm_score
        return doc

    output = [
        get_scored_formatted_search_result(index, str(llm_score))
        for index, llm_score in enumerate(llm_scores)
    ]
    return output
