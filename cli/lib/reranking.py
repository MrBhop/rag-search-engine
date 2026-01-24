from sentence_transformers import CrossEncoder
import json
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
cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")


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


def rerank_batch(results: list[FormattedSearchResult], query: str, limit: int):
    # list of 'document strings' for passing to the LLM.
    doc_list = []
    # map doc ids, back to the indices in the original result.
    ids_to_index = {}
    for index, doc in enumerate(results):
        ids_to_index[doc.id] = index
        doc_list.append(f"{doc.id}: {doc.title} - {doc.document[:200]}...")
    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else, also no code block. For example:

[75, 12, 34, 2, 1]
"""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    if response.text is None:
        raise ValueError("Failed to get a response from the LLM.")
    reranked_ids = json.loads(response.text.strip())

    def get_formatted_reranked_result(rank: int, id: str):
        # get index in original results from document id.
        index = ids_to_index[id]
        result = results[index]
        result.metadata["batch_rank"] = rank
        return result

    reranked_results = [
        get_formatted_reranked_result(rank, str(id))
        for rank, id in enumerate(reranked_ids[:limit], 1)
    ]
    return reranked_results


def rerank_cross_encoder(results: list[FormattedSearchResult], query: str, limit: int):
    pairs = []
    for doc in results:
        pairs.append([query, f"{doc.title} - {doc.document}"])

    scores = cross_encoder.predict(pairs)

    sorted_indices = sorted(
        range(len(scores)), key=lambda index: scores[index], reverse=True
    )

    def get_formatted_reranked_result(index: int):
        result = results[index]
        result.metadata["cross_encoder_score"] = scores[index]
        return result

    reranked_results = list(map(get_formatted_reranked_result, sorted_indices[:limit]))
    return reranked_results
