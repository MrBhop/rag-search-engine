import os
from dotenv.main import load_dotenv

from google import genai


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise Exception("Missing API Key.")
model_name = "gemini-2.0-flash-001"
client = genai.Client(api_key=api_key)


def prompt_model_for_query_enhancement(query: str, prompt_template: str):
    prompt = prompt_template % query
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query


def spell_correct(query: str) -> str:
    prompt_template = """Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "%s"

If no errors, return the original query.
Corrected:"""
    return prompt_model_for_query_enhancement(query, prompt_template)


def rewrite(query: str):
    prompt_template = """Rewrite this movie search query to be more specific and searchable.

Original: "%s"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    return prompt_model_for_query_enhancement(query, prompt_template)


def expand(query: str):
    prompt_template = """Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "%s"
"""
    expanded_terms = prompt_model_for_query_enhancement(query, prompt_template)
    return f"{query} {expanded_terms}"


def enhance_query(query: str, method: str | None = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite(query)
        case "expand":
            return expand(query)
        case _:
            return query
