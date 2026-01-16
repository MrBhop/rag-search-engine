import os
from dotenv import load_dotenv
from google import genai

MODEL_NAME = "gemini-2.0-flash-001"


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("Missing API Key.")
    print(f"Using key {api_key[:6]}...")

    client = genai.Client(api_key=api_key)
    res = client.models.generate_content(
        model=MODEL_NAME,
        contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    )
    print(res.text)
    metadata = res.usage_metadata
    if metadata:
        print(f"Prompt Tokens: {metadata.prompt_token_count}")
        print(f"Response Tokens: {metadata.candidates_token_count}")


if __name__ == "__main__":
    main()
