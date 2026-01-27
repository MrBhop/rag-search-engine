import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

model_name = "gemini-2.0-flash-001"
system_prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise Exception("Missing API Key.")

    parser = argparse.ArgumentParser(description="Image + text -> rewritten query")
    _ = parser.add_argument(
        "--image", type=str, required=True, help="Path to image file"
    )
    _ = parser.add_argument("--query", type=str, required=True, help="Text query")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    with open(image_path, "rb") as f:
        image = f.read()

    query = args.query.strip()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            system_prompt,
            types.Part.from_bytes(data=image, mime_type=mime),
            query,
        ],
    )

    rewritten_query = (response.text or "").strip()

    print(f"Rewritten query: {rewritten_query}")
    if response.usage_metadata is not None:
        print(f"Total tokens:\t{response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
