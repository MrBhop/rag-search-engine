import argparse

from lib.multimodal_search import verify_image_embedding, image_search_command
from lib.search_utils import DEFAULT_PEVIEW_LENGTH


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding"
    )
    verify_image_embedding_parser.add_argument(
        "image", type=str, help="Path to image file"
    )
    image_search_parser = subparsers.add_parser(
        "image_search", help="Search documents using an image"
    )
    image_search_parser.add_argument(
        "image", type=str, help="Path to image file"
    )

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            results = image_search_command(args.image)
            for index, res in enumerate(results, 1):
                doc = res["doc"]
                score = res["score"]
                print(f"{index}. {doc.title} (similarity: {score:.3f})\n\t{doc.description[:DEFAULT_PEVIEW_LENGTH]}...\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
