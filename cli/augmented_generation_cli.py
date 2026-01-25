import argparse

from lib.augmented_generation import rag_command, summarize_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate multi-document summary"
    )
    _ = summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    _ = summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of documents to summarize"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            result = rag_command(query)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"\t-{doc.title}")
            print()
            print("RAG Response:")
            print(result["answer"])
        case "summarize":
            result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for doc in result["search_results"]:
                print(f"\t-{doc.title}")
            print()
            print("LLM Summary:")
            print(result["answer"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
