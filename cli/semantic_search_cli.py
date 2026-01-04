#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_FIXED_CHUNK_SIZE, DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_OVERLAP, DEFAULT_SEMANTIC_CHUNK_SIZE
from lib.semantic_search import (
    chunk_command,
    embed_chunks_command,
    embed_query_text,
    embed_text,
    semantic_chunk_text,
    semantic_search,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser(
        "verify", help="Verify that the embedding model is loaded"
    )

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    _ = single_embed_parser.add_argument("text", type=str, help="Text to embed")

    _ = subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset"
    )

    query_embed_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a search query"
    )
    _ = query_embed_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search for movies using semantic search"
    )
    _ = search_parser.add_argument("query", type=str, help="Search query")
    _ = search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Number of results to return",
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size chunks with optional overlap"
    )
    _ = chunk_parser.add_argument("text", type=str, help="Text to chunk")
    _ = chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        help="Size of each chunk in words",
        default=DEFAULT_FIXED_CHUNK_SIZE,
    )
    _ = chunk_parser.add_argument(
        "--overlap",
        type=int,
        help="Number of words to overlap between chunks",
        default=DEFAULT_CHUNK_OVERLAP,
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text on sentence boundaries to preserve meaning"
    )
    _ = semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    _ = semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        help="Maximum size of each chunk in sentences",
        default=DEFAULT_SEMANTIC_CHUNK_SIZE,
    )
    _ = semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        help="Number of sentences to overlap between chunks",
        default=DEFAULT_CHUNK_OVERLAP,
    )

    _ = subparsers.add_parser("embed_chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
