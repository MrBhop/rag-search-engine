#!/usr/bin/env python3

import argparse

from lib.search_utils import BM25_B, BM25_K1
from lib.keyword_search import (
    bm25_idf_command,
    bm25_tf_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tf_idf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search documents using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")

    _ = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a given document ID and term"
    )
    _ = tf_parser.add_argument("doc_id", type=int, help="Document Id")
    _ = tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a given term"
    )
    _ = idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a given document ID and term"
    )
    _ = tfidf_parser.add_argument("doc_id", type=int, help="Document Id")
    _ = tfidf_parser.add_argument("term", type=str, help="Search term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    _ = bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    _ = bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    _ = bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for"
    )
    _ = bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    _ = bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "idf":
            print("Checking iverse document frequency...")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf}")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, movie in enumerate(results, 1):
                print(f"{i}. ({movie.id}) {movie.title}")
        case "tf":
            print("Checking term frequency...")
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "tfidf":
            print("Checking TF-IDF score...")
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            print("Checking BM25 idf score...")
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            print("Cheking BM25 tf score...")
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
