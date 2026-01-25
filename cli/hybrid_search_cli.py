import logging
import argparse

from lib.search_utils import DEFAULT_ALPHA, RRF_K, DEFAULT_SEARCH_LIMIT
from lib.hybrid_search import (
    normalize_command,
    rrf_search_command,
    weighted_search_command,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "values", type=float, nargs="+", help="List of scores to normalize"
    )

    weighted_search_parser = subparser.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    _ = weighted_search_parser.add_argument("query", type=str, help="Search query")
    _ = weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default={DEFAULT_ALPHA})",
    )
    _ = weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Number of results to return (default={DEFAULT_SEARCH_LIMIT})",
    )

    rrf_search_parser = subparser.add_parser(
        "rrf-search", help="Perform Reciprocal Rank Fusion search"
    )
    _ = rrf_search_parser.add_argument("query", type=str, help="Search query")
    _ = rrf_search_parser.add_argument(
        "-k",
        type=int,
        help=f"RRF k parameter controlling weight distribution (default={RRF_K})",
        default=RRF_K,
    )
    _ = rrf_search_parser.add_argument(
        "--limit",
        type=int,
        help=f"Number of results to return (default={DEFAULT_SEARCH_LIMIT})",
        default=DEFAULT_SEARCH_LIMIT,
    )
    _ = rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    _ = rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.values)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(
                args.query, args.k, args.limit, args.enhance, args.rerank_method
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
