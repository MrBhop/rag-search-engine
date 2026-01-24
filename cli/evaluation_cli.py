import json
import argparse

from lib.search_utils import GOLDEN_DATASET_PATH, load_movies
from lib.hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit: int = args.limit
    movies = load_movies()
    srch = HybridSearch(movies)

    # run evaluation logic here
    with open(GOLDEN_DATASET_PATH, "r") as f:
        test_data = json.load(f)

    print(f"k={limit}")

    for case in test_data["test_cases"]:
        query: str = case["query"]
        relevant_docs: list[str] = case["relevant_docs"]
        retrieved_docs = srch.rrf_search(query, 60, limit)

        relevant_retrieved = 0
        for doc in retrieved_docs:
            if doc.title in relevant_docs:
                relevant_retrieved += 1

        precision_at_k = relevant_retrieved / len(retrieved_docs)
        recall_at_k = relevant_retrieved / len(relevant_docs)
        f1_score = (
            0
            if precision_at_k + recall_at_k == 0
            else 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
        )

        print(
            f"\n- Query: {query}\n"
            f"\t- Precision@{limit}: {precision_at_k:.4f}\n"
            f"\t- Recall@{limit}: {recall_at_k:.4f}\n"
            f"\t- F1 Score: {f1_score:.4f}\n"
            f"\t- Retrieved: {', '.join([doc.title for doc in retrieved_docs])}\n"
            f"\t- Relevant: {', '.join(relevant_docs)}\n"
        )


if __name__ == "__main__":
    main()
