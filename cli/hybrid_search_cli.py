import argparse

from lib.hybrid_search import normalize_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "values", type=float, nargs="+", help="List of scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.values)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
