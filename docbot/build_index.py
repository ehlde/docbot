#!/usr/bin/env python
from pathlib import Path
import argparse
from docbot.build_index import run_indexing
from docbot.common import DEFAULT_DATABASE, DEFAULT_CHROMA_PATH

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Build the local document index.")
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help=f"Chroma collection/database name (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--docs-path",
        default=str(BASE_DIR / "docs"),
        help="Directory containing source documents (default: ./docs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    docs_path = Path(args.docs_path)
    db_dir = BASE_DIR / DEFAULT_CHROMA_PATH

    run_indexing(
        repo_dir=BASE_DIR,
        db_dir=db_dir,
        database_name=args.database,
        docs_dir=docs_path,
    )


if __name__ == "__main__":
    main()
