import argparse
from pathlib import Path
import sys

from docbot.common import (
    get_stored_model,
    set_stored_model,
    DEFAULT_CHROMA_PATH,
    DEFAULT_DATABASE,
)
from docbot.build_index import run_indexing
from docbot.app import run_query, run_chat
from docbot.ci import run_ci_index


def main():
    parser = argparse.ArgumentParser(
        description="Docbot CLI: Search and index codebase documentation."
    )

    subparsers = parser.add_subparsers(dest="command")

    # docbot chat / default
    chat_parser = subparsers.add_parser(
        "chat", help="Start an interactive chat session (Default)"
    )
    chat_parser.add_argument(
        "--model", help="Override default model for this session"
    )
    chat_parser.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help="Path to database folder",
    )
    chat_parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE,
        help="Name of the Chroma collection",
    )

    # docbot index
    index_parser = subparsers.add_parser(
        "index", help="Recursively scan and index documentation files"
    )
    index_parser.add_argument(
        "--dir", type=str, default=".", help="Root directory of codebase to scan"
    )
    index_parser.add_argument(
        "--docs-dir",
        type=str,
        help="Specific folder containing docs to index (relative to --dir)",
    )
    index_parser.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help="Path to database folder",
    )
    index_parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE,
        help="Name of the Chroma collection",
    )
    index_parser.add_argument(
        "--ext",
        type=str,
        help="Comma-separated additional file extensions to index (e.g. log,csv)",
    )
    index_parser.add_argument(
        "--include",
        type=str,
        help="Comma-separated specific files to explicitly index",
    )

    # docbot query
    query_parser = subparsers.add_parser(
        "query", help="Ask a question about the documentation"
    )
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument(
        "--model", help="Override default model for this query"
    )
    query_parser.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help="Path to database folder",
    )
    query_parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE,
        help="Name of the Chroma collection",
    )

    # docbot model
    model_parser = subparsers.add_parser(
        "model", help="Configure or print the default Ollama model"
    )
    model_parser.add_argument(
        "model_name",
        nargs="?",
        help="Model name to store as default (e.g. gemma4:12b)",
    )

    # docbot ci
    ci_parser = subparsers.add_parser(
        "ci",
        help="CI integration to fetch and index PR metadata and discussions",
    )
    ci_parser.add_argument(
        "--dir", type=str, default=".", help="Root directory of codebase"
    )
    ci_parser.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help="Path to database folder",
    )
    ci_parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE,
        help="Name of the Chroma collection",
    )

    # Default to "chat" if no command provided
    if len(sys.argv) == 1:
        sys.argv.append("chat")

    args = parser.parse_args()

    if args.command == "chat":
        model = args.model or get_stored_model()
        db_dir = Path(args.db_dir).resolve()
        run_chat(db_dir=db_dir, database_name=args.database, model_name=model)

    elif args.command == "index":
        repo_dir = Path(args.dir).resolve()
        db_dir = Path(args.db_dir).resolve()
        docs_dir = Path(args.docs_dir).resolve() if args.docs_dir else None

        extra_exts = [x.strip() for x in args.ext.split(",")] if args.ext else []
        include_files = (
            [x.strip() for x in args.include.split(",")] if args.include else []
        )

        run_indexing(
            repo_dir=repo_dir,
            db_dir=db_dir,
            database_name=args.database,
            docs_dir=docs_dir,
            extra_exts=extra_exts,
            include_files=include_files,
        )

    elif args.command == "query":
        model = args.model or get_stored_model()
        db_dir = Path(args.db_dir).resolve()
        run_query(
            query=args.question,
            db_dir=db_dir,
            database_name=args.database,
            model_name=model,
        )

    elif args.command == "model":
        if args.model_name:
            set_stored_model(args.model_name)
        else:
            print(f"Current default model: {get_stored_model()}")

    elif args.command == "ci":
        repo_dir = Path(args.dir).resolve()
        db_dir = Path(args.db_dir).resolve()
        run_ci_index(
            repo_dir=repo_dir, db_dir=db_dir, database_name=args.database
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
