#!/usr/bin/env python
import argparse
from pathlib import Path
from docbot.app import run_query
from docbot.common import DEFAULT_DATABASE, DEFAULT_CHROMA_PATH, get_stored_model

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local support assistant.")
    parser.add_argument(
        "--model",
        default=get_stored_model(),
        help="Ollama model to use",
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help=f"Chroma collection/database to query (default: {DEFAULT_DATABASE})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db_dir = BASE_DIR / DEFAULT_CHROMA_PATH

    print("🔍 Local Support Assistant")
    print("=" * 50)
    print(f"🧩 Model: {args.model}")
    print(f"🗄️ Database: {args.database}")

    print("\n💡 Ask your support questions (type 'quit' or 'exit' to stop)")
    print("-" * 50)

    while True:
        try:
            query = input("\n🤔 Your question: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            if not query:
                continue

            run_query(
                query=query,
                db_dir=db_dir,
                database_name=args.database,
                model_name=args.model,
            )
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
