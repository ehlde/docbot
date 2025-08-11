import argparse
from pathlib import Path
import time

from common import setup_logger, DEFAULT_LLM, DEFAULT_EMBEDDING_MODEL
from query_engine import setup_query_engine

LOGGER = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Local Support Assistant")
    parser.add_argument("--llm", default=DEFAULT_LLM, help="LLM model to use")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model to use for indexing",
    )
    parser.add_argument(
        "--num-sources", type=int, default=3, help="Number of sources to display"
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.48,
        help="Relevance threshold for filtering sources",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--db-path",
        default=str(Path(".chroma")),
        help="Path to the ChromaDB directory",
    )
    return parser.parse_args()


def main(
    llm: str,
    embedding_model: str,
    database_path: Path,
    num_sources: int,
    timeout: int,
    relevance_threshold: float,
):
    """Main CLI interface."""
    LOGGER.info("Starting Local Support Assistant")

    print("🔍 Local Support Assistant")
    print("=" * 50)

    # Setup query engine
    query_engine = setup_query_engine(
        llm, embedding_model, database_path, num_sources, timeout, relevance_threshold
    )

    print("\n💡 Ask your support questions, in English. Type 'quit' or 'exit' to stop.")
    print("-" * 50)

    while True:
        try:
            # Get user input
            query = input("\n🤔 Your question: ").strip()

            # Check for exit commands
            if query.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            # Skip empty queries
            if not query:
                continue

            # Process the query
            print("\n🧠 Thinking...")

            start_time = time.monotonic()

            response = query_engine.query(query)

            # Calculate and print the duration of the query
            duration = time.monotonic() - start_time
            print(f"\n⏱️ Query processed in {duration:.2f} seconds.")

            # Display source information first (compact format)
            if hasattr(response, "source_nodes") and response.source_nodes:
                print("\n📚 Sources:")
                for i, node in enumerate(response.source_nodes, 1):
                    # Get metadata
                    file_name = node.node.metadata.get("file_name", "Unknown file")
                    page_label = node.node.metadata.get("page_label", "")

                    # Display compact source info
                    source_info = f"{i}. {file_name}"
                    if page_label:
                        source_info += f" - Page {page_label}."
                    if hasattr(node, "score"):
                        source_info += f" Relevance: {node.score:.3f}"
                    print(source_info)

                    if i > 5:
                        print("... (more sources not displayed)")
                        break
                print()

            # Display the response after sources
            print("📖 Answer:")
            print("-" * 30)
            print(str(response).strip("\n"))
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    args = parse_args()
    LOGGER.info(
        f"LLM: {args.llm}\nEmbedding Model: {args.embedding_model}\nNumber of sources: {args.num_sources}\nTimeout: {args.timeout}s"
    )

    main(
        args.llm,
        args.embedding_model,
        args.db_path,
        args.num_sources,
        args.timeout,
        args.relevance_threshold,
    )
