from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import argparse
from pathlib import Path

from common import get_chroma_settings, CHROMA_PATH, DEFAULT_DATABASE, setup_logger

BASE_DIR = Path(__file__).resolve().parent
LOGGER = setup_logger()
MODEL_NAME = "gemma3n:e2b"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local support assistant.")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Ollama model to use (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help=f"Chroma collection/database to query (default: {DEFAULT_DATABASE})",
    )
    return parser.parse_args()


def setup_query_engine(model_name: str, database_name: str):
    """Set up the query engine with vector store and LLM."""
    LOGGER.info("🔧 Setting up query engine...")

    # Set embedding model first
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    # Set up vector store + LLM
    chroma_settings = get_chroma_settings(BASE_DIR)
    LOGGER.info(f"ChromaDB settings: {chroma_settings.persist_directory}")
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=chroma_settings,
    )
    available_collections = {collection.name for collection in chroma_client.list_collections()}
    if database_name not in available_collections:
        available_label = ", ".join(sorted(available_collections)) or "<none>"
        raise ValueError(
            f"Database '{database_name}' was not found. "
            f"Available databases: {available_label}. "
            "Build it first with build_index.py."
        )

    chroma_collection = chroma_client.get_collection(database_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=chroma_settings.persist_directory
    )

    index = load_index_from_storage(storage_context)
    LOGGER.info("✅ Index loaded successfully!")

    llm = Ollama(
        model=model_name,
        request_timeout=60,
        system_prompt="You are a precise assistant. Only use the provided context to answer. If unsure, say 'I don't know.'",
    )

    retriever = index.as_retriever(similarity_top_k=3)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        response_synthesizer=CompactAndRefine(llm=llm),
        node_postprocessors=[],
    )

    return query_engine


def main():
    """Main CLI interface."""
    args = parse_args()
    LOGGER.info("Starting Local Support Assistant")

    print("🔍 Local Support Assistant")
    print("=" * 50)
    print(f"🧩 Model: {args.model}")
    print(f"🗄️ Database: {args.database}")

    # Setup query engine
    query_engine = setup_query_engine(model_name=args.model, database_name=args.database)

    print("\n💡 Ask your support questions (type 'quit' or 'exit' to stop)")
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
            response = query_engine.query(query)

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
                        source_info += f" (Page {page_label})"
                    if hasattr(node, "score"):
                        source_info += f" - Relevance: {node.score:.3f}"
                    print(source_info)
                print()

            # Display the response after sources
            print("📖 Answer:")
            print("-" * 30)
            print(str(response))
            print("-" * 30)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
