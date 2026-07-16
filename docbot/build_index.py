from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb
import argparse
from common import get_chroma_settings, CHROMA_PATH, DEFAULT_DATABASE, setup_logger

BASE_DIR = Path(__file__).resolve().parent
LOGGER = setup_logger()


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
    """Main function to build the index."""
    args = parse_args()
    docs_path = Path(args.docs_path)

    LOGGER.info("Starting index build process...")
    LOGGER.info(f"Using database: {args.database}")
    LOGGER.info(f"Using docs path: {docs_path}")

    LOGGER.info("🔧 Building index...")

    # Set up embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    # Load and chunk documents
    documents = SimpleDirectoryReader(docs_path).load_data()
    parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)

    # Chroma DB setup
    chroma_settings = get_chroma_settings(BASE_DIR)
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=chroma_settings
    )
    existing_collections = {collection.name for collection in chroma_client.list_collections()}
    if args.database in existing_collections:
        LOGGER.info(f"Replacing existing database: {args.database}")
        chroma_client.delete_collection(args.database)

    chroma_collection = chroma_client.get_or_create_collection(args.database)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Build and store index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context)
    storage_context.persist(persist_dir=chroma_settings.persist_directory)

    print(f"✅ Index built for database '{args.database}'.")


if __name__ == "__main__":
    main()
