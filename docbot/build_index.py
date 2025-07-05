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
from common import get_chroma_settings, CHROMA_PATH, setup_logger
import shutil

BASE_DIR = Path(__file__).resolve().parent
LOGGER = setup_logger()


def main():
    """Main function to build the index."""

    LOGGER.info("Starting index build process...")

    chroma_dir = BASE_DIR / CHROMA_PATH
    if chroma_dir.exists():
        LOGGER.info(f"Removing existing ChromaDB directory at {chroma_dir}")
        shutil.rmtree(chroma_dir)

    LOGGER.info("🔧 Building index...")

    # Set up embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    # Load and chunk documents
    documents = SimpleDirectoryReader(Path("docs/")).load_data()
    parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)

    # Chroma DB setup
    chroma_settings = get_chroma_settings(BASE_DIR)
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=chroma_settings
    )
    chroma_collection = chroma_client.get_or_create_collection("support_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Build and store index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    storage_context.persist(persist_dir=chroma_settings.persist_directory)

    print("✅ Index built.")


if __name__ == "__main__":
    main()
