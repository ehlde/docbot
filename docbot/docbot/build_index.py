from pathlib import Path
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from docbot.common import get_chroma_settings, setup_logger, DEFAULT_DATABASE

LOGGER = setup_logger()


def run_indexing(
    repo_dir: Path,
    db_dir: Path,
    database_name: str = DEFAULT_DATABASE,
    docs_dir: Path = None,
    extra_exts: list = None,
    include_files: list = None,
):
    """Recursively scan for documentation files and build/rebuild the Chroma index."""
    LOGGER.info(f"Building index from {repo_dir} for database '{database_name}'...")

    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    input_dir = docs_dir if docs_dir else repo_dir
    LOGGER.info(f"Scanning directory: {input_dir}")

    # Build default extension list
    default_exts = [".md", ".txt", ".rst", ".pdf", ".docx"]
    if extra_exts:
        for ext in extra_exts:
            if not ext.startswith("."):
                ext = f".{ext}"
            if ext not in default_exts:
                default_exts.append(ext)

    LOGGER.info(f"Allowed extensions: {default_exts}")

    # 1. Load data recursively from folder
    documents = []
    if input_dir.exists():
        reader = SimpleDirectoryReader(
            input_dir=str(input_dir),
            recursive=True,
            required_exts=default_exts,
            exclude=[
                "**/node_modules/**",
                "**/.venv/**",
                "**/.git/**",
                "**/.chroma/**",
            ],
        )
        documents = reader.load_data()
    else:
        LOGGER.warning(f"Scan directory does not exist: {input_dir}")

    # 2. Add explicitly included files
    if include_files:
        LOGGER.info(f"Explicitly including files: {include_files}")
        for file_path in include_files:
            p = Path(file_path)
            if p.exists() and p.is_file():
                try:
                    file_reader = SimpleDirectoryReader(input_files=[str(p)])
                    documents.extend(file_reader.load_data())
                except Exception as e:
                    LOGGER.warning(f"Failed to read included file {file_path}: {e}")
            else:
                LOGGER.warning(f"Included file path does not exist or is not a file: {file_path}")

    if not documents:
        LOGGER.warning("⚠️ No documents were found to index.")
        # If no documents exist, we still want to make sure the database setup is clean
        # but we can return early.
        return

    parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)

    # 3. Setup Chroma and replace the specific collection
    chroma_settings = get_chroma_settings(db_dir)
    chroma_client = chromadb.PersistentClient(
        path=str(db_dir), settings=chroma_settings
    )

    existing_collections = {collection.name for collection in chroma_client.list_collections()}
    if database_name in existing_collections:
        LOGGER.info(f"Replacing existing database collection: {database_name}")
        chroma_client.delete_collection(database_name)

    chroma_collection = chroma_client.get_or_create_collection(database_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context)
    storage_context.persist(persist_dir=str(db_dir))

    LOGGER.info(f"✅ Index successfully built and saved to {db_dir} (collection: {database_name}).")
