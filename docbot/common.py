from chromadb.config import Settings
import logging
import sys
from pathlib import Path

DEFAULT_CHROMA_PATH = ".chroma"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM = "gemma3n:e2b"
BASE_DIR = Path(__file__).resolve().parent


def get_chroma_settings(chroma_path: Path) -> Settings:
    """Get ChromaDB settings for the application."""
    return Settings(
        persist_directory=str(chroma_path),
        anonymized_telemetry=False,
    )


def setup_logger():
    """Set up a simple logger."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger
