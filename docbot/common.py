from chromadb.config import Settings
import logging
import sys
from pathlib import Path

CHROMA_PATH = ".chroma"


def get_chroma_settings(base_dir: Path) -> Settings:
    """Get ChromaDB settings for the application."""
    return Settings(
        persist_directory=str(base_dir / CHROMA_PATH),
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
