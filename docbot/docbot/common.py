import json
import logging
import sys
from pathlib import Path
import ollama
from chromadb.config import Settings

DEFAULT_CHROMA_PATH = ".chroma"
DEFAULT_DATABASE = "support_docs"
DEFAULT_MODEL = "gemma4:12b"
CONFIG_DIR = Path.home() / ".config" / "docbot"
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_chroma_settings(db_path: Path) -> Settings:
    """Get ChromaDB settings for the application."""
    return Settings(
        persist_directory=str(db_path),
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
    logger = logging.getLogger("docbot")
    return logger


def get_stored_model() -> str:
    """Load stored model from config file, fallback to DEFAULT_MODEL."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("model", DEFAULT_MODEL)
        except Exception:
            pass
    return DEFAULT_MODEL


def set_stored_model(model_name: str):
    """Save model to config file."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump({"model": model_name}, f)
        print(f"💾 Default model set to '{model_name}' for future sessions.")
    except Exception as e:
        print(f"❌ Failed to save default model: {e}")


def ensure_model_available(model_name: str):
    """Check if model is pulled in Ollama, pull if missing."""
    try:
        local_models = ollama.list()
        model_names = []
        if hasattr(local_models, "models"):
            model_names = [m.model for m in local_models.models]
        elif isinstance(local_models, dict) and "models" in local_models:
            model_names = [m.get("model") or m.get("name") for m in local_models["models"]]
        else:
            model_names = [getattr(m, "model", "") or getattr(m, "name", "") for m in local_models]

        normalized_name = model_name if ":" in model_name else f"{model_name}:latest"
        exists = any(name == model_name or name == normalized_name for name in model_names)

        if not exists:
            print(f"📥 Model '{model_name}' not found locally. Pulling from Ollama registry...")
            ollama.pull(model_name)
            print(f"✅ Model '{model_name}' successfully pulled!")
    except Exception as e:
        print(f"⚠️ Warning: Could not check or pull model '{model_name}' via Ollama: {e}")
