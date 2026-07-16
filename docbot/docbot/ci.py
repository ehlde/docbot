import os
import json
import requests
from pathlib import Path
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from docbot.common import get_chroma_settings, setup_logger, DEFAULT_DATABASE

LOGGER = setup_logger()


def run_ci_index(repo_dir: Path, db_dir: Path, database_name: str = DEFAULT_DATABASE):
    """Run in CI environment to pull PR description and comments, and index them."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    github_token = os.environ.get("GITHUB_TOKEN")

    if not event_path:
        LOGGER.error("❌ GITHUB_EVENT_PATH not set. CI command must run in GitHub Actions.")
        return

    try:
        with open(event_path, "r") as f:
            event = json.load(f)
    except Exception as e:
        LOGGER.error(f"❌ Failed to parse GITHUB_EVENT_PATH file: {e}")
        return

    repo_name = event.get("repository", {}).get("full_name")
    pr_number = event.get("pull_request", {}).get("number")

    if not pr_number:
        pr_number = event.get("issue", {}).get("number")

    if not repo_name or not pr_number:
        LOGGER.error("❌ Could not extract repository name or PR number from event.")
        return

    LOGGER.info(f"Indexing PR #{pr_number} from {repo_name}...")

    # 1. Fetch description and comments
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    comments_url = f"https://api.github.com/repos/{repo_name}/issues/{pr_number}/comments"
    comments_text = ""
    try:
        res = requests.get(comments_url, headers=headers, timeout=30)
        if res.status_code == 200:
            comments = res.json()
            comments_text = "\n".join([f"- {c['user']['login']}: {c['body']}" for c in comments])
        else:
            LOGGER.warning(f"Failed to fetch PR comments: {res.status_code} - {res.text}")
    except Exception as e:
        LOGGER.warning(f"Failed to fetch comments: {e}")

    pr_title = event.get("pull_request", {}).get("title", f"PR #{pr_number}")
    pr_body = event.get("pull_request", {}).get("body", "")

    # 2. Build virtual PR document
    pr_content = f"# GitHub PR #{pr_number}: {pr_title}\n\n## Description\n{pr_body}\n\n## Discussion/Comments\n{comments_text}"
    document = Document(
        text=pr_content,
        metadata={
            "file_name": f"pr_{pr_number}_discussion.md",
            "source": "github_pr",
            "pr_number": str(pr_number),
        },
    )

    # 3. Add to existing index (load existing collection, insert, and save)
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chroma_settings = get_chroma_settings(db_dir)
    chroma_client = chromadb.PersistentClient(
        path=str(db_dir), settings=chroma_settings
    )

    chroma_collection = chroma_client.get_or_create_collection(database_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        # Load existing index to append
        VectorStoreIndex.from_documents(
            [document], storage_context=storage_context
        )
        storage_context.persist(persist_dir=str(db_dir))
        LOGGER.info(f"✅ Indexed PR #{pr_number} description and comments successfully.")
    except Exception as e:
        LOGGER.error(f"❌ Failed to update Chroma index with PR: {e}")
