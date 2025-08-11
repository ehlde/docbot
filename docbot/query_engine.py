from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

from common import (
    get_chroma_settings,
    setup_logger,
    BASE_DIR,
)
from pathlib import Path
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from typing import List

LOGGER = setup_logger()


class RelevanceThresholdFilter(BaseNodePostprocessor):
    """Filters out nodes with a relevance score below a threshold."""

    threshold: float = 0.25

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle=None, **kwargs
    ) -> List[NodeWithScore]:
        filtered_nodes = [
            n for n in nodes if n.score is not None and n.score >= self.threshold
        ]
        return filtered_nodes


def setup_query_engine(
    model_name: str,
    embedding_model: str,
    database_path: Path,
    timeout: int = 180,
    num_sources: int = 3,
    relevance_threshold: float = 0.25,
) -> RetrieverQueryEngine:
    """Set up the query engine with vector store and LLM."""
    LOGGER.info("🔧 Setting up query engine...")

    # Set embedding model first
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

    # Set up vector store + LLM
    chroma_settings = get_chroma_settings(BASE_DIR)
    LOGGER.info(f"ChromaDB settings: {chroma_settings.persist_directory}")
    chroma_client = chromadb.PersistentClient(
        path=database_path,
        settings=chroma_settings,
    )
    chroma_collection = chroma_client.get_or_create_collection("support_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=chroma_settings.persist_directory
    )

    index = load_index_from_storage(storage_context)
    LOGGER.info("✅ Index loaded successfully!")

    llm = Ollama(
        model=model_name,
        request_timeout=timeout,
        system_prompt="You are a precise assistant. Only use the provided context to answer. If unsure, say 'I don't know.'",
    )

    retriever = index.as_retriever(similarity_top_k=num_sources)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        response_synthesizer=CompactAndRefine(llm=llm),
        node_postprocessors=[RelevanceThresholdFilter(threshold=relevance_threshold)],
    )

    return query_engine
