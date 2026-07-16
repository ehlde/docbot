from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

from docbot.common import (
    get_chroma_settings,
    setup_logger,
    ensure_model_available,
)

LOGGER = setup_logger()


def setup_query_engine(model_name: str, db_dir: Path, database_name: str):
    """Set up the query engine with vector store and LLM."""
    LOGGER.info("🔧 Setting up query engine...")

    ensure_model_available(model_name)

    # Set embedding model first
    Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

    # Set up vector store + LLM
    chroma_settings = get_chroma_settings(db_dir)
    chroma_client = chromadb.PersistentClient(
        path=str(db_dir),
        settings=chroma_settings,
    )

    available_collections = {collection.name for collection in chroma_client.list_collections()}
    if database_name not in available_collections:
        available_label = ", ".join(sorted(available_collections)) or "<none>"
        raise ValueError(
            f"Database collection '{database_name}' was not found at {db_dir}. "
            f"Available database collections: {available_label}. "
            "Build the index first using 'docbot index'."
        )

    chroma_collection = chroma_client.get_collection(database_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=str(db_dir)
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

    return query_engine, index, llm, retriever


def check_if_code_query(query: str, llm: Ollama) -> bool:
    """Analyze query and classify if it requires reading source code."""
    prompt = (
        f"Question: {query}\n\n"
        "Does answering this question require reading/inspecting the source code files of the repository? "
        "Respond with exactly 'YES' if it requires code inspection or codebase implementation details, "
        "or 'NO' if it can be answered solely via high-level documentation. "
        "Do not include any other words."
    )
    try:
        response = llm.complete(prompt)
        text = str(response).strip().upper()
        return "YES" in text
    except Exception as e:
        LOGGER.warning(f"Failed to run code classification query: {e}")
        return False


def generate_escalation_prompt(query: str, retrieved_nodes) -> str:
    """Generate prompt optimized for advanced code reasoning agents."""
    context = ""
    for i, node in enumerate(retrieved_nodes, 1):
        file_name = node.node.metadata.get("file_name") or "Unknown file"
        page_label = node.node.metadata.get("page_label", "")
        source_label = f"{file_name} (Page {page_label})" if page_label else file_name
        context += f"\n[Doc {i}] Source: {source_label}\n{node.node.text}\n"

    return f"""================================================================================
⚠️ CODE QUERY DETECTED - ESCALATING TO POWERFUL MODEL
Copy and run the prompt below in your coding agent (e.g., Claude Code, AGY):
================================================================================

You are an advanced agent working on this codebase. Please answer the following user query:
"{query}"

Here is relevant documentation context retrieved from our database:
{context}

Please inspect the source code files, locate the relevant implementation details, and answer the question.
================================================================================"""


def run_query(query: str, db_dir: Path, database_name: str, model_name: str):
    """Run a single query."""
    query_engine, index, llm, retriever = setup_query_engine(
        model_name=model_name, db_dir=db_dir, database_name=database_name
    )

    # 1. Classify the query
    is_code = check_if_code_query(query, llm)

    # 2. Retrieve nodes
    retrieved_nodes = retriever.retrieve(query)

    if is_code:
        # Output escalation prompt
        prompt = generate_escalation_prompt(query, retrieved_nodes)
        print(prompt)
    else:
        # Run local Q&A
        print("\n🧠 Thinking...")
        response = query_engine.query(query)

        # Display sources
        if retrieved_nodes:
            print("\n📚 Sources:")
            for i, node in enumerate(retrieved_nodes, 1):
                file_name = node.node.metadata.get("file_name", "Unknown file")
                page_label = node.node.metadata.get("page_label", "")
                source_info = f"{i}. {file_name}"
                if page_label:
                    source_info += f" (Page {page_label})"
                if hasattr(node, "score") and node.score is not None:
                    source_info += f" - Relevance: {node.score:.3f}"
                print(source_info)
            print()

        print("📖 Answer:")
        print("-" * 30)
        print(str(response))
        print("-" * 30)


def run_chat(db_dir: Path, database_name: str, model_name: str):
    """Run an interactive chat session."""
    _, index, llm, _ = setup_query_engine(
        model_name=model_name, db_dir=db_dir, database_name=database_name
    )

    chat_engine = index.as_chat_engine(
        chat_mode="condense_question", llm=llm, verbose=False
    )

    print(f"💬 Interactive Docbot Chat Session [Model: {model_name}] (type 'exit' or 'quit' to end)")
    print("=" * 60)

    while True:
        try:
            user_msg = input("\n🤔 You: ").strip()
            if user_msg.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            if not user_msg:
                continue

            print("\n🧠 Thinking...")
            response = chat_engine.chat(user_msg)
            print(f"\n📖 Docbot:\n{response}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
