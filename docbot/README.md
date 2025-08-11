Docbot!

First-time setup:
1. Download ollama.
2. Pull the desired model if not already downloaded: `ollama pull <MODEL>`.
3. Place your documents in the `docs` folder.
4. Run the index builder: `uv run build_index.py`.

Usage:
1. Start the app `uv run app.py`.
2. Start asking questions!
3. Type `quit` or `exit` to quit.

Note: This runs the model on the local machine. The quality of the outputs will be dependent on the quality of the inputs. This is a work in progress, have low expectations.

Notes:
- The main models used in development have been:
  - `"BAAI/bge-small-en-v1.5"` for embedding.
  - `gemma3n:e2b` for queries.