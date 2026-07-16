Docbot!

First-time setup:
1. Place your documents in the `docs` folder.
2. Run setup script: `./setup.sh`

Usage:
1. Start ollama if not already running: `ollama run gemma3n:e2b`.
2. Start the app: `uv run app.py`.
3. Start asking questions.
4. Type `q` to quit.

Optional CLI flags:
- Rebuild a specific database: `uv run build_index.py --database product-docs`
- Use a specific docs folder: `uv run build_index.py --docs-path /path/to/docs`
- Query a specific database: `uv run app.py --database product-docs`
- Use a different model: `uv run app.py --model llama3.1:8b`

Note: This runs the model on the local machine. The quality of the outputs will be dependent on the quality of the inputs. This is a work in progress, have low expectations.
