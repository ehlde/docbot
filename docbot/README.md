Docbot!

First-time setup:
1. Download ollama.
2. Start ollama (pulls the model if not already downloaded): `ollama run gemma3n:e2b`.
3. Place your documents in the `docs` folder.
4. Run the index builder: `uv run build_index.py`.

Usage:
1. Start ollama if not already started `ollama run gemma3n:e2b`.
2. Start the app `uv run app.py`.
3. Start asking questions!
4. Type `q` to quit.

Note: This runs the model on the local machine. The quality of the outputs will be dependent on the quality of the inputs. This is a work in progress, have low expectations.
