#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-gemma3n:e2b}"
DATABASE="${2:-support_docs}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/."
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl --silent --fail http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama server..."
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  sleep 3
fi

echo "Pulling model: ${MODEL}"
ollama pull "${MODEL}"

echo "Building index for database: ${DATABASE}"
uv run build_index.py --database "${DATABASE}"

echo "Setup complete."
