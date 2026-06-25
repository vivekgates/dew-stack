#!/usr/bin/env bash
# One-time (or after-migration) setup. Creates a virtualenv ON the /workspace
# volume so installed packages survive Pod restarts, and prepares folders.
set -e
WORKSPACE="${WORKSPACE:-/workspace}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$WORKSPACE/data/docs" "$WORKSPACE/hf" "$WORKSPACE/chroma" "$WORKSPACE/adapters"

# --- Serving + RAG venv ---
if [ ! -d "$WORKSPACE/venv" ]; then
  python -m venv "$WORKSPACE/venv"
fi
source "$WORKSPACE/venv/bin/activate"
pip install --upgrade pip
pip install -r "$HERE/requirements.txt"
deactivate

echo ""
echo "Setup complete."
echo "  Serving/RAG venv:  source $WORKSPACE/venv/bin/activate"
echo "  Start the API:     bash $HERE/serve/start_vllm.sh"
echo ""
echo "For finetuning, build the separate training venv when you need it:"
echo "  python -m venv $WORKSPACE/venv-finetune"
echo "  source $WORKSPACE/venv-finetune/bin/activate"
echo "  pip install -r $HERE/requirements-finetune.txt"
