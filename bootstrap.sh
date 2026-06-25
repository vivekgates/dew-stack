#!/usr/bin/env bash
# =====================================================================
# ONE-COMMAND SETUP after a fresh Pod or migration.
# Assumes: the /workspace network volume is attached, and this repo is
# cloned at /workspace/dew-stack. Re-running it is safe (idempotent).
#
#   bash /workspace/dew-stack/bootstrap.sh
# =====================================================================
set -e
WORKSPACE="${WORKSPACE:-/workspace}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo ">>> [1/4] Creating folders on the persistent volume"
mkdir -p "$WORKSPACE/data/docs" "$WORKSPACE/hf" "$WORKSPACE/chroma" "$WORKSPACE/adapters"

echo ">>> [2/4] Building the serving/RAG venv from the locked versions"
# Rebuild the venv from scratch (it is tied to the OS image, so never reuse
# an old one after migration). Model weights in $WORKSPACE/hf are NOT touched.
rm -rf "$WORKSPACE/venv"
python -m venv "$WORKSPACE/venv"
source "$WORKSPACE/venv/bin/activate"
pip install --upgrade pip

if [ -f "$HERE/requirements-lock.txt" ]; then
  echo "    using requirements-lock.txt (exact pinned versions)"
  pip install -r "$HERE/requirements-lock.txt"
else
  echo "    lockfile missing, falling back to known-good pins"
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  pip install vllm==0.6.6.post1 "transformers==4.46.3"
  pip install openai chromadb sentence-transformers pypdf
fi
deactivate

echo ">>> [3/4] Installing the vLLM systemd service"
bash "$HERE/install_service.sh"

echo ">>> [4/4] Done."
echo "    The 'vllm' service is enabled and will auto-start on every boot."
echo "    Check status:  systemctl status vllm"
echo "    Live logs:     journalctl -u vllm -f"
echo "    API base URL:  http://localhost:8000/v1   (and the Pod's port-8000 proxy URL)"
