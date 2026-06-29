#!/usr/bin/env bash
# =====================================================================
# ONE-COMMAND MIGRATION / SETUP for the dew-stack Qwen API.
#
# Run on any fresh or migrated RunPod Pod AFTER:
#   1. the /workspace network volume is attached
#   2. HTTP port 8000 is exposed on the Pod
#
#   bash /workspace/dew-stack/migrate.sh
#
# Safe to re-run. Rebuilds the venv, installs an always-on vLLM service
# (supervisor, since RunPod containers have no systemd), and starts it.
# Your model weights / vector DB / docs on /workspace are never touched.
# =====================================================================
set -e
WORKSPACE="${WORKSPACE:-/workspace}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo ">>> [1/5] Folders on the persistent volume"
mkdir -p "$WORKSPACE/data/docs" "$WORKSPACE/hf" "$WORKSPACE/chroma" "$WORKSPACE/adapters"

echo ">>> [2/5] Rebuilding the venv (tied to the OS image; never reuse an old one)"
rm -rf "$WORKSPACE/venv"
python -m venv "$WORKSPACE/venv"
source "$WORKSPACE/venv/bin/activate"
pip install --upgrade pip
if [ -f "$HERE/requirements-lock.txt" ]; then
  echo "    installing exact pinned versions from requirements-lock.txt"
  pip install -r "$HERE/requirements-lock.txt"
else
  echo "    lockfile missing -> installing known-good pins"
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  pip install vllm==0.6.6.post1 "transformers==4.46.3" hf_transfer
  pip install openai chromadb sentence-transformers pypdf
fi
deactivate

echo ">>> [3/5] Installing supervisor (process manager that keeps vLLM alive)"
if ! command -v supervisord >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq supervisor
fi

echo ">>> [4/5] Writing the vLLM supervisor service"
mkdir -p /etc/supervisor/conf.d
cat > /etc/supervisor/conf.d/vllm.conf <<EOF
[program:vllm]
command=${WORKSPACE}/venv/bin/vllm serve ${MODEL} --host 0.0.0.0 --port 8000 --max-model-len 8192 --download-dir ${WORKSPACE}/hf
directory=${WORKSPACE}/dew-stack
environment=HF_HOME="${WORKSPACE}/hf",HF_HUB_DISABLE_XET="1",HF_HUB_ENABLE_HF_TRANSFER="0"
autostart=true
autorestart=true
startretries=999
stopwaitsecs=30
stdout_logfile=${WORKSPACE}/vllm.log
stderr_logfile=${WORKSPACE}/vllm.log
EOF

echo ">>> [5/5] Starting the service"
if ! pgrep -x supervisord >/dev/null 2>&1; then
  supervisord -c /etc/supervisor/supervisord.conf
fi
supervisorctl reread
supervisorctl update
supervisorctl restart vllm || supervisorctl start vllm

echo ""
echo "=========================================================="
echo " Done. vLLM is serving ${MODEL} and auto-restarts on crash."
echo "   status:  supervisorctl status vllm"
echo "   logs:    tail -f ${WORKSPACE}/vllm.log"
echo "   local:   curl http://localhost:8000/v1/models"
echo " First start downloads the model only if it isn't cached yet."
echo "=========================================================="
