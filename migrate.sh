#!/usr/bin/env bash
# =====================================================================
# ONE-COMMAND MIGRATION / SETUP for the dew-stack Qwen API.
#   bash /workspace/dew-stack/migrate.sh
# Safe to re-run. Rebuilds venv, installs supervisor, runs vLLM service.
# Data on /workspace (hf, chroma, data, adapters) is never touched.
# =====================================================================
set -e
WORKSPACE="${WORKSPACE:-/workspace}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="Qwen/Qwen2.5-7B-Instruct"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

echo ">>> [1/5] Folders on the persistent volume"
mkdir -p "$WORKSPACE/data/docs" "$WORKSPACE/hf" "$WORKSPACE/chroma" "$WORKSPACE/adapters"

echo ">>> [2/5] Rebuilding the venv"
rm -rf "$WORKSPACE/venv"
python -m venv "$WORKSPACE/venv"
source "$WORKSPACE/venv/bin/activate"
pip install --upgrade pip

# torch MUST come from the PyTorch index (the +cu124 build is not on PyPI).
echo "    installing torch from $TORCH_INDEX"
pip install torch==2.5.1 --index-url "$TORCH_INDEX"

if [ -f "$HERE/requirements-lock.txt" ]; then
  echo "    installing the rest of requirements-lock.txt (torch already done)"
  grep -v '^torch==' "$HERE/requirements-lock.txt" > /tmp/req-notorch.txt
  # Use PyPI as primary, PyTorch index as fallback for any cuda wheels.
  pip install -r /tmp/req-notorch.txt --extra-index-url "$TORCH_INDEX"
else
  echo "    lockfile missing -> installing known-good pins"
  pip install vllm==0.6.6.post1 "transformers==4.46.3" hf_transfer
  pip install openai chromadb sentence-transformers pypdf
fi
deactivate

echo ">>> [3/5] Installing supervisor"
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
echo " Done. Serving ${MODEL}."
echo "   status: supervisorctl status vllm"
echo "   logs:   tail -f ${WORKSPACE}/vllm.log"
echo "   test:   curl http://localhost:8000/v1/models"
echo "=========================================================="
