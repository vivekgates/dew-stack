#!/usr/bin/env bash
# Installs a systemd service so vLLM always runs and restarts on crash/reboot.
set -e
WORKSPACE="${WORKSPACE:-/workspace}"
HERE="$(cd "$(dirname "$0")" && pwd)"

cat > /etc/systemd/system/vllm.service <<EOF
[Unit]
Description=vLLM Mistral API server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${WORKSPACE}/dew-stack
Environment=HF_HOME=${WORKSPACE}/hf
Environment=HF_HUB_DISABLE_XET=1
Environment=HF_HUB_ENABLE_HF_TRANSFER=0
ExecStart=${WORKSPACE}/venv/bin/vllm serve mistralai/Mistral-7B-Instruct-v0.3 \\
  --host 0.0.0.0 --port 8000 --max-model-len 8192 --download-dir ${WORKSPACE}/hf
Restart=always
RestartSec=5
# Keep trying even if the first boots fail while the GPU warms up
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable vllm
systemctl restart vllm

echo "vllm service installed and started."
echo "  status: systemctl status vllm"
echo "  logs:   journalctl -u vllm -f"
