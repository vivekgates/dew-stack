#!/usr/bin/env bash
# Phase 1: serve Mistral as a warm, OpenAI-compatible API on port 8000.
# The model downloads ONCE into /workspace/hf and is reused forever after.
set -e
WORKSPACE="${WORKSPACE:-/workspace}"
source "$WORKSPACE/venv/bin/activate"

export HF_HOME="$WORKSPACE/hf"
export HF_HUB_DISABLE_XET=1

# To serve a finetuned LoRA adapter as well, add:
#   --enable-lora --lora-modules myadapter=/workspace/adapters/myadapter
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --download-dir "$WORKSPACE/hf"
