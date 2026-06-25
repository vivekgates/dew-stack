"""
Central configuration. EVERY path lives under /workspace (the network volume)
so nothing is lost when the Pod restarts or you migrate to a new machine.
"""
import os

# Root of the persistent network volume (mounted at /workspace on RunPod Pods).
WORKSPACE = os.environ.get("WORKSPACE", "/workspace")

# Persistent locations
HF_HOME      = os.path.join(WORKSPACE, "hf")          # model + embedding cache
DATA_DIR     = os.path.join(WORKSPACE, "data")
DOCS_DIR     = os.path.join(DATA_DIR, "docs")          # put your documents here
CHROMA_DIR   = os.path.join(WORKSPACE, "chroma")       # vector DB on disk
ADAPTERS_DIR = os.path.join(WORKSPACE, "adapters")     # finetuned LoRA adapters

# Send the HuggingFace cache to the volume and disable the flaky Xet backend.
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Models
BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Local vLLM OpenAI-compatible endpoint (served on the same Pod)
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

COLLECTION = "docs"

# Make sure the persistent directories exist
for _d in (HF_HOME, DOCS_DIR, CHROMA_DIR, ADAPTERS_DIR):
    os.makedirs(_d, exist_ok=True)
