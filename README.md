# Mistral Stack on RunPod (persistent)

A warm Mistral-7B API, RAG with whitelisting, and QLoRA finetuning — all
storing their data on the `/workspace` network volume so **nothing is lost**
when the Pod restarts or you migrate to a new machine.

## Why a Pod, not Serverless

Serverless kept failing with "no machine available" and cold-start downloads.
A persistent **on-demand Pod** holds the model in GPU memory, runs the vector
DB, and can finetune — no cold starts, no queueing for capacity.

## What persists (and what doesn't)

Everything important lives on the `/workspace` volume:

| Path | Contents |
|------|----------|
| `/workspace/hf` | model + embedding weights (downloaded once) |
| `/workspace/chroma` | vector database |
| `/workspace/data/docs` | your source documents |
| `/workspace/adapters` | finetuned LoRA adapters |
| `/workspace/venv` | Python packages (so installs survive restarts) |

The OS image and GPU are ephemeral. The code in this repo plus the volume is
everything you need; if you migrate, attach the same volume and re-run nothing
but `setup.sh` (which rebuilds the venv in ~2 min if the base image changed).

## Machine

Deploy a **Secure Cloud Pod**, GPU **A40 48GB (~$0.44/hr)** or **RTX A6000
48GB (~$0.49/hr)**. Attach a **100 GB network volume** (mounts at
`/workspace`). Use a PyTorch base template and expose **HTTP port 8000**.
Finetuning fits on the same 48 GB card with QLoRA; for faster runs use an
**A100 80GB (~$1.39/hr)** only while training, then stop it.

## First-time setup

```bash
# clone or copy this folder onto the Pod, into /workspace
cd /workspace/mistral-stack
bash setup.sh
```

## Phase 1 — Mistral with a fast API

```bash
bash serve/start_vllm.sh
```

Gives an OpenAI-compatible endpoint on port 8000. Test it (via the Pod proxy
URL RunPod shows for port 8000, or localhost on the Pod):

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistralai/Mistral-7B-Instruct-v0.3",
       "messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Phase 2 — RAG

Put documents (`.txt`, `.md`, `.pdf`) in `/workspace/data/docs`, then:

```bash
source /workspace/venv/bin/activate
python rag/ingest.py --path /workspace/data/docs --acl public
python rag/query.py "What is our refund policy?"
```

`ingest.py` chunks, embeds (bge-small), and stores vectors in Chroma on the
volume. `query.py` retrieves the most relevant chunks and asks Mistral using
only that context.

## Phase 3 — Search with whitelisting

Whitelisting = restricting retrieval to an allowed set. Every chunk is tagged
with an `acl` group at ingest; queries only see chunks whose `acl` is in the
caller's allowed list.

```bash
# Ingest sensitive docs under a restricted group
python rag/ingest.py --path /workspace/data/docs/hr --acl hr_team

# A public user can't see hr_team content
python rag/query.py "salary bands?"                       # only public docs
# An HR user can
python rag/query.py "salary bands?" --acl hr_team --acl public
```

To whitelist **web** sources instead, keep an allowed-domains list and filter
search results before feeding them to RAG (same metadata-filter idea).

## Phase 4 — Finetune on your data

Use a separate venv (training libs clash with vLLM):

```bash
python -m venv /workspace/venv-finetune
source /workspace/venv-finetune/bin/activate
pip install -r requirements-finetune.txt

python finetune/train_qlora.py --data /workspace/data/train.jsonl --name mistral-lora
```

Training data is JSONL in chat format (see `finetune/data_example.jsonl`).
The adapter lands in `/workspace/adapters/mistral-lora`. Serve it on top of the
base model by adding to `serve/start_vllm.sh`:

```
  --enable-lora --lora-modules mistral-lora=/workspace/adapters/mistral-lora
```

Then call it with `"model": "mistral-lora"` in your API request.

## Notes

- vLLM bundles the tokenizer and chat template, so the sentencepiece/Xet
  issues from the serverless build don't apply here.
- Keep the Pod stopped when idle to avoid paying for unused GPU hours; your
  data on `/workspace` stays intact while stopped.
- TRL changes APIs across versions. If `processing_class=` errors in
  `train_qlora.py`, change it to `tokenizer=tokenizer`.
