"""
Phase 2 + 3: answer a question using only whitelisted documents.

Retrieval is restricted to chunks whose ACL is in the caller's allowed list,
so a user only ever sees content they are permitted to (whitelisting).

Usage:
  python rag/query.py "What is our refund policy?"
  python rag/query.py "Salary bands?" --acl hr_team --acl public
"""
import os, sys, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def retrieve(question, allowed_acls, k=4):
    embedder = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    coll = client.get_or_create_collection(config.COLLECTION)
    q_emb = embedder.encode([question]).tolist()

    # WHITELISTING: only consider chunks whose acl is in the allowed list.
    where = {"acl": {"$in": allowed_acls}} if allowed_acls else None
    res = coll.query(query_embeddings=q_emb, n_results=k, where=where)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def answer(question, allowed_acls, k=4):
    hits = retrieve(question, allowed_acls, k)
    if not hits:
        return "No documents within your access scope can answer this."

    context = "\n\n".join(f"[{m.get('source')}] {d}" for d, m in hits)
    prompt = (
        "Use ONLY the context below to answer. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    client = OpenAI(base_url=config.VLLM_BASE_URL, api_key="not-needed")
    resp = client.chat.completions.create(
        model=config.BASE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--acl", action="append", default=None,
                    help="Allowed whitelist group(s); repeat to allow several. Default: public")
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    acls = args.acl or ["public"]
    print(answer(args.question, acls, args.k))
