"""
Phase 2 + 3: ingest documents into the persistent vector DB, tagging each
chunk with an ACL (access group) for whitelisting at query time.

Usage:
  python rag/ingest.py --path /workspace/data/docs --acl public
  python rag/ingest.py --path /workspace/data/docs/hr --acl hr_team
"""
import os, sys, glob, argparse, uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


def read_file(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    with open(path, "r", errors="ignore") as f:
        return f.read()


def chunk_text(text, size=800, overlap=100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=config.DOCS_DIR, help="File or directory to ingest")
    ap.add_argument("--acl", default="public",
                    help="Whitelist group allowed to see these docs (e.g. public, hr_team)")
    args = ap.parse_args()

    embedder = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    coll = client.get_or_create_collection(config.COLLECTION)

    if os.path.isdir(args.path):
        paths = []
        for ext in ("*.txt", "*.md", "*.pdf"):
            paths += glob.glob(os.path.join(args.path, "**", ext), recursive=True)
    else:
        paths = [args.path]

    total = 0
    for p in paths:
        chunks = chunk_text(read_file(p))
        if not chunks:
            continue
        embeddings = embedder.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": os.path.basename(p), "acl": args.acl} for _ in chunks]
        coll.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        total += len(chunks)
        print(f"  + {p}: {len(chunks)} chunks (acl={args.acl})")

    print(f"Ingested {total} chunks into '{config.COLLECTION}' at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
