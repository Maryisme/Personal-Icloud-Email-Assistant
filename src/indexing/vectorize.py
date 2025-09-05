"""
Vectorization pipeline using LangChain + Chroma with OpenAI embeddings (1536 dims).
Reads normalized/cleaned email JSONL, converts to Documents, chunks, and upserts.
"""
from __future__ import annotations
import os, json, hashlib, pathlib
from typing import Dict, Any, Iterable, List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

NORMALIZED_DIR = os.getenv("NORMALIZED_DIR", "mailstore/normalized")
PERSIST_DIR    = os.getenv("CHROMA_DIR", "mailstore/chroma_openai1536")
COLLECTION     = os.getenv("CHROMA_COLLECTION", "email_chunks_openai1536")
OPENAI_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

load_dotenv(override=True)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " "],
    length_function=len,
)

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def looks_binaryish(s: str) -> bool:
    if not s: return False
    if "\x00" in s: return True
    ctrl = sum(1 for ch in s if ord(ch) < 32 and ch not in ("\n","\r","\t"))
    return (ctrl > 64) or (ctrl / max(len(s),1) > 0.01)

def make_document(rec: Dict[str, Any]) -> Document | None:
    body = (rec.get("text") or rec.get("text_plain") or rec.get("body_text") or "").strip()
    if not body or looks_binaryish(body) or "JFIF" in body or "Exif" in body or "<?xpacket" in body:
        return None
    meta = {
        "message_id": rec.get("message_id"),
        "folder": rec.get("folder"),
        "imap_uid": rec.get("imap_uid"),
        "subject": rec.get("subject"),
        "from_addr": (rec.get("from") or {}).get("email"),
        "date_utc": rec.get("date_utc"),
        "raw_path": rec.get("raw_path"),
        "has_attachments": bool(rec.get("attachments")),
    }
    return Document(page_content=body, metadata=meta)

def chunk_one_document(doc: Document) -> List[Document]:
    return splitter.split_documents([doc])

def stable_id_for_chunk(doc_meta: Dict[str, Any], chunk_index: int) -> str:
    base_key = (doc_meta.get("message_id") or f"{doc_meta.get('folder')}:{doc_meta.get('imap_uid')}")
    base = f"{base_key}|{chunk_index}"
    return "mid:" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]

def build_vectorstore():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for embeddings")
    emb = OpenAIEmbeddings(model=OPENAI_MODEL, dimensions=1536)
    vs = Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR, embedding_function=emb)
    return vs

def upsert_one_email(rec: Dict[str, Any], vs) -> int:
    doc = make_document(rec)
    if not doc:
        return 0
    chunks = chunk_one_document(doc)
    if not chunks:
        return 0
    ids = []
    for i, ch in enumerate(chunks):
        cid = stable_id_for_chunk(doc.metadata, i)
        ch.metadata = dict(ch.metadata)
        ch.metadata["chunk_index"] = i
        ch.metadata["id"] = cid
        ids.append(cid)
    BATCH = 2000
    for i in range(0, len(chunks), BATCH):
        vs.add_documents(chunks[i:i+BATCH], ids=ids[i:i+BATCH])
    return len(chunks)

def index_files(jsonl_files: List[str]) -> int:
    vs = build_vectorstore()
    added_total = 0
    for path in jsonl_files:
        p = pathlib.Path(path)
        if p.exists():
            for idx, rec in enumerate(iter_jsonl(str(p)), start=1):
                added_total += upsert_one_email(rec, vs)
                if idx % 200 == 0:
                    print(f"{p.name}: {idx} emails processed")
        else:
            print(f"Missing file: {p}")
    print(f"Indexed chunks: {added_total}")
    return added_total

if __name__ == "__main__":
    files = [
        "mailstore/normalized/merged.clean.labeled.jsonl",
    ]
    index_files(files)
