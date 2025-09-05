import os, json, hashlib, pathlib
from typing import Dict, Any, Iterable, List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from chromadb import PersistentClient

# --- CONFIG (use NEW names/paths) ---
NORMALIZED_DIR = "mailstore/normalized"
PERSIST_DIR    = "mailstore/chroma_openai1536"   # NEW directory
COLLECTION     = "email_chunks_openai1536"       # NEW collection
OPENAI_MODEL   = "text-embedding-3-small"        # 1536 dims by default

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# --- utilities ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " "],
    length_function=len,
)
count = 0 
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
    # Prefer safe fields produced by your cleaner
    body = (rec.get("text") or rec.get("text_plain") or rec.get("body_text") or "").strip()
    if not body or looks_binaryish(body) or "JFIF" in body or "Exif" in body or "<?xpacket" in body:
        return None  # skip binary / empty

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
    base_key = (doc_meta.get("message_id")
                or f"{doc_meta.get('folder')}:{doc_meta.get('imap_uid')}")
    base = f"{base_key}|{chunk_index}"
    return "mid:" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]

# --- 1) Create NEW vector store with OpenAI embeddings (1536) ---
if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY")

# (Optional but recommended) lock dimensions explicitly to 1536
emb = OpenAIEmbeddings(model=OPENAI_MODEL, dimensions=1536)

# Verify query embedding dim
test_dim = len(emb.embed_query("ping"))
assert test_dim == 1536, f"Unexpected embed dim: {test_dim}"

vs = Chroma(
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR,   # NEW dir → fresh collection storage
    embedding_function=emb,
)

# --- 2) Index data ---
def upsert_one_email(rec: Dict[str, Any]) -> int:
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

    # Add in safe batches (Chroma 1.x can have a batch cap)
    BATCH = 2000
    for i in range(0, len(chunks), BATCH):
        vs.add_documents(chunks[i:i+BATCH], ids=ids[i:i+BATCH])

    return len(chunks)

# Example: (adjust to your files)
jsonl_files = [
    pathlib.Path(NORMALIZED_DIR, "merged.clean.labeled.jsonl"),
    # pathlib.Path(NORMALIZED_DIR, "Sent Messages.jsonl"),
]
added_total = 0
for path in jsonl_files:
    if path.exists():
        for idx, rec in enumerate(iter_jsonl(str(path)), start=1):
            # print(f'Processing {idx}')
            added_total += upsert_one_email(rec)
            # print(f'finsihed {}')
            if idx % 200 == 0:
                print(f"{path.name}: {idx} emails processed")
    else:
        print(f"Missing file: {path}")

# client = PersistentClient(path=PERSIST_DIR)
# vs = Chroma(
#     client=client,
#     collection_name='email_chunks_openai1536',
#     embedding_function=emb,
# )

print(f"Indexed chunks: {added_total}")

# # --- 3) Assert index dimension really is 1536 ---
# peek = vs._collection.get(limit=1, include=['embeddings'])
# if peek['embeddings']:
#     idx_dim = len(peek['embeddings'][0])
#     assert idx_dim == 1536, f"Index dim mismatch: {idx_dim} (expected 1536)"
#     print("OK: index dimension is 1536.")
# else:
#     print("No embeddings retrieved in peek (did you add any docs?).")
