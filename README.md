# Email RAG with LangChain, Chroma, OpenAI Embeddings, and Local Qwen3 (Ollama)

This repo contains a small end‑to‑end Retrieval‑Augmented Generation (RAG) project over a private email corpus.
It indexes cleaned email messages into a Chroma vector store using OpenAI's `text-embedding-3-small` (1536‑D),
and runs a local **Qwen3** model via **Ollama** for generation to keep sensitive data local.

## What this does
- **Clean & normalize** raw email JSONL into readable `text_plain` while preserving essential metadata.
- **Optionally label spam** locally (Qwen) and filter or tag during indexing.
- **Chunk & index** messages into Chroma with stable chunk IDs.
- **Query via Gradio UI**, retrieving top‑K chunks and answering with context‑bound generation.

## Data cleaning performed
The cleaner:
- Decodes RFC‑2047 encoded headers (subjects/names), normalizes Unicode to NFC, and unescapes HTML entities.
- Removes zero‑width and other control characters, normalizing `\r\n`/`\r` to `\n`.
- Converts HTML bodies to plain text with a robust parser (falls back to regex when BS4/LXML unavailable).
- Chooses a safe `text_plain` from `body_text` when not binary, or from `body_html` converted to text; keeps optionally the raw bodies for debugging.
- Leaves well‑formed address lists (`from`, `to`, `cc`, `bcc`) in decoded, structured form.
See `src/cleaning/clean_from_sources.py` for details.

## Repo layout
```text
src/
  app/
    notebook_port.py          # code migrated from the original notebook (code cells only, magics stripped)
  cleaning/
    clean_from_sources.py     # robust email cleaner
  indexing/
    vectorize.py              # builds Chroma with OpenAI embeddings
  classification/
    spam_classifier.py        # labels emails via local Qwen (Ollama)
gradio_app.py                 # simple RAG UI (retrieval + local generation)
config/
  .env.example                # environment variables template
mailstore/                    # local data (git‑ignored)
legacy/                       # original uploaded .py files (for reference)
requirements.txt
.gitignore
README.md
```

## Quickstart

1) **Environment**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config/.env.example .env  # set your keys/URLs
```

2) **Prepare data** (place your JSONL into `mailstore/normalized/`)
```bash
python -m src.cleaning.clean_from_sources  # optional: cleans common inputs to *.clean.jsonl
```

3) **(Optional) Label spam locally**
```bash
python -m src.classification.spam_classifier
```

4) **Index with embeddings**
```bash
python -m src.indexing.vectorize
```

5) **Run the RAG UI**
```bash
python gradio_app.py
```

## Configuration

- `OPENAI_API_KEY`: required for embeddings.
- `CHROMA_DIR`, `CHROMA_COLLECTION`: embeddings storage, the collection name.
- `NORMALIZED_DIR`: cleaned JSONL with data.
- `OLLAMA_URL`, `OLLAMA_MODEL`: local generation endpoint/model.

## Privacy & security
- The email content remains local. Generation is performed with a locally hosted Qwen model via Ollama.
- The `mailstore/` directory is **git‑ignored** and never uploaded.

## GitHub setup

```bash
git init
git add .
git commit -m "email RAG: cleaning, labeling, indexing, and Gradio UI"
git branch -M main
git remote add origin git@github.com:<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```
