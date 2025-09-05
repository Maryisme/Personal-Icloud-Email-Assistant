"""
Gradio RAG UI for querying your email vector store.
- Retrieves top-k chunks from Chroma.
- Uses local Ollama (Qwen) to generate an answer with citations.
"""
from __future__ import annotations
import os, json
import gradio as gr
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import requests

CHROMA_DIR = os.getenv("CHROMA_DIR", "mailstore/chroma_openai1536")
COLLECTION = os.getenv("CHROMA_COLLECTION", "email_chunks_openai1536")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OPENAI_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def load_vs():
    emb = OpenAIEmbeddings(model=OPENAI_MODEL, dimensions=1536)
    return Chroma(collection_name=COLLECTION, persist_directory=CHROMA_DIR, embedding_function=emb)

def answer(query: str, k: int = 5) -> str:
    vs = load_vs()
    docs = vs.similarity_search(query, k=k)
    context = "\n\n".join([f"[{i}] {d.page_content}" for i,d in enumerate(docs, start=1)])
    meta = [d.metadata for d in docs]

    prompt = (
        "You are an assistant that answers using ONLY the provided context.\n"
        "If the answer isn't present, say "I don't know from the given emails."\n"
        "Return a concise answer and list the citation indices you used.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
    )
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    ans = r.json().get("response", "").strip()
    return ans + "\n\nCitations metadata:\n" + json.dumps(meta, ensure_ascii=False, indent=2)

with gr.Blocks(title="Email RAG (Local Qwen)") as demo:
    gr.Markdown("# Email RAG (Local Qwen)")
    gr.Markdown("Query your email vector store. Embeddings are OpenAI; generation is local Qwen via Ollama.")
    inp = gr.Textbox(label="Query")
    topk = gr.Slider(1, 10, value=5, step=1, label="Top-K")
    out = gr.Textbox(label="Answer", lines=12)
    btn = gr.Button("Search")
    btn.click(fn=answer, inputs=[inp, topk], outputs=out)
if __name__ == "__main__":
    demo.launch()
