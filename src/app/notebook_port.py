"""
RAG helper utilities (refactored from the original notebook code).

What this module gives you:
  - quick JSONL sanity check for your normalized email dump
  - helpers to open your persisted Chroma collection
  - a tiny 2D Plotly visualization of embeddings with t-SNE (for eyeballing)
  - a ready-to-use ConversationalRetrievalChain wired to Ollama (local LLM)

Notes:
  * This keeps your data local. Embeddings come from OpenAI; generation is local (Ollama).
  * t-SNE on all points can be slow—use `sample_n` to downsample when plotting.
"""

# -------------------------
# Imports (kept tidy)
# -------------------------
import json
from typing import Optional, List

import numpy as np
from dotenv import load_dotenv
from sklearn.manifold import TSNE
import plotly.graph_objects as go

import chromadb
from chromadb import PersistentClient

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# -------------------------
# Configuration
# -------------------------
# Paths/IDs: adjust if your layout differs
NORMALIZED_JSONL = "mailstore/normalized/merged.clean.jsonl"
PERSIST_DIR      = "mailstore/chroma_openai1536"
COLLECTION_NAME  = "email_chunks_openai1536"
OPENAI_MODEL     = "text-embedding-3-small"   # 1536-D

# -------------------------
# Utilities
# -------------------------
def validate_jsonl(path: str = NORMALIZED_JSONL, *, max_errors: int = 10) -> None:
    """Quick pass over a JSONL file to catch broken lines early.
    Prints the first few offending lines (if any) and stops.
    """
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.rstrip("\n")
            if not s.strip():
                continue
            try:
                json.loads(s)
            except Exception as e:
                bad += 1
                print(f"[BAD] {path}:{i} -> {e}")
                print(s[:500])
                if bad >= max_errors:
                    print("...more errors omitted...")
                    break
    if bad == 0:
        print(f"[OK] {path} looks fine.")

def open_chroma_vectorstore(
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    openai_embed_model: str = OPENAI_MODEL,
) -> Chroma:
    """Connect to an existing Chroma collection using LangChain's Chroma wrapper.
    We use PersistentClient under the hood so nothing is re-ingested here.
    """
    load_dotenv()
    client = PersistentClient(path=persist_dir)
    embeddings = OpenAIEmbeddings(model=openai_embed_model)  # fine even if we don't add docs
    vs = Chroma(client=client, collection_name=collection_name, embedding_function=embeddings)
    # Sanity check—helpful when you're not sure you targeted the right DB path
    try:
        count = vs._collection.count()
        print(f"[Chroma] Connected to '{collection_name}' @ '{persist_dir}' with {count} chunks.")
    except Exception:
        print(f"[Chroma] Connected to '{collection_name}' @ '{persist_dir}'.")
    return vs

def get_raw_collection(persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
    """Return the underlying ChromaDB collection to access raw embeddings/documents.
    Use this for visualization-heavy tasks that need the raw vectors.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(collection_name)

def tsne_plot_from_collection(collection, *, sample_n: Optional[int] = 2000) -> go.Figure:
    """Build a 2D t-SNE scatter of embeddings in a Chroma collection.

    Parameters
    ----------
    collection : chromadb.api.models.Collection.Collection
        The raw Chroma collection (NOT the LangChain wrapper).
    sample_n : int or None
        If set, randomly downsample to at most `sample_n` points to keep it responsive.

    Returns
    -------
    plotly.graph_objects.Figure
        A simple 2D scatter—hover shows a preview of each chunk's text.
    """
    print("[t-SNE] Fetching embeddings from collection ...")
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.array(result["embeddings"])
    documents = result["documents"]
    n = len(documents)

    if n == 0:
        raise RuntimeError("Collection has no documents/embeddings to visualize.")

    # Downsample for speed if requested
    if sample_n is not None and n > sample_n:
        idx = np.random.choice(n, size=sample_n, replace=False)
        vectors = vectors[idx]
        documents = [documents[i] for i in idx]

    print("[t-SNE] Computing 2D projection (this can take a minute on large sets) ...")
    tsne = TSNE(n_components=2, random_state=42, init="random", learning_rate="auto")
    reduced = tsne.fit_transform(vectors)

    fig = go.Figure(data=[go.Scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        mode="markers",
        marker=dict(size=5, opacity=0.8),
        text=[f"Text: {d[:100]}..." for d in documents],
        hoverinfo="text",
    )])
    fig.update_layout(
        title="2D Chroma Vector Store Visualization (t-SNE)",
        xaxis_title="x", yaxis_title="y",
        width=900, height=650, margin=dict(r=20, b=10, l=10, t=40),
    )
    return fig

def build_conversational_chain(
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    openai_embed_model: str = OPENAI_MODEL,
    *, k: int = 25,
    ollama_model: str = "llama3",
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.2,
) -> ConversationalRetrievalChain:
    """Create a ConversationalRetrievalChain that:
       - retrieves from your persisted Chroma collection
       - generates with a local Ollama model
       - remembers chat history in-memory
    """
    vs = open_chroma_vectorstore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        openai_embed_model=openai_embed_model,
    )
    
    retriever = vs.as_retriever(search_kwargs={"k": k})
    print(retriever)

    # Keep the prompt tight and retrieval-grounded.
    qa_prompt = PromptTemplate.from_template(
        """You are an email assistant. Answer using ONLY the provided context.
        If the answer is not in the context, respond with "I don't know from the given emails."

        Context:
        {context}

        Question: {question}
        Answer concisely:
        """
    )

    llm = ChatOllama(model=ollama_model, base_url=ollama_url, temperature=temperature)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return chain

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":

    # Build the conversational chain and test a query
    chain = build_conversational_chain()
    q = "When did I travel to London?"
    result = chain.invoke({"question": q})
    print("\nAnswer:", result.get("answer"))
    for i, d in enumerate(result.get("source_documents", []), 1):
        meta = d.metadata
        print(f"[{i}] {meta} :: {d.page_content[:160]!r}")
