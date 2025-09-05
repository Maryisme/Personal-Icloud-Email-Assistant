########################################################################################
# Code migrated from 'Personal Assistant.ipynb' (code cells only)
########################################################################################


# --- cell ---
import os, json, hashlib, time
from typing import Iterator, List, Dict, Any
from typing import Iterable, Dict, Any, List

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

NORMALIZED_DIR = "mailstore/normalized/merged.clean.jsonl"
PERSIST_DIR    = "mailstore/chroma_openai1536"
COLLECTION     = "email_chunks_openai1536"
OPENAI_MODEL   = "text-embedding-3-small"



# --- cell ---
import json

path = "mailstore/normalized/merged.clean.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        s = line.rstrip("\n")
        if not s.strip():
            continue
        try:
            json.loads(s)
        except Exception as e:
            print(f"[BAD] {path}:{i} -> {e}")
            print(s[:500])  # peek
            break

# --- cell ---
load_dotenv()

# --- cell ---
client = chromadb.PersistentClient(path=PERSIST_DIR)
# collection = client.get_or_create_collection("collection_name")

# --- cell ---
collection = client.get_collection('email_chunks_openai1536')

# --- cell ---
import numpy as np
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
# doc_types = [metadata['doc_type'] for metadata in result['metadatas']]
# colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]

# --- cell ---
# We humans find it easier to visalize things in 2D!
# Reduce the dimensionality of the vectors to 2D using t-SNE
# (t-distributed stochastic neighbor embedding)
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, opacity=0.8),
    text=[f"Text: {d[:100]}..." for d in documents],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()

# --- cell ---
vectorstore

# --- cell ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from chromadb import PersistentClient

client = PersistentClient(path=PERSIST_DIR)
embeddings = OpenAIEmbeddings(model=OPENAI_MODEL)  # ok even if you’re not adding docs
vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION,
    embedding_function=embeddings,
)
# Now you can get a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

# Ollama model
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434",
    temperature=0.2,
)

# System prompt
system_prompt = "You are an email assistant."
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
model = prompt | llm

from langchain_core.callbacks import StdOutCallbackHandler
# Conversation memory
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# Build ConversationalRetrievalChain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=model,           # use your wrapped model (with system prompt)
    retriever=retriever,
    memory=memory
)


conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

query = "When did I travel to London?"
result = conversation_chain.invoke({"question": query})
answer = result["answer"]
print("\nAnswer:", answer)

# --- cell ---
from chromadb import PersistentClient
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

PERSIST_DIR = "mailstore/chroma_openai1536"
COLLECTION  = "email_chunks_openai1536"
OPENAI_MODEL = "text-embedding-3-small"

# 1) Open the persisted Chroma collection CORRECTLY
client = PersistentClient(path=PERSIST_DIR)
embeddings = OpenAIEmbeddings(model=OPENAI_MODEL)  # ok even if you’re not adding docs
vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION,
    embedding_function=embeddings,
)

print("Doc count in collection:", vectorstore._collection.count())  # sanity check

# 2) Build a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

# 3) LLM (Ollama)
llm = ChatOllama(model="llama3", base_url="http://localhost:11434", temperature=0.2)

# 4) Optional: custom system prompt for QA
from langchain.prompts import PromptTemplate
qa_prompt = PromptTemplate.from_template(
    "You are an email assistant. Use ONLY the context.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer concisely:"
)

# 5) Build the ConversationalRetrievalChain and RETURN sources
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="question",   # <- tell memory what your input field is
    output_key="answer",    # <- tell memory which output to store
)

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
)

# 6) Test a query
result = conversation_chain.invoke({"question": "check my emails and tell me when I got a trip to London booked"})
print("\nAnswer:", result["answer"])
for i, d in enumerate(result.get("source_documents", []), 1):
    print(f"[{i}] {d.metadata} :: {d.page_content[:200]!r}")

# --- cell ---
query = "be more specific and explicit"
result = conversation_chain.invoke({"question": query})
print(result["answer"])
