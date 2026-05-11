import os
import re
import time

import openai
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into os.environ
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR  = r"c:\Users\Nisha\Downloads\AI_Assignment\chroma_db"
COLLECTION  = "upwork_api_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL       = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are a Senior Upwork API Consultant with deep technical expertise.
You answer questions strictly based on the documentation context provided to you.
If the answer cannot be found in the provided context, respond with exactly:
"I'm sorry, but the provided documentation does not contain that information."
Do not guess, infer, or use knowledge outside the provided context."""


# ══════════════════════════════════════════════════════════════════════════════
# B1 — SEMANTIC RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def load_vectorstore() -> Chroma:
    embedding_fn = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_DIR,
    )


def retrieve(vectorstore: Chroma, query: str, k: int = 5) -> list:
    """
    Hybrid retrieval: semantic similarity + keyword matching.
    Keyword chunks are ranked by how many query terms they match,
    so the most specific matches surface first.
    """
    # 1. Semantic search
    semantic_results = vectorstore.similarity_search(query, k=k)

    # 2. Extract keywords — strip punctuation so "Upwork?" matches "upwork"
    query_keywords = [
        re.sub(r"[^\w]", "", w.lower())
        for w in query.split() if len(w) > 3
    ]
    query_keywords = [kw for kw in query_keywords if kw]

    # 3. Keyword scan — score each chunk by number of keywords matched
    all_docs = vectorstore.get()
    scored = []
    for doc_text in all_docs["documents"]:
        score = sum(1 for kw in query_keywords if kw in doc_text.lower())
        if score > 0:
            scored.append((score, doc_text))

    # Sort highest-scoring first so most relevant keyword chunks come first
    scored.sort(key=lambda x: x[0], reverse=True)

    # 4. Merge: semantic results first, then keyword matches (no duplicates)
    seen = {doc.page_content for doc in semantic_results}
    merged = list(semantic_results)
    for _, doc_text in scored:
        if doc_text not in seen:
            merged.append(Document(page_content=doc_text))
            seen.add(doc_text)

    return merged[:10]  # cap at 10 chunks for LLM context


# ══════════════════════════════════════════════════════════════════════════════
# B2 — API INTEGRATION & PROMPTING
# ══════════════════════════════════════════════════════════════════════════════

def get_client() -> openai.OpenAI:
    # Try Streamlit secrets first (cloud), fall back to .env (local)
    try:
        import streamlit as st
        api_key = st.secrets.get("DEEPINFRA_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = os.environ.get("DEEPINFRA_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY is not set in secrets or environment.")
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )


def ask(query: str, vectorstore: Chroma) -> dict:
    """
    Retrieve relevant chunks, call the LLM, and return answer + sources + latency.

    Returns:
        {
            "answer":  str,        # LLM response text
            "sources": list[str],  # raw chunk texts used as context
            "latency": float,      # API response time in seconds
        }
    """
    chunks = retrieve(vectorstore, query, k=5)
    context = "\n\n---\n\n".join(chunk.page_content for chunk in chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]

    client = get_client()

    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    latency = round(time.time() - start, 2)

    return {
        "answer":  response.choices[0].message.content,
        "sources": [chunk.page_content for chunk in chunks[:3]],
        "latency": latency,
    }
