import sys
import os
import re

sys.stdout.reconfigure(encoding="utf-8")

# Constants
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
PDF_PATH       = os.path.join(BASE_DIR, "API Documentation Partial.pdf")
CHROMA_DIR     = os.path.join(BASE_DIR, "chroma_db")
COLLECTION     = "upwork_api_docs"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50


# A1 - DATA INGESTION

def load_and_clean_pdf(path: str) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    raw = "".join(page.get_text() for page in doc)
    doc.close()

    raw = raw.replace("’", "'")
    raw = raw.replace(" ", " ")
    raw = raw.replace("ℹ", "")
    raw = re.sub(r"Stack Overflow\s+Getting Started\s+TOS\s+FAQ\s+Changelog\s*", "", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    return raw.strip()


# A2 - DOCUMENT CHUNKING

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return splitter.create_documents(
        texts=[text],
        metadatas=[{"source": "API Documentation Partial.pdf"}],
    )


# A3 - VECTOR STORAGE

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def build_vectorstore(chunks: list) -> Chroma:
    print(f"Embedding model       : {EMBED_MODEL}")
    print("(First run downloads ~90 MB to ~/.cache/huggingface/hub)")

    embedding_fn = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print(f"Existing ChromaDB found - loading from '{CHROMA_DIR}'...")
        vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=embedding_fn,
            persist_directory=CHROMA_DIR,
        )
    else:
        print(f"Building new ChromaDB at '{CHROMA_DIR}'...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_fn,
            collection_name=COLLECTION,
            persist_directory=CHROMA_DIR,
        )

    return vectorstore


if __name__ == "__main__":
    print("=" * 60)
    print("A1: DATA INGESTION")
    print("=" * 60)

    text = load_and_clean_pdf(PDF_PATH)
    print(f"Total character count : {len(text)}")
    print(f"\nSample (first 500 chars):\n{'-'*40}")
    print(text[:500])

    print("\n" + "=" * 60)
    print("A2: DOCUMENT CHUNKING")
    print("=" * 60)

    chunks = chunk_text(text)
    print(f"Chunk size / overlap  : {CHUNK_SIZE} chars / {CHUNK_OVERLAP} chars")
    print(f"Total chunks created  : {len(chunks)}")
    print(f"\nSample chunk [index 5]:\n{'-'*40}")
    print(chunks[5].page_content)
    print(f"\nChunk length          : {len(chunks[5].page_content)} chars")

    print("\n" + "=" * 60)
    print("A3: VECTOR STORAGE")
    print("=" * 60)

    vectorstore = build_vectorstore(chunks)
    count = vectorstore._collection.count()
    print(f"Vectors stored        : {count}")
    print(f"ChromaDB location     : {os.path.abspath(CHROMA_DIR)}")
    print(f"Chunks == Vectors     : {count == len(chunks)}")

    print(f"\nSmoke-test similarity search: 'How do I authenticate with OAuth2?'")
    print("-" * 40)
    results = vectorstore.similarity_search("How do I authenticate with OAuth2?", k=3)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:\n{doc.page_content[:250]}")

    print("\n" + "=" * 60)
    print("Part A complete. ChromaDB is ready for Part B retrieval.")
    print("=" * 60)
