import os
import streamlit as st
from part_a_ingest import load_and_clean_pdf, chunk_text
from part_b_rag import ask, load_vectorstore, CHROMA_DIR, COLLECTION, EMBED_MODEL

st.set_page_config(page_title="Upwork API Support Bot", page_icon="🤖", layout="centered")

st.title("Upwork API Support Bot")
st.caption("Powered by Meta-Llama 3.1 · DeepInfra · ChromaDB")

# Build vector store if it doesn't exist (first run on Streamlit Cloud)
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_vectorstore():
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        pdf_path = os.path.join(os.path.dirname(__file__), "API Documentation Partial.pdf")
        text = load_and_clean_pdf(pdf_path)
        chunks = chunk_text(text)
        embedding_fn = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return Chroma.from_documents(
            documents=chunks,
            embedding=embedding_fn,
            collection_name=COLLECTION,
            persist_directory=CHROMA_DIR,
        )
    return load_vectorstore()

vectorstore = get_vectorstore()

# ── Query input ────────────────────────────────────────────────────────────────
query = st.text_input(
    label="Ask a question about the Upwork API",
    placeholder="e.g. How do I authenticate with OAuth2?",
)
ask_btn = st.button("Ask", type="primary", disabled=not query)

# ── Response ───────────────────────────────────────────────────────────────────
if ask_btn and query:
    with st.spinner("Thinking..."):
        try:
            result = ask(query, vectorstore)
        except ValueError as e:
            st.error(f"Configuration error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # Answer
    st.subheader("Answer")
    st.write(result["answer"])

    # Sources
    st.subheader("Sources")
    for i, snippet in enumerate(result["sources"], start=1):
        with st.expander(f"Source {i}"):
            st.text(snippet)

    # Latency
    st.caption(f"⏱ Response time: {result['latency']} seconds")
