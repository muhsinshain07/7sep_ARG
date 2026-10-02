import os
from io import BytesIO

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 5


# -----------------------------
# Load models / clients
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(pdf_bytes: bytes):
    """Extract text from every PDF page and keep page numbers."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# -----------------------------
# Chunking / tokenization
# -----------------------------
def create_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Create overlapping word-based chunks.

    This is a simple tokenizer/chunker for a beginner-friendly RAG app.
    Each chunk keeps its original PDF page number.
    """
    chunks = []

    for page_data in pages:
        words = page_data["text"].split()

        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_data["page"],
                    }
                )

            if end >= len(words):
                break

            start = end - overlap

    return chunks


# -----------------------------
# Create FAISS vector database
# -----------------------------
def build_faiss_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# -----------------------------
# Retrieve relevant chunks
# -----------------------------
def retrieve_chunks(question, index, chunks, embedding_model, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            }
        )

    return results


# -----------------------------
# Generate answer with Groq
# -----------------------------
def generate_answer(question, retrieved_chunks, client):
    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the provided document context.

Rules:
1. Do not invent facts that are not supported by the context.
2. If the answer is not present in the context, clearly say:
   "I couldn't find the answer in the uploaded document."
3. Keep the answer clear and useful.
4. Mention relevant PDF page numbers when possible.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer based only on the document context.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload a PDF, build a FAISS vector index, and ask questions about the document."
)

with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=2,
        max_value=10,
        value=TOP_K,
    )

    st.info(
        "Embeddings: all-MiniLM-L6-v2\n\n"
        "Vector DB: FAISS\n\n"
        f"LLM: {GROQ_MODEL}"
    )

uploaded_file = st.file_uploader(
    "📄 Upload a PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        st.session_state.file_name = uploaded_file.name
        st.session_state.pdf_bytes = uploaded_file.getvalue()
        st.session_state.pages = None
        st.session_state.chunks = None
        st.session_state.index = None

    if st.session_state.pages is None:
        with st.spinner("Extracting PDF text..."):
            st.session_state.pages = extract_pdf_text(
                st.session_state.pdf_bytes
            )

    if not st.session_state.pages:
        st.error(
            "No readable text was found in this PDF. "
            "Scanned/image-only PDFs need OCR before this app can read them."
        )
        st.stop()

    if st.session_state.chunks is None:
        with st.spinner("Creating document chunks..."):
            st.session_state.chunks = create_chunks(
                st.session_state.pages
            )

    with st.spinner("Loading the open-source embedding model..."):
        embedding_model = load_embedding_model()

    if st.session_state.index is None:
        with st.spinner("Creating FAISS vector database..."):
            st.session_state.index = build_faiss_index(
                st.session_state.chunks,
                embedding_model,
            )

    col1, col2, col3 = st.columns(3)
    col1.metric("PDF pages", len(st.session_state.pages))
    col2.metric("Text chunks", len(st.session_state.chunks))
    col3.metric("Vector dimensions", st.session_state.index.d)

    st.success("✅ PDF processed and FAISS vector database is ready.")

    question = st.text_input(
        "💬 Ask a question about your PDF",
        placeholder="Example: What are the main conclusions of this document?",
    )

    if question:
        client = get_groq_client()

        if client is None:
            st.error(
                "GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets "
                "or set it as an environment variable."
            )
            st.stop()

        with st.spinner("Searching the document..."):
            retrieved = retrieve_chunks(
                question,
                st.session_state.index,
                st.session_state.chunks,
                embedding_model,
                top_k=top_k,
            )

        with st.spinner("Generating the answer..."):
            try:
                answer = generate_answer(
                    question,
                    retrieved,
                    client,
                )

                st.subheader("🤖 Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Groq API error: {e}")

        with st.expander("🔎 View retrieved document chunks"):
            for i, item in enumerate(retrieved, start=1):
                st.markdown(
                    f"**Chunk {i} — Page {item['page']} — "
                    f"Similarity: {item['score']:.3f}**"
                )
                st.write(item["text"])
                st.divider()

else:
    st.info("👆 Upload a PDF to start.")

st.markdown("---")
st.caption(
    "RAG pipeline: PDF → text extraction → chunking → embeddings → "
    "FAISS similarity search → Groq LLM answer"
)
