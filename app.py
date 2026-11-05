import os
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import process_uploaded_files, query_rag_pipeline

# Load environment configuration
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="⚡",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }
    .source-card {
        border-left: 3px solid #4CAF50;
        background-color: rgba(128, 128, 128, 0.05);
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    div.stButton > button {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Chabot for PDFs")

# ---------------- SESSION STATE SETUP ----------------
if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- DOCUMENT UPLOADER ----------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if st.button("Process Documents"):
    if not API_KEY:
        st.error("⚠️ GROQ_API_KEY is not set in your .env file.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
    else:
        with st.spinner("Processing documents into vector store..."):
            vectors = process_uploaded_files(uploaded_files)
            if vectors:
                st.session_state.vectors = vectors
            else:
                st.error("Failed to process documents. Please check file format.")

# Prompt user when ready
if st.session_state.vectors is not None:
    st.success("✅ **Vector database ready!** You can now start chatting with your documents below.")

st.divider()

# ---------------- CHAT HEADER & CONTROLS ----------------
header_col1, header_col2, header_col3 = st.columns([6, 2, 2])

with header_col1:
    st.subheader("💬 Chat Interface")

with header_col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

with header_col3:
    if st.session_state.chat_history:
        transcript = "# Glean Chat Transcript\n\n"
        for msg in st.session_state.chat_history:
            role = "**User**" if msg["role"] == "user" else "**Assistant**"
            transcript += f"{role}: {msg['content']}\n\n"

        st.download_button(
            label="📥 Export Chat",
            data=transcript,
            file_name="chat_transcript.md",
            mime="text/markdown",
            use_container_width=True
        )

# ---------------- CHAT RENDER & INPUT ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about your uploaded PDFs..."):
    if not API_KEY:
        st.error("⚠️ GROQ_API_KEY is missing from .env.")
        st.stop()

    if not st.session_state.vectors:
        st.error("⚠️ Please upload and process your PDFs first using the button above.")
        st.stop()

    # User message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving answers..."):
            answer, source_docs = query_rag_pipeline(
                st.session_state.vectors,
                user_input,
                API_KEY,
                MODEL_NAME
            )

            st.markdown(answer)

            # Metadata source attribution cards
            if source_docs:
                with st.expander("📚 Retrieved Source Chunks"):
                    for i, doc in enumerate(source_docs):
                        source_file = os.path.basename(doc.metadata.get("source", "Unknown PDF"))
                        page_num = doc.metadata.get("page", 0) + 1
                        char_count = len(doc.page_content)

                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Chunk {i+1}</strong> &bull; 📄 <code>{source_file}</code> &bull; 📑 Page {page_num} &bull; 🔤 {char_count} chars
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(doc.page_content)
                        st.divider()

    st.session_state.chat_history.append({"role": "assistant", "content": answer})