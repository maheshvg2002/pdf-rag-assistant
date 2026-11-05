# ⚡ PDF RAG Assistant

A secure, lightweight PDF RAG chatbot built with LangChain, Groq LLMs, and FAISS vector search, featuring an interactive Streamlit UI for document processing and source-chunk verification.

## 🚀 Features
- **PDF Document Ingestion:** Upload multiple PDFs and process them directly into local vector embeddings.
- **Fast Local Embeddings:** Utilizes `sentence-transformers/all-MiniLM-L6-v2` running efficiently on CPU.
- **Groq LLM Integration:** Fast inference using Groq API and LangChain Expression Language (LCEL).
- **Source Verification:** Expandable source cards showing exact file names, page numbers, and chunk text for every response.
- **Chat Management:** Export conversation history as a Markdown transcript or clear chat history instantly.

---

## 🛠️ Tech Stack
* **Frontend/UI:** [Streamlit](https://streamlit.io/)
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss)
* **LLM Provider:** [Groq API](https://groq.com/)

---

## 📦 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/maheshvg2002/pdf-rag-assistant.git
   cd pdf-rag-assistant
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL_NAME=openai/gpt-oss-120b
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deployment on Streamlit Community Cloud
1. Push your code to a public GitHub repository.
2. Head over to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Deploy a new app by linking your repository and setting `app.py` as the main file path.
4. Add your API credentials securely under **Advanced settings > Secrets**:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   GROQ_MODEL_NAME = "openai/gpt-oss-120b"
   ```