import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CPU THREAD OPTIMIZATIONS ----------------
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ---------------- PROMPT TEMPLATE ----------------
RAG_PROMPT_TEMPLATE = """
You are a question-answering assistant for a PDF document.

Answer the user's question using ONLY the information provided in the context below.

Follow these rules:
1. Use only facts explicitly stated in the context.
2. Do not use your own knowledge, assumptions, or outside information.
3. Do not make up or infer information that is not clearly supported by the context.
4. If the context does not contain enough information to answer the question, respond exactly:
   "I don't know"
5. Give a concise and direct answer to the question.
6. If the context contains multiple relevant pieces of information, combine them carefully to provide the most accurate answer.
7. Preserve important names, numbers, dates, definitions, and technical terms exactly as stated in the context.

Context:
{context}

Question:
{question}
"""


def process_uploaded_files(files, temp_dir="temp_pdfs"):
    """
    Saves uploaded Streamlit files temporarily, splits text into chunks,
    generates CPU embeddings, and builds a FAISS vector database.
    """
    if not files:
        return None

    os.makedirs(temp_dir, exist_ok=True)
    all_docs = []

    try:
        # Load documents from temporary disk storage
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())

        # Chunk the text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = splitter.split_documents(all_docs)

        # Generate embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 64}
        )

        vectors = FAISS.from_documents(docs, embeddings)
        return vectors

    finally:
        # Guarantee cleanup of temporary files
        for file in files:
            file_path = os.path.join(temp_dir, file.name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass


def query_rag_pipeline(vectors, query, groq_api_key, model_name, k=4):
    """
    Runs the LCEL RAG chain and returns the generated answer and retrieved source documents.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0.3
    )

    retriever = vectors.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)
    source_docs = retriever.invoke(query)

    return answer, source_docs