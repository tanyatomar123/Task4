# app.py
import streamlit as st
import tempfile
import os

from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# -----------------------
# Sidebar: API Key Input
# -----------------------
st.set_page_config(page_title="📄 RAG App - Chat with Docs")
st.sidebar.title("🔐 API Key & Settings")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if not openai_api_key:
    st.warning("🔑 Please enter your OpenAI API key in the sidebar to start.")
    st.stop()

# -----------------------
# Upload Documents
# -----------------------
st.title("🧠 Chat with Your Documents")
st.markdown("Upload **PDF, DOCX, or TXT** files and ask questions.")

uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

documents = []

if uploaded_files:
    with st.spinner("📄 Loading and parsing documents..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                loader = Docx2txtLoader(tmp_path)
            elif uploaded_file.type == "text/plain":
                loader = TextLoader(tmp_path)
            else:
                st.error(f"Unsupported file format: {uploaded_file.name}")
                continue

            docs = loader.load()
            documents.extend(docs)

    # -----------------------
    # Chunk and Embed Docs
    # -----------------------
    with st.spinner("🔍 Splitting and embedding documents..."):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
            retriever=retriever,
            return_source_documents=True
        )

    # -----------------------
    # Chat Interface
    # -----------------------
    st.success("✅ Documents uploaded and indexed!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about your documents")

    if query:
        with st.spinner("🤖 Generating answer..."):
            result = qa_chain({"query": query})

            st.session_state.chat_history.append((query, result["result"]))

        for q, a in st.session_state.chat_history[::-1]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**RAG Answer:** {a}")
            st.markdown("---")
