import streamlit as st
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os
import tempfile

st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 Chat with your Documents (RAG-powered)")

uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        file_ext = uploaded_file.name.split(".")[-1].lower()

        if file_ext == "pdf":
            loader = PyMuPDFLoader(tmp_path)
        elif file_ext == "docx":
            loader = Docx2txtLoader(tmp_path)
        elif file_ext == "txt":
            loader = TextLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue

        documents.extend(loader.load())

    st.success(f"{len(documents)} documents loaded!")

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)

    # RAG chain
    retriever = vectorstore.as_retriever()
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Chat Interface
    st.subheader("💬 Ask a question")
    query = st.text_input("Your question:")
    if query:
        with st.spinner("Generating answer..."):
            response = qa_chain.run(query)
            st.success("Answer:")
            st.write(response)
