import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import tempfile
import os

st.set_page_config(page_title="RAG App", layout="wide")
st.title("📄 Chat with Your Documents (RAG)")

# Upload multiple files
uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Temporary directory for saving files
temp_dir = tempfile.TemporaryDirectory()

# Function to load documents
def load_documents(files):
    docs = []
    for file in files:
        file_path = os.path.join(temp_dir.name, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        if file.name.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        docs.extend(loader.load())
    return docs

if uploaded_files:
    st.success("✅ Files uploaded successfully!")

    with st.spinner("Processing documents..."):
        raw_docs = load_documents(uploaded_files)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(raw_docs)

        # Use sentence-transformers embedding
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Use Chroma (in-memory)
        vectorstore = Chroma.from_documents(chunks, embeddings)

        # Load LLM (Open-source one from Hugging Face Hub)
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

        # Create Retrieval QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        st.success("✅ Documents processed. You can now chat!")

        # Chat Interface
        query = st.text_input("Ask a question about your documents:")
        if query:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(query)
                st.markdown(f"**Answer:** {answer}")
