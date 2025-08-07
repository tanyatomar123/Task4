import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile

# Page settings
st.set_page_config(page_title="Chat with Your Documents", layout="wide")
st.title("📄 Chat with Your Documents (RAG App)")

# File uploader
uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# OpenAI API key input
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

# Start processing
if st.button("📚 Process Documents") and uploaded_files and openai_api_key:
    all_texts = []

    for uploaded_file in uploaded_files:
        suffix = uploaded_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if suffix == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == "docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == "txt":
            loader = TextLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue

        docs = loader.load()
        all_texts.extend(docs)

    st.info("🔍 Splitting and indexing documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_texts)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
        retriever=retriever,
        return_source_documents=True
    )

    st.success("✅ Documents processed. Ask your questions!")

    query = st.text_input("💬 Ask something from your documents:")
    if query:
        result = qa_chain({"query": query})
        st.markdown("### 🧠 Answer")
        st.write(result['result'])

        with st.expander("📚 Sources"):
            for doc in result['source_documents']:
                st.write(doc.metadata)
                st.write(doc.page_content[:300] + "...")
else:
    st.info("Please upload documents and enter your OpenAI API key.")
