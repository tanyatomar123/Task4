import streamlit as st
import tempfile

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# -----------------------------
# 🔐 OpenAI API Key Input
# -----------------------------
st.set_page_config(page_title="📄 Chat with Your Documents")
st.title("🧠 Chat with Your Documents")
st.markdown("Upload **PDF, DOCX, or TXT** files and ask questions.")

st.sidebar.header("🔐 API Key & Settings")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

# -----------------------------
# 📂 Upload and Process Files
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

documents = []

if uploaded_files:
    with st.spinner("🔄 Loading and processing documents..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                loader = UnstructuredFileLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                st.error(f"❌ Failed to load file: {uploaded_file.name}\n\nError: {e}")

# -----------------------------
# 🧠 Create Vectorstore & QA Chain
# -----------------------------
if documents:
    with st.spinner("🔍 Splitting and embedding documents..."):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
            retriever=retriever,
            return_source_documents=True,
        )

    st.success("✅ Documents indexed! You can now chat with them.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("💬 Ask a question")

    if query:
        with st.spinner("🤖 Generating response..."):
            result = qa_chain({"query": query})
            st.session_state.chat_history.append((query, result["result"]))

    for q, a in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**RAG Answer:** {a}")
        st.markdown("---")
