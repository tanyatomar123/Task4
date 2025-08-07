import streamlit as st
import tempfile

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFacePipeline

from transformers import pipeline

# -----------------------------
# 🌟 Page Config
# -----------------------------
st.set_page_config(page_title="🧠 Chat with Your Documents")
st.title("📄 Offline RAG: Chat with Your Documents")
st.markdown("Upload **PDF, DOCX, or TXT** files and ask questions — no API key needed!")

# -----------------------------
# 📂 Upload and Process Files
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload files", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

documents = []

if uploaded_files:
    with st.spinner("🔄 Reading documents..."):
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
# 🧠 Embeddings & Vector DB
# -----------------------------
if documents:
    with st.spinner("🔍 Splitting and embedding documents..."):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(split_docs, embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # -----------------------------
    # 🤖 Load Local LLM (Flan-T5)
    # -----------------------------
    with st.spinner("🧠 Loading local language model..."):
        hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

    st.success("✅ Documents indexed. Ask your questions below!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("💬 Ask a question")

    if query:
        with st.spinner("✍️ Thinking..."):
            relevant_docs = retriever.get_relevant_documents(query)
            result = qa_chain.run(input_documents=relevant_docs, question=query)
            st.session_state.chat_history.append((query, result))

    for q, a in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Answer:** {a}")
        st.markdown("---")
