import os
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader
import pandas as pd
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG App for Streamlit Cloud", layout="centered")
st.title("📄 Ask Your Files (RAG App - Streamlit Cloud Compatible)")

openai_api_key = st.secrets["OPENAI_API_KEY"]

uploaded_files = st.file_uploader("Upload PDFs, DOCX, or Excel files", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    docs = []
    for f in uploaded_files:
        suffix = f.name.split(".")[-1]
        with NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        if suffix == "pdf":
            loader = PyMuPDFLoader(tmp_path)
            docs.extend(loader.load())
        elif suffix == "docx":
            loader = Docx2txtLoader(tmp_path)
            docs.extend(loader.load())
        elif suffix == "xlsx":
            try:
                df = pd.read_excel(tmp_path)
                content = df.to_string()
                docs.append(Document(page_content=content, metadata={"source": f.name}))
            except Exception as e:
                st.warning(f"Error reading Excel file: {e}")
                continue
        else:
            st.warning(f"Unsupported file type: {suffix}")
            continue

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(split_docs, embedding_model)
    retriever = db.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever
    )

    question = st.text_input("Ask a question about your documents:")

    if question:
        with st.spinner("Searching your documents..."):
            answer = qa_chain.run(question)
            st.markdown("**Answer:**")
            st.write(answer)