import os
import streamlit as st
from tempfile import NamedTemporaryFile
import fitz  # PyMuPDF
import docx
import pandas as pd
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG App (Cloud Friendly)", layout="centered")
st.title("📄 Ask Your Files (RAG App - Streamlit Cloud)")

openai_api_key = st.secrets["OPENAI_API_KEY"]

uploaded_files = st.file_uploader("Upload PDFs, DOCX, or Excel files", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)

def load_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def load_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

docs = []

if uploaded_files:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    for f in uploaded_files:
        suffix = f.name.split(".")[-1]
        with NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        try:
            if suffix == "pdf":
                content = load_pdf(tmp_path)
            elif suffix == "docx":
                content = load_docx(tmp_path)
            elif suffix == "xlsx":
                df = pd.read_excel(tmp_path)
                content = df.to_string()
            else:
                st.warning(f"Unsupported file type: {suffix}")
                continue

            docs.append(Document(page_content=content, metadata={"source": f.name}))
        except Exception as e:
            st.warning(f"Error reading {f.name}: {e}")

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