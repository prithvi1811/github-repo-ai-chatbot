import os
import shutil
import streamlit as st
import requests

from git import Repo
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
LOCAL_REPO_PATH = "./repo"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()


def clone_repo(repo_url):
    if os.path.exists(LOCAL_REPO_PATH):
        shutil.rmtree(LOCAL_REPO_PATH)
    Repo.clone_from(repo_url, LOCAL_REPO_PATH)


def load_documents():
    loader = GitLoader(
        repo_path=LOCAL_REPO_PATH,
        branch="main"
    )
    docs = loader.load()

    for doc in docs:
        source = doc.metadata.get("file_path") or doc.metadata.get("source") or ""
        doc.metadata["source"] = source

    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


def create_vector_db(chunks):
    embeddings = LocalSentenceTransformerEmbeddings()

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )


def ingest_repo(repo_url):
    clone_repo(repo_url)
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_db(chunks)
    return len(docs), len(chunks)


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def answer_question(question):
    embeddings = LocalSentenceTransformerEmbeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    docs = vectorstore.similarity_search(question, k=5)

    sources = []
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        if source not in sources:
            sources.append(source)

        snippet = doc.page_content[:4000]
        context_parts.append(f"SOURCE {i}: {source}\n{snippet}")

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI assistant helping a developer understand a GitHub repository.

Use ONLY the repository context below.

Explain:
1. what the project does
2. how it works
3. the role of the important files

If the full answer is not obvious, give your best interpretation from the retrieved code and text.

Question:
{question}

Repository Context:
{context}

At the end, include:
Sources:
- file1
- file2
"""

    answer = ask_ollama(prompt)
    return answer, sources


st.set_page_config(page_title="GitHub Repo Chatbot", layout="wide")
st.title("GitHub Repository Chatbot")

with st.sidebar:
    st.header("Repository Setup")
    repo_url = st.text_input("GitHub Repo URL")
    if st.button("Index Repository"):
        if not repo_url.strip():
            st.error("Please enter a GitHub repository URL.")
        else:
            with st.spinner("Indexing repository..."):
                try:
                    doc_count, chunk_count = ingest_repo(repo_url.strip())
                    st.success(f"Indexed {doc_count} files into {chunk_count} chunks.")
                except Exception as e:
                    st.error(f"Error: {e}")

st.subheader("Ask Questions")
question = st.text_input("Ask about the repository")

if st.button("Ask"):
    if not os.path.exists(CHROMA_PATH):
        st.error("Please index a repository first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(question.strip())
                st.markdown("### Answer")
                st.write(answer)

                st.markdown("### Retrieved Sources")
                for src in sources:
                    st.code(src)
            except Exception as e:
                st.error(f"Error: {e}")
