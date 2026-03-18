import hashlib
import os
import shutil
from pathlib import Path
from typing import List

import streamlit as st
from git import Repo
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="GitHub Repository AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "repo_indexed" not in st.session_state:
    st.session_state.repo_indexed = False
if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""
if "repo_name" not in st.session_state:
    st.session_state.repo_name = ""
if "repo_id" not in st.session_state:
    st.session_state.repo_id = ""
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

BASE_DIR = Path(__file__).parent.resolve()
REPOS_DIR = BASE_DIR / "tmp_repos"
CHROMA_DIR = BASE_DIR / "chroma_db"

REPOS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".php",
    ".swift", ".kt", ".scala", ".c", ".cpp", ".h", ".hpp", ".cs", ".sql",
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".env.example",
    ".html", ".css", ".scss", ".sh", ".bash", ".zsh", ".dockerfile",
}

IGNORED_DIRS = {
    ".git", "node_modules", ".next", "dist", "build", "__pycache__",
    ".venv", "venv", ".idea", ".vscode", "coverage", ".mypy_cache",
    ".pytest_cache", ".streamlit", "chroma_db", "tmp_repos",
}

MAX_FILE_SIZE_BYTES = 300_000

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .sub-title {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border: 1px solid transparent;
    }

    .status-ready {
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.28);
    }

    .status-waiting {
        background: rgba(250, 204, 21, 0.10);
        color: #facc15;
        border-color: rgba(250, 204, 21, 0.25);
    }

    .helper-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,0.02);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .empty-state {
        border: 1px dashed rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.25rem;
        color: #9ca3af;
        text-align: center;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_repo_url(repo_url: str) -> str:
    repo_url = repo_url.strip()
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]
    return repo_url


def repo_hash(repo_url: str) -> str:
    return hashlib.md5(repo_url.encode("utf-8")).hexdigest()


def extract_repo_name(repo_url: str) -> str:
    cleaned = normalize_repo_url(repo_url).rstrip("/")
    return cleaned.split("/")[-1] if cleaned else "repository"


def repo_paths(repo_url: str):
    rid = repo_hash(repo_url)
    local_repo_path = REPOS_DIR / rid
    persist_dir = CHROMA_DIR / rid
    return rid, local_repo_path, persist_dir


def is_text_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in ALLOWED_EXTENSIONS:
        return True
    if path.name.lower() in {"dockerfile", "makefile", "readme", "license"}:
        return True
    return False


def read_file_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return ""
    except Exception:
        return ""


def clone_repository(repo_url: str, local_repo_path: Path) -> None:
    if local_repo_path.exists():
        shutil.rmtree(local_repo_path)
    Repo.clone_from(repo_url, local_repo_path)


def load_repo_documents(local_repo_path: Path) -> List[Document]:
    docs: List[Document] = []

    for root, dirs, files in os.walk(local_repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for filename in files:
            file_path = Path(root) / filename

            try:
                if not file_path.is_file():
                    continue
                if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
            except Exception:
                continue

            if not is_text_file(file_path):
                continue

            content = read_file_safely(file_path)
            if not content or not content.strip():
                continue

            rel_path = str(file_path.relative_to(local_repo_path))
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": rel_path},
                )
            )

    return docs


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )


def build_vectorstore(repo_url: str):
    rid, local_repo_path, persist_dir = repo_paths(repo_url)

    clone_repository(repo_url, local_repo_path)
    raw_docs = load_repo_documents(local_repo_path)

    if not raw_docs:
        raise ValueError("No readable source files were found in this repository.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )
    split_docs = splitter.split_documents(raw_docs)

    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=get_embeddings(),
        persist_directory=str(persist_dir),
        collection_name=f"repo_docs_{rid}",
    )

    return rid, vectorstore, len(raw_docs), len(split_docs)


def load_vectorstore(repo_url: str):
    rid, _, persist_dir = repo_paths(repo_url)
    if not persist_dir.exists():
        return None

    return Chroma(
        collection_name=f"repo_docs_{rid}",
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
    )


def answer_question(question: str, repo_url: str) -> str:
    vectorstore = load_vectorstore(repo_url)
    if vectorstore is None:
        return "Please index the repository first."

    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join(
        [f"FILE: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in docs]
    )

    repo_name = extract_repo_name(repo_url)

    prompt = f"""
You are a helpful AI assistant for understanding the GitHub repository: {repo_name}.

Answer ONLY using the repository context below.
If the answer is not clearly supported by the context, say that you are not fully sure.
Do not describe any other repository.
Keep the answer practical and concise.
Mention relevant file names when useful.

Repository Name:
{repo_name}

Repository Context:
{context}

User Question:
{question}
"""

    response = get_llm().invoke(prompt)
    return response.content


def queue_sample_prompt(prompt: str):
    st.session_state.pending_prompt = prompt


with st.sidebar:
    st.header("Repository Setup")

    repo_input = st.text_input(
        "GitHub Repo URL",
        value=st.session_state.repo_url,
        placeholder="https://github.com/username/repository",
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Index Repo", use_container_width=True):
            repo_input = normalize_repo_url(repo_input)

            if not repo_input.startswith("https://github.com/"):
                st.warning("Please enter a valid public GitHub repository URL.")
            elif "GROQ_API_KEY" not in os.environ:
                st.error("Missing GROQ_API_KEY. Add it in Streamlit app secrets.")
            else:
                try:
                    with st.spinner("Cloning, reading files, and building embeddings..."):
                        rid, _, raw_count, chunk_count = build_vectorstore(repo_input)

                    st.session_state.messages = []
                    if "pending_prompt" in st.session_state:
                        del st.session_state.pending_prompt

                    st.session_state.repo_url = repo_input
                    st.session_state.repo_name = extract_repo_name(repo_input)
                    st.session_state.repo_id = rid
                    st.session_state.repo_indexed = True
                    st.session_state.vectorstore_ready = True

                    st.success(
                        f"Indexed {raw_count} files into {chunk_count} searchable chunks."
                    )
                except Exception as e:
                    st.session_state.repo_indexed = False
                    st.session_state.vectorstore_ready = False
                    st.error(f"Indexing failed: {e}")

    with c2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if "pending_prompt" in st.session_state:
                del st.session_state.pending_prompt

    st.markdown("---")

    if st.session_state.repo_indexed:
        st.markdown(f"**Active Repo:** `{st.session_state.repo_name}`")
        st.caption("Repository indexed and ready for questions.")
    else:
        st.caption("Index a repository first to start chatting.")

st.markdown(
    '<div class="main-title">GitHub Repository AI Chatbot</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Ask questions about a GitHub repository with retrieval-backed answers and a cleaner chat-style interface.</div>',
    unsafe_allow_html=True,
)

if st.session_state.repo_indexed:
    st.markdown(
        '<div class="status-pill status-ready">Repo Indexed and Ready</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-pill status-waiting">Waiting for Repository Index</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="helper-card">
        <strong>Try asking things like:</strong><br><br>
        • What does this repository do?<br>
        • Summarize the project architecture<br>
        • Which files should I read first?<br>
        • How does the authentication flow work?<br>
        • What are the key dependencies in this project?
    </div>
    """,
    unsafe_allow_html=True,
)

sample_cols = st.columns(3)
with sample_cols[0]:
    if st.button("What does this repo do?", use_container_width=True):
        queue_sample_prompt("What does this repository do?")
with sample_cols[1]:
    if st.button("Summarize architecture", use_container_width=True):
        queue_sample_prompt("Summarize the project architecture.")
with sample_cols[2]:
    if st.button("Key files to read first", use_container_width=True):
        queue_sample_prompt("Which files should I read first and why?")

if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
else:
    st.markdown(
        '<div class="empty-state">No messages yet. Index a public GitHub repository and ask your first question.</div>',
        unsafe_allow_html=True,
    )

queued_prompt = ""
if "pending_prompt" in st.session_state:
    queued_prompt = st.session_state.pending_prompt
    del st.session_state.pending_prompt

question = st.chat_input("Ask a question about the repository...")

if queued_prompt and not question:
    question = queued_prompt

if question:
    if not st.session_state.repo_indexed:
        st.warning("Please index a repository before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(question, st.session_state.repo_url)
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception:
                    clean_error = (
                        "I couldn't generate a response right now. "
                        "Please try again after re-indexing the repository."
                    )
                    st.error(clean_error)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": clean_error}
                    )
