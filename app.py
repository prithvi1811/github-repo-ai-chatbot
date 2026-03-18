import time
import streamlit as st

st.set_page_config(
    page_title="GitHub Repository AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------
# Session state
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repo_indexed" not in st.session_state:
    st.session_state.repo_indexed = False
if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""
if "indexed_repo_name" not in st.session_state:
    st.session_state.indexed_repo_name = ""

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
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
        border-color: rgba(16, 185, 129, 0.25);
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

    .repo-note {
        color: #9ca3af;
        font-size: 0.92rem;
        margin-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helper functions
# ----------------------------
def get_repo_name(repo_url: str) -> str:
    repo_url = repo_url.rstrip("/")
    if repo_url:
        return repo_url.split("/")[-1]
    return ""

def index_repository(repo_url: str) -> bool:
    # TODO: Replace this with your real indexing logic
    time.sleep(1.5)
    return True

def ask_repo(question: str) -> str:
    # TODO: Replace this with your real RAG / LLM call
    # This is a placeholder so the UI behaves nicely
    return f"Demo response for: '{question}'\n\nConnect this function to your retrieval + LLM pipeline."

def add_sample_prompt(prompt: str):
    st.session_state.pending_prompt = prompt

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Repository Setup")

    repo_url = st.text_input(
        "GitHub Repo URL",
        value=st.session_state.repo_url,
        placeholder="https://github.com/username/repository",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Index Repo", use_container_width=True):
            if repo_url.strip():
                st.session_state.repo_url = repo_url.strip()
                repo_name = get_repo_name(repo_url.strip())
                with st.spinner("Indexing repository..."):
                    success = index_repository(repo_url.strip())
                if success:
                    st.session_state.repo_indexed = True
                    st.session_state.indexed_repo_name = repo_name
                    st.success(f"Indexed {repo_name}")
                else:
                    st.error("Indexing failed. Please try again.")
            else:
                st.warning("Please enter a GitHub repo URL first.")

    with col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if "pending_prompt" in st.session_state:
                del st.session_state.pending_prompt

    st.markdown("---")

    if st.session_state.repo_indexed:
        st.markdown(
            f"**Active Repo:** `{st.session_state.indexed_repo_name}`"
        )
        st.caption("Repository indexed and ready for questions.")
    else:
        st.caption("Index a repository first to start chatting.")

# ----------------------------
# Main content
# ----------------------------
st.markdown('<div class="main-title">GitHub Repository AI Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions about a GitHub repository using a cleaner chat-style interface.</div>',
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
        • What does this repo do?<br>
        • Summarize the project architecture<br>
        • What are the key files I should read first?<br>
        • How does the authentication flow work?<br>
        • What dependencies does this project use?
    </div>
    """,
    unsafe_allow_html=True,
)

sample_cols = st.columns(3)
with sample_cols[0]:
    if st.button("What is this repo about?", use_container_width=True):
        add_sample_prompt("What is this repo about?")
with sample_cols[1]:
    if st.button("Summarize the architecture", use_container_width=True):
        add_sample_prompt("Summarize the architecture")
with sample_cols[2]:
    if st.button("Which files matter most?", use_container_width=True):
        add_sample_prompt("Which files matter most?")

# Render chat history
if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
else:
    st.markdown(
        '<div class="empty-state">No messages yet. Index a repository and ask your first question.</div>',
        unsafe_allow_html=True,
    )

# Pick prompt from sample button if present
default_prompt = ""
if "pending_prompt" in st.session_state:
    default_prompt = st.session_state.pending_prompt
    del st.session_state.pending_prompt

question = st.chat_input("Ask a question about the repository...")

if default_prompt and not question:
    question = default_prompt

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
                    response = ask_repo(question)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception:
                    clean_error = (
                        "I couldn't generate a response right now. "
                        "Please check the model connection or try again."
                    )
                    st.error(clean_error)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": clean_error}
                    )
