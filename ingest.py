import os
import shutil
from git import Repo
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
LOCAL_REPO_PATH = "./repo"


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

    return loader.load()


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


if __name__ == "__main__":

    repo_url = input("Enter GitHub repo URL: ")

    print("Cloning repo...")
    clone_repo(repo_url)

    print("Loading documents...")
    docs = load_documents()

    print("Splitting docs...")
    chunks = split_documents(docs)

    print("Creating embeddings...")
    create_vector_db(chunks)

    print("Done. Repository indexed.")
