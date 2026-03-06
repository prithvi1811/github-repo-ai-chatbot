import requests
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

import requests
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()


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


if __name__ == "__main__":
    while True:
        question = input("\nAsk about the repo (or type 'exit'): ").strip()

        if question.lower() == "exit":
            break

        answer, sources = answer_question(question)

        print("\nAnswer:\n")
        print(answer)

        print("\nRetrieved sources:")
        for src in sources:
            print("-", src)
