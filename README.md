# GitHub Repository AI Chatbot

An AI chatbot that lets you ask questions about any GitHub repository.

The system indexes a repository, converts code into vector embeddings, and uses a local language model to answer questions about the project.

## Example Questions

* What does this repository do?
* Explain the project architecture
* Which files send emails?
* How does the automation workflow work?

## Architecture

GitHub Repo
→ Document Loader
→ Text Chunking
→ Sentence Transformer Embeddings
→ Chroma Vector Database
→ Retriever
→ Local LLM (Llama3 via Ollama)
→ Answer Generation

## Tech Stack

* Python
* LangChain
* ChromaDB
* Sentence Transformers
* Ollama (Llama3)
* Streamlit

## Run the Project

Clone the repo

```
git clone https://github.com/YOUR_USERNAME/github-repo-chatbot.git
cd github-repo-chatbot
```

Install dependencies

```
pip install -r requirements.txt
```

Index a repository

```
python ingest.py
```

Run chatbot

```
python query.py
```

Run the UI

```
streamlit run app.py
```

## Features

* Chat with any GitHub repository
* Code-aware retrieval
* Source file citations
* Fully local AI pipeline
