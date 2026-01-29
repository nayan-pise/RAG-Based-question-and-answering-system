# 📚 RAG-Based Question Answering API

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![LangChain](https://img.shields.io/badge/AI-LangChain-orange)

## 🚀 Project Overview
This project is a **Retrieval-Augmented Generation (RAG)** system built to answer user questions based on uploaded PDF documents.

Unlike standard chatbots, this API allows users to upload their own private data (PDFs). The system processes the text, converts it into vector embeddings, and retrieves specific paragraphs relevant to the user's question. It is designed with **FastAPI** for high performance and includes **background processing** and **rate limiting**.

---

## 🏗️ System Architecture

The system follows a standard RAG pipeline:

```mermaid
graph LR
    User[User] -->|Upload PDF| API[FastAPI /upload]
    API -->|Background Task| Splitter[Text Chunking]
    Splitter -->|Chunks| Embed[HuggingFace Embeddings]
    Embed -->|Vectors| FAISS[(FAISS Vector Store)]
    
    User -->|Ask Question| Query[FastAPI /ask]
    Query -->|Embed Query| Embed
    Embed -->|Similarity Search| FAISS
    FAISS -->|Top Matches| Context[Retrieved Context]
    Context -->|Generate Answer| User
