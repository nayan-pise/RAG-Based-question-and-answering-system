import os
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- CONFIGURATION ---
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- RATE LIMITER SETUP ---
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(title="RAG Question Answering API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None 

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- HELPER FUNCTIONS ---
def process_pdf(file_path: str):
    global vector_store
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    
    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        new_db = FAISS.from_documents(chunks, embeddings)
        vector_store.merge_from(new_db)
    print(f"Processed {len(chunks)} chunks.")

# --- API ENDPOINTS ---

@app.post("/upload")
@limiter.limit("5/minute")
async def upload_document(
    request: Request, 
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(process_pdf, file_path)
    return {"message": "File upload started", "filename": file.filename}

@app.post("/ask", response_model=QueryResponse)
@limiter.limit("10/minute")
async def ask_question(
    request: Request, 
    query: QueryRequest
):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    
    docs = vector_store.similarity_search(query.question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    generated_answer = (
        f"Based on the document:\n{context[:500]}...\n"
        "(Connect LLM for full answer)"
    )
    
    sources = [f"Page {d.metadata.get('page', 'unknown')}" for d in docs]
    return QueryResponse(answer=generated_answer, sources=sources)

@app.get("/")
def home():
    return {"message": "RAG System is Running."}