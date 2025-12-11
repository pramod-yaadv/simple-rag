from fastapi import FastAPI
from rag import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

@app.get("/")
def read_root():
    return {"message": "Welcome to RAG Antigravity"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
