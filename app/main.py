import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import RAGPipeline
from ingestion import chunk_text, scrape_url
from generation import assemble_prompt, call_llm

app = FastAPI()

# CORS Setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline
# Note: In a real app, you might want a singleton dependency injection
rag_pipeline = RAGPipeline()
COLLECTION_NAME = "rag_collection"
# Ensure collection exists on startup
rag_pipeline.create_collection_if_not_exists(COLLECTION_NAME)

class UpsertRequest(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[dict] = None

class BulkUpsertRequest(BaseModel):
    documents: List[UpsertRequest]

class IngestUrlRequest(BaseModel):
    url: str
    metadata: Optional[dict] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG Antigravity API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upsert")
def upsert_document(request: UpsertRequest):
    try:
        # 1. Chunking
        chunks = chunk_text(request.text)
        
        # 2. Prepare for RAG upsert
        # If id is not provided, we rely on ingestion pipeline or generate here.
        # But rag.upsert_documents expects list of dicts with 'id' and 'text'
        import uuid
        docs = []
        base_id = request.id if request.id else str(uuid.uuid4())
        
        # Simple strategy: use base_id for the first chunk or derived for others?
        # Actually RAGPipeline.upsert_documents expects us to pass chunks if we want chunk-level retrieval.
        # But wait, rag.upsert_documents takes docs and embeds them.
        # If we just pass request.text as one doc, it might be too large.
        # We should upsert the CHUNKS.
        
        for i, chunk in enumerate(chunks):
            # Qdrant requires UUID or Unsigned Integer IDs
            # We generate a new UUID for each chunk to satisfy this
            chunk_id = str(uuid.uuid4())
            docs.append({"id": chunk_id, "text": chunk})

        # 3. Upsert
        rag_pipeline.upsert_documents(COLLECTION_NAME, docs)
        
        return {"message": f"Successfully processed and upserted {len(docs)} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk_upsert")
def bulk_upsert_documents(request: BulkUpsertRequest):
    try:
        all_docs = []
        import uuid
        for doc_req in request.documents:
            chunks = chunk_text(doc_req.text)
            base_id = doc_req.id if doc_req.id else str(uuid.uuid4())
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                all_docs.append({"id": chunk_id, "text": chunk})
        
        rag_pipeline.upsert_documents(COLLECTION_NAME, all_docs)
        return {"message": f"Successfully processed and upserted {len(all_docs)} chunks from {len(request.documents)} documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest_url")
def ingest_url_endpoint(request: IngestUrlRequest):
    try:
        # 1. Scrape
        text = scrape_url(request.url)
        if not text:
             raise HTTPException(status_code=400, detail=f"Failed to scrape content from {request.url}")

        # 2. Chunking
        chunks = chunk_text(text)
        
        # 3. Prepare for RAG upsert
        import uuid
        docs = []
        base_id = str(uuid.uuid4())
        
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            docs.append({"id": chunk_id, "text": chunk})

        # 4. Upsert
        rag_pipeline.upsert_documents(COLLECTION_NAME, docs)
        
        return {"message": f"Successfully scraped and upserted {len(docs)} chunks from {request.url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        # 1. Retrieve
        retrieved_results = rag_pipeline.retrieve(COLLECTION_NAME, request.query, request.top_k)
        # retrieved_results is list of (id, score, text)
        
        source_texts = [res[2] for res in retrieved_results]
        
        # 2. Assemble Prompt
        prompt = assemble_prompt(request.query, source_texts)
        
        # 3. Call LLM
        answer = call_llm(prompt)
        
        return QueryResponse(answer=answer, sources=source_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
