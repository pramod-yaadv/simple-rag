import os
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from embeddings import EmbeddingsUtils

class RAGPipeline:
    def __init__(self):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=host, port=port)
        self.embeddings = EmbeddingsUtils()

    def create_collection_if_not_exists(self, collection_name: str, dim: int = 384):
        """
        Create a Qdrant collection if it doesn't already exist.
        
        Args:
            collection_name (str): Name of the collection.
            dim (int): Vector dimension. Defaults to 384 (all-MiniLM-L6-v2).
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    def upsert_documents(self, collection_name: str, docs: List[Dict[str, str]]):
        """
        Upsert documents into the Qdrant collection.
        
        Args:
            collection_name (str): Name of the collection.
            docs (List[Dict[str, str]]): List of documents with "id" and "text" keys.
        """
        if not docs:
            return

        texts = [doc["text"] for doc in docs]
        vectors = self.embeddings.batch_embeddings(texts)
        
        points = [
            models.PointStruct(
                id=doc["id"],
                vector=vector,
                payload={"text": doc["text"]}
            )
            for doc, vector in zip(docs, vectors)
        ]
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Upserted {len(points)} points into '{collection_name}'.")

    def retrieve(self, collection_name: str, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            collection_name (str): Name of the collection.
            query (str): Query text.
            top_k (int): Number of results to return.

        Returns:
            List[Tuple[str, float, str]]: List of (id, score, text) tuples.
        """
        query_vector = self.embeddings.get_embedding(query)
        
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        ).points
        
        return [
            (hit.id, hit.score, hit.payload.get("text", ""))
            for hit in results
        ]

# Sample upsert call for verification (commented out)
# if __name__ == "__main__":
#     rag = RAGPipeline()
#     collection = "test_collection"
#     rag.create_collection_if_not_exists(collection, 384)
#     
#     docs = [
#         {"id": "67375267-3172-4751-a90b-6cbf26252928", "text": "Qdrant is a vector database."},
#         {"id": "31275267-3172-4751-a90b-6cbf26252123", "text": "FastAPI is a modern web framework."}
#     ]
#     rag.upsert_documents(collection, docs)
#     
#     results = rag.retrieve(collection, "What is Qdrant?")
#     print(results)
