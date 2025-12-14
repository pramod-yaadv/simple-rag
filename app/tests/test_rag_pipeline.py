import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rag import RAGPipeline

@pytest.fixture
def mock_qdrant_client():
    with patch("rag.QdrantClient") as mock:
        yield mock

@pytest.fixture
def mock_embeddings():
    with patch("rag.EmbeddingsUtils") as mock:
        instance = mock.return_value
        # Mock get_embedding to return a dummy vector
        instance.get_embedding.return_value = [0.1] * 384
        instance.batch_embeddings.return_value = [[0.1] * 384]
        yield instance

def test_upsert_documents(mock_qdrant_client, mock_embeddings):
    pipeline = RAGPipeline()
    # Replace the client and embeddings with mocks
    pipeline.client = mock_qdrant_client.return_value
    pipeline.embeddings = mock_embeddings
    
    docs = [{"id": "1", "text": "Test doc"}]
    pipeline.upsert_documents("test_collection", docs)
    
    # Verify client.upsert called
    pipeline.client.upsert.assert_called_once()
    args, kwargs = pipeline.client.upsert.call_args
    assert kwargs['collection_name'] == "test_collection"
    assert len(kwargs['points']) == 1

def test_retrieve(mock_qdrant_client, mock_embeddings):
    pipeline = RAGPipeline()
    pipeline.client = mock_qdrant_client.return_value
    pipeline.embeddings = mock_embeddings
    
    # Mock search result
    mock_point = MagicMock()
    mock_point.id = "1"
    mock_point.score = 0.9
    mock_point.payload = {"text": "Test result"}
    
    # Mock query_points return value (QueryResponse object with points attribute)
    mock_response = MagicMock()
    mock_response.points = [mock_point]
    pipeline.client.query_points.return_value = mock_response
    
    results = pipeline.retrieve("test_collection", "query", top_k=2)
    
    assert len(results) == 1
    assert results[0] == ("1", 0.9, "Test result")
    pipeline.client.query_points.assert_called_once()
