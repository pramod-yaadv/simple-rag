import os
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from embeddings import EmbeddingsUtils

@pytest.fixture(scope="module")
def embeddings_utils():
    # Using default model "all-MiniLM-L6-v2"
    return EmbeddingsUtils()

def test_get_embedding_dimension(embeddings_utils):
    text = "Hello, world!"
    vector = embeddings_utils.get_embedding(text)
    
    # all-MiniLM-L6-v2 output dimension is 384
    assert len(vector) == 384
    assert isinstance(vector, list)
    assert isinstance(vector[0], float)

def test_batch_embeddings(embeddings_utils):
    texts = ["Hello", "World"]
    vectors = embeddings_utils.batch_embeddings(texts)
    
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384
