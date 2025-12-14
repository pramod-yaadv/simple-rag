import os
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ingestion import chunk_text, ingest_file

def test_chunk_text_sizes():
    text = "1234567890" * 10  # 100 chars
    # Chunk size 20, overlap 5
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    
    assert len(chunks) > 0
    assert len(chunks[0]) <= 20
    
    # Check overlap (roughly)
    # 0-20, 15-35, 30-50 ...
    # second chunk should start with last 5 chars of first chunk
    assert chunks[1].startswith(chunks[0][-5:])

def test_ingest_file_txt():
    # Use the sample file we created
    file_path = os.path.join(os.path.dirname(__file__), "../../data/sample.txt")
    if not os.path.exists(file_path):
        pytest.skip("Sample data file not found")
        
    chunks = ingest_file(file_path)
    assert len(chunks) > 0
    # Check structure (id, text)
    assert isinstance(chunks[0], tuple)
    assert len(chunks[0]) == 2
    assert isinstance(chunks[0][0], str) # uuid
    assert isinstance(chunks[0][1], str) # text
