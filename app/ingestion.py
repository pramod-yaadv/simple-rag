import uuid
from typing import List, Tuple
import PyPDF2

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of `chunk_size` characters with `overlap`.
    
    Args:
        text (str): Input text to chunk.
        chunk_size (int): Check size in characters. Defaults to 500.
        overlap (int): Overlap size in characters. Defaults to 50.

    Returns:
        List[str]: List of text chunks.
        
    Example:
        >>> text = "This is a long text..."
        >>> chunks = chunk_text(text, chunk_size=10, overlap=2)
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end == text_len:
            break
            
        start += chunk_size - overlap
        
    return chunks

def ingest_file(path: str) -> List[Tuple[str, str]]:
    """
    Ingest a file (PDF or Text) and return chunks with IDs.
    
    Args:
        path (str): File path to ingest.

    Returns:
        List[Tuple[str, str]]: List of (uuid, chunk_text) tuples.
    """
    text = ""
    try:
        if path.lower().endswith('.pdf'):
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        else:
            # Fallback to plain text
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return []

    chunks = chunk_text(text)
    return [(str(uuid.uuid4()), chunk) for chunk in chunks]

import requests
from bs4 import BeautifulSoup

def scrape_url(url: str) -> str:
    """
    Scrape text content from a URL.
    
    Args:
        url (str): The URL to scrape.

    Returns:
        str: extracted text content.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Kill all script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error scraping URL {url}: {e}")
        return ""

# Sample usage for verification (commented out)
# if __name__ == "__main__":
#     sample_text = "Microservice architecture is an architectural style that structures an application as a collection of services."
#     print(chunk_text(sample_text, chunk_size=20, overlap=5))
