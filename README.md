# Simple RAG

A RAG (Retrieval-Augmented Generation) pipeline project using FastAPI and Qdrant.

## Setup and Run

1.  Make sure you have Docker and Docker Compose installed.
2.  Run the application:

    ```bash
    docker-compose up --build
    ```

3.  The API will be available at `http://localhost:8000`.
4.  Qdrant will be available at `http://localhost:6333`.

## Configuration

- `OPENAI_API_KEY`: Required for LLM generation (if using OpenAI).
- `QDRANT_HOST`: Hostname of Qdrant (default: `qdrant`).
- `QDRANT_PORT`: Port of Qdrant (default: `6333`).
- `OLLAMA_BASE_URL`: URL for Ollama (default: `http://localhost:11434` or `http://host.docker.internal:11434` in Docker).

## Manual Testing with Postman

A Postman collection is provided in `postman_collection.json`.

1.  Open Postman.
2.  Click **Import**.
3.  Drag and drop or select `postman_collection.json` from the project root.
4.  You will see a collection named "Simple RAG" with ready-to-use requests for all endpoints.
