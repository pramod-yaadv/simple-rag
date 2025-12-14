import os
from typing import List
import openai

def assemble_prompt(query: str, retrieved_chunks: List[str]) -> str:
    """
    Assemble a prompt for the LLM using the query and retrieved context chunks.

    Args:
        query (str): The user's query.
        retrieved_chunks (List[str]): List of relevant text chunks.

    Returns:
        str: The formatted prompt.
    """
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Use the following documents to answer the question. Cite source chunks when possible.

Documents:
{context}

Question: {query}
"""
    return prompt

def call_llm(prompt: str, model: str = "llama3.1:8b") -> str:
    """
    Call the LLM to generate a response.
    
    Args:
        prompt (str): The input prompt.
        model (str): Detailed model name. Defaults to "llama3.1:8b" 
                     (which is similar to text-davinci-003 usage).
                     For chat models like gpt-4, use chat completions.

    Returns:
        str: The generated response.
    """
    # 2. Local LLM (Ollama)
    if "llama" in model.lower():
        try:
            import requests
            # Use host.docker.internal for Mac/Windows Docker, or localhost if running natively.
            # Default to localhost for local execution; Docker Compose overrides this.
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            response = requests.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
            
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not set. Cannot call LLM."

    # 1. OpenAI Example
    try:
        # Using the older completions API if simulating text-davinci-003 style
        # or switching to ChatCompletion if model is gpt-3.5-turbo/gpt-4
        if "gpt-3.5-turbo" in model or "gpt-4" in model:
             # Chat Completion
             client = openai.OpenAI(api_key=api_key)
             response = client.chat.completions.create(
                 model="gpt-3.5-turbo", # force chat model for this path
                 messages=[
                     {"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": prompt}
                 ]
             )
             return response.choices[0].message.content.strip()
        else:
            # Legacy Completion (depreciated but requested style roughly)
            client = openai.OpenAI(api_key=api_key)
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=256,
                temperature=0.7
            )
            return response.choices[0].text.strip()

    except Exception as e:
        return f"Error calling OpenAI: {str(e)}"

    # 2. Local LLM Note (Pseudo-code)
    # To use a local LLM (e.g., Llama.cpp, Ollama):
    # import requests
    # payload = {"prompt": prompt, "model": "llama2"}
    # response = requests.post("http://localhost:11434/api/generate", json=payload)
    # return response.json()["response"]
