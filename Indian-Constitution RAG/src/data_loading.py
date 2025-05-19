import json
from typing import List, Dict
import openai
from dotenv import load_dotenv
import os
from pinecone_setup import get_index, init_pinecone, create_index

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_text_data(file_path: str) -> str:
    """
    Load text data from file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_chunks(text: str, chunk_size: int = 150) -> list[str]:
    """
    Split text into chunks of approximately chunk_size words.
    Each chunk will be a paragraph or part of a paragraph to maintain context.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    
    for paragraph in paragraphs:
        words = paragraph.split()
        # If paragraph is shorter than chunk_size, keep it as one chunk
        if len(words) <= chunk_size:
            chunks.append(paragraph)
        else:
            # Split longer paragraphs into smaller chunks
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
    
    return chunks

def get_embedding(text: str) -> list[float]:
    """
    Generate embeddings using OpenAI's API.
    """
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def store_embeddings(chunks: list[str]):
    """
    Generate embeddings and store them in Pinecone.
    """
    pc = init_pinecone()
    index = get_index(pc)
    
    total_chunks = len(chunks)
    print(f"Total chunks to process: {total_chunks}")
    
    # Process in batches of 100
    batch_size = 100
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        vectors = []
        for j, chunk in enumerate(batch):
            embedding = get_embedding(chunk)
            vectors.append((f"chunk_{i+j}", embedding, {"text": chunk}))
        
        # Upsert batch
        index.upsert(vectors=vectors)
        print(f"Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks")

def main():
    # Initialize Pinecone and create index
    pc = init_pinecone()
    create_index(pc)
    
    # Load and process data
    text = load_text_data("data/data.txt")
    print("Loaded text data")
    
    chunks = create_chunks(text)
    print(f"Created {len(chunks)} chunks")
    
    # Store embeddings
    store_embeddings(chunks)

if __name__ == "__main__":
    main()
