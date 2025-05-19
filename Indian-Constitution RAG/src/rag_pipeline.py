import openai
from typing import List, Dict
import os
import streamlit as st
from pinecone_setup import get_index, init_pinecone

# Use Streamlit secrets instead of os.getenv
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_embedding(text: str) -> List[float]:
    """
    Generate embeddings using OpenAI's API.
    """
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def retrieve_context(query: str, top_k: int = 3) -> List[str]:
    """
    Retrieve relevant context from Pinecone.
    """
    pc = init_pinecone()
    index = get_index(pc)
    query_embedding = get_embedding(query)
    
    # Search in Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract text from results
    contexts = [match.metadata['text'] for match in results.matches]
    return contexts

def generate_response(query: str, contexts: List[str]) -> str:
    """
    Generate response using GPT-3.5-turbo.
    """
    # Prepare the prompt
    context_str = "\n\n".join(contexts)
    prompt = f"""You are an expert on the Indian Constitution. Answer the question based on the provided context. 
    If the context doesn't contain relevant information, say so.
    Be specific and accurate in your response.

Context:
{context_str}

Question: {query}

Answer:"""

    # Generate response
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert on the Indian Constitution, providing accurate and detailed answers based on the provided context."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def process_query(query: str) -> str:
    """
    Process a user query through the RAG pipeline.
    """
    # Retrieve relevant context
    contexts = retrieve_context(query)
    
    # Generate response
    response = generate_response(query, contexts)
    
    return response
