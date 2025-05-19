import os
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

# Use Streamlit secrets instead of os.getenv
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENVIRONMENT"]

INDEX_NAME = "persona-chatbot"

def init_pinecone():
    """
    Initialize Pinecone with the API key.
    """
    if not PINECONE_API_KEY:
        raise ValueError("Pinecone API key not found in Streamlit secrets.")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

def create_index(pc, dim: int = 1536):
    """
    Create a Pinecone index if it doesn't exist already.
    """
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENV
            )
        )
        print(f"Created Pinecone index '{INDEX_NAME}'.")
    else:
        print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")

def get_index(pc):
    """
    Connect to and return the Pinecone index for further operations.
    """
    return pc.Index(INDEX_NAME)

if __name__ == "__main__":
    pc = init_pinecone()
    create_index(pc)
