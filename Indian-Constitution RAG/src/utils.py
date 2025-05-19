import re
from typing import List

def clean_text(text: str) -> str:
    """
    Clean and preprocess text.
    """
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    """
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def format_qa_pair(question: str, answer: str) -> str:
    """
    Format a question-answer pair for display.
    """
    return f"Q: {question}\nA: {answer}"
