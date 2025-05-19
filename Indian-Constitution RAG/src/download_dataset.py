from datasets import load_dataset
import json

# Load SQuAD dataset with 500 examples
dataset = load_dataset("squad", split="train[:100]")  # First 500 examples

# Save as JSON Lines format
dataset.to_json("data/squad.jsonl", orient='records', lines=True) 