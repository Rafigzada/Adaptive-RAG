import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, Markdown
import PyPDF2

def create_augmented_prompt(query, retrieved_docs):
    """Create a prompt with retrieved context and citations"""
    print(f"\n=== CONTEXT INTEGRATION PHASE ===")
    start_time = time.time()

    # Format context with citations, noting original document sources
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("title", f"Document {i+1}")

        context_parts.append(f"[{i+1}] From {source}:\n{doc['content']}")

    context = "\n\n".join(context_parts)

    # Create structured prompt with clear instructions
    prompt = f"""Question: {query}

Context information:
{context}

Answer the question based on the context information provided above. Include citation numbers [1], [2], etc. when referencing specific information from the context.
If the context doesn't contain enough information to fully answer the question, acknowledge this limitation in your response.
Present your answer in a clear and concise manner.
"""

    prompt_time = time.time() - start_time
    print(f"Created prompt with {len(retrieved_docs)} chunks in {prompt_time:.4f} seconds")
    print(f"Prompt length: {len(prompt)} characters")

    return prompt, prompt_time