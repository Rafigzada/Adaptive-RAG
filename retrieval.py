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

def chunk_documents(documents, chunk_size=100, overlap=20):
    """Split longer documents into smaller chunks with overlap"""
    chunked_docs = []
    doc_mapping = []  # To track which chunk belongs to which original document

    for doc_idx, doc in enumerate(documents):
        if len(doc) <= chunk_size:
            # If document is already small enough, keep it as is
            chunked_docs.append(doc)
            doc_mapping.append(doc_idx)
        else:
            # Split into overlapping chunks
            for i in range(0, len(doc), chunk_size - overlap):
                chunk = doc[i:i + chunk_size]
                # Only add chunk if it's substantial (at least half the chunk size)
                if len(chunk) >= chunk_size // 2:
                    chunked_docs.append(chunk)
                    doc_mapping.append(doc_idx)

    print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs, doc_mapping


def adaptive_chunking(documents, document_metadata,
                      default_chunk_size=300,
                      default_overlap=50,
                      large_doc_threshold=10000,
                      very_large_doc_threshold=50000):
    """Apply different chunking strategies based on document size"""
    chunked_docs = []
    doc_mapping = []
    chunk_metadata = []

    for doc_idx, (doc, metadata) in enumerate(zip(documents, document_metadata)):
        # Choose chunk size based on document length
        if metadata["length"] > very_large_doc_threshold:
            # For very large documents, use larger chunks with less overlap
            chunk_size = 800
            overlap = 100
        elif metadata["length"] > large_doc_threshold:
            # For moderately large documents
            chunk_size = 500
            overlap = 75
        else:
            # For smaller documents
            chunk_size = default_chunk_size
            overlap = default_overlap

        # Apply chunking
        if len(doc) <= chunk_size:
            # If document is small enough, keep it whole
            chunked_docs.append(doc)
            doc_mapping.append(doc_idx)
            chunk_metadata.append({
                "original_doc_idx": doc_idx,
                "chunk_idx": 0,
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "is_full_doc": True
            })
        else:
            # Split into chunks - using a sentence-aware approach when possible
            # by trying to split at paragraph or sentence boundaries
            chunks = []
            chunk_start_indices = list(range(0, len(doc), chunk_size - overlap))

            for i, start_idx in enumerate(chunk_start_indices):
                end_idx = min(start_idx + chunk_size, len(doc))
                chunk = doc[start_idx:end_idx]

                # Only add chunk if it's substantial (at least half the chunk size)
                if len(chunk) >= chunk_size // 2:
                    chunked_docs.append(chunk)
                    doc_mapping.append(doc_idx)
                    chunk_metadata.append({
                        "original_doc_idx": doc_idx,
                        "chunk_idx": i,
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", ""),
                        "is_full_doc": False,
                        "chunk_start": start_idx,
                        "chunk_end": end_idx
                    })

    print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs, doc_mapping, chunk_metadata

def retrieve_relevant_docs(query, k=3):
    """Retrieve the k most relevant document chunks for a query using TF-IDF similarity"""
    print(f"\n=== RETRIEVAL PHASE for: '{query}' ===")
    start_time = time.time()

    # Convert query to vector
    query_vector = vectorizer.transform([query])

    # Calculate similarity with all document chunks
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Get top k document indices
    top_indices = similarities.argsort()[-k:][::-1]

    # Create result list
    results = []
    for i, idx in enumerate(top_indices):
        # Add chunk metadata if available
        metadata = chunk_metadata[idx]
        source_title = metadata.get("title", f"Document {doc_mapping[idx] + 1}")
        source_doc = metadata.get("source", "")

        results.append({
            "index": idx,
            "score": float(similarities[idx]),
            "content": chunked_docs[idx],
            "original_doc_idx": doc_mapping[idx],
            "metadata": metadata
        })
        print(f"Retrieved chunk {i+1} (score: {similarities[idx]:.4f}):")
        print(f"- From: {source_title}")
        print(f"- Content preview: {chunked_docs[idx][:150]}...")

    retrieval_time = time.time() - start_time
    print(f"Retrieval completed in {retrieval_time:.4f} seconds")

    return results, retrieval_time

def process_retrieved_chunks(retrieved_chunks, max_chunks_per_doc=2):
    """Merge chunks from the same document and deduplicate results"""
    processed_results = []
    doc_chunks = {}

    # Group chunks by original document
    for chunk in retrieved_chunks:
        orig_doc_idx = chunk["original_doc_idx"]
        if orig_doc_idx not in doc_chunks:
            doc_chunks[orig_doc_idx] = []
        doc_chunks[orig_doc_idx].append(chunk)

    # For each document, select the best chunks
    for doc_idx, chunks in doc_chunks.items():
        # Sort chunks by score
        sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)

        # Take top chunks based on limit
        top_chunks = sorted_chunks[:max_chunks_per_doc]

        # Create merged content
        content = "\n[...]\n".join([chunk["content"] for chunk in top_chunks])

        # Add processed result
        processed_results.append({
            "index": top_chunks[0]["index"],  # Use index of highest scoring chunk
            "score": top_chunks[0]["score"],  # Use highest score
            "content": content,
            "original_doc_idx": doc_idx,
            "metadata": top_chunks[0].get("metadata", {}),
            "num_chunks_merged": len(top_chunks)
        })

    # Sort final results by score
    return sorted(processed_results, key=lambda x: x["score"], reverse=True)
