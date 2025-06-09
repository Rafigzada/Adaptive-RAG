import os
import glob
import shutil
import PyPDF2
from typing import List, Dict, Tuple
from datasets import load_dataset

def load_wikipedia_articles(limit: int = 50) -> Tuple[List[str], List[Dict]]:
    """
    Loads a sample of Wikipedia articles using HuggingFace Datasets.
    """
    print(f"🔍 Loading {limit} Wikipedia articles...")
    wiki_dataset = load_dataset("wikipedia", "20220301.en", split="train",streaming=True, trust_remote_code=True)

    documents = []
    metadata = []

    for i, entry in enumerate(wiki_dataset):
        if i >= limit:
            break

        text = entry["text"].strip()
        if not text:
            continue

        documents.append(text)
        metadata.append({
            "title": entry["title"],
            "length": len(text),
            "source": "Wikipedia"
        })

    print(f"✅ Loaded {len(documents)} Wikipedia articles (streamed).")
    return documents, metadata


def chunk_documents(documents: List[str], chunk_size: int = 100, overlap: int = 20) -> Tuple[List[str], List[int]]:
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


def adaptive_chunking(documents: List[str], 
                     document_metadata: List[Dict],
                     default_chunk_size: int = 300,
                     default_overlap: int = 50,
                     large_doc_threshold: int = 10000,
                     very_large_doc_threshold: int = 50000) -> Tuple[List[str], List[int], List[Dict]]:
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
