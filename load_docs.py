import os
import glob
import shutil
import PyPDF2
from typing import List, Dict, Tuple, Any
import google.generativeai as genai
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


def pdf_directory(pdf_dir: str = "./pdfs") -> str:
    """
    Checks if the PDF directory exists and prints a message if not.
    It does NOT create the directory, expecting the user to provide it.
    """
    if not os.path.exists(pdf_dir):
        print(f"Warning: The specified PDF directory '{os.path.abspath(pdf_dir)}' does not exist.")
        print("Please create this directory and place your PDF files inside it.")
        # We won't exit here; let the main script check for actual files and handle exiting.
    else:
        print(f"Using PDF directory: {os.path.abspath(pdf_dir)}")
    
    return pdf_dir


def list_pdf_files(pdf_dir: str) -> List[str]:
    """List all PDF files in the specified directory"""
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if pdf_files:
        print(f"\nFound {len(pdf_files)} PDF files ready for processing:")
        for pdf in pdf_files:
            print(f"  - {os.path.basename(pdf)}")
    else:
        print(f"\nNo PDF files found in {pdf_dir}. Please upload some PDFs using the file browser.")
    return pdf_files


def load_pdfs_from_directory(directory_path: str) -> Tuple[List[str], List[Dict]]:
    """Load PDFs from the specified directory and extract their text content"""
    documents = []
    document_metadata = []

    # Find all PDF files in the directory
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))

    for file_path in pdf_files:
        try:
            with open(file_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)

                # Extract text from each page and combine
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:  # Check if text extraction was successful
                        text += page_text + "\n"

                # Store document and its metadata
                documents.append(text)
                document_metadata.append({
                    "source": file_path,
                    "title": os.path.basename(file_path),
                    "length": len(text),
                    "pages": len(pdf_reader.pages)
                })

                print(f"Loaded PDF: {file_path} ({len(pdf_reader.pages)} pages, {len(text)} characters)")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents, document_metadata


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
                      default_chunk_size: int = 500,
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


def find_relevant_documents(
    query: str,
    document_summaries: List[Dict],
    summary_vectorizer: TfidfVectorizer,
    summary_vectors: Any,
    top_n_summaries: int = 5
) -> List[Dict]:
    """
    Finds relevant documents based on TF-IDF similarity to summaries.
    Returns a list of dictionaries with doc_idx, summary, and similarity score.
    """
    if not document_summaries:
        print("No document summaries available for pre-filtering.")
        return []

    print("\n--- Pre-filtering Documents (by Summary TF-IDF) ---")
    query_vector = summary_vectorizer.transform([query])
    similarity_scores = summary_vectors.dot(query_vector.T).toarray().flatten()

    # Get top_n_summaries indices
    top_indices = np.argsort(similarity_scores)[::-1][:top_n_summaries]

    relevant_docs_info = []
    for idx in top_indices:
        score = similarity_scores[idx]
        if score > 0: # Only include if there's some similarity
            relevant_docs_info.append({
                "doc_idx": document_summaries[idx]["doc_idx"],
                "summary": document_summaries[idx]["summary"],
                "original_source": document_summaries[idx]["original_source"],
                "original_title": document_summaries[idx]["original_title"],
                "similarity_score": score
            })
            print(f"  - Document {document_summaries[idx]['doc_idx']} ('{document_summaries[idx]['original_title']}') - Score: {score:.4f}")
    
    if not relevant_docs_info:
        print("  - No relevant documents found based on summary similarity.")
    else:
        print(f"Found {len(relevant_docs_info)} documents relevant based on summaries.")

    return relevant_docs_info
