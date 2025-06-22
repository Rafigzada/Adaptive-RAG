import os
import glob
import shutil
import PyPDF2
import json
from typing import List, Dict, Tuple, Any
import google.generativeai as genai
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset


from src.caching.caching import get_file_hash, load_cache, save_cache, PDF_HASH_CACHE_DIR

PDF_HASH_CACHE_FILE = os.path.join(PDF_HASH_CACHE_DIR, "pdf_hashes.json")


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
    else:
        print(f"Using PDF directory: {os.path.abspath(pdf_dir)}")
    
    return pdf_dir


def list_pdf_files(pdf_dir: str) -> List[str]:
    """List all PDF files in the specified directory"""
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if pdf_files:
        print(f"\nFound {len(pdf_files)} PDF files ready for processing:")
    else:
        print(f"\nNo PDF files found in {pdf_dir}. Please upload some PDFs using the file browser.")
    return pdf_files


def load_pdfs_from_directory(directory_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Load PDFs from the specified directory and extract their text content.
    Provides summarized logging for unchanged files and detailed logging for new/modified ones.
    """
    documents = []
    document_metadata = []
    processed_file_hashes = {} 

    new_modified_count = 0
    existing_count = 0

    previous_hashes = {}
    if os.path.exists(PDF_HASH_CACHE_FILE):
        try:
            with open(PDF_HASH_CACHE_FILE, 'r') as f:
                previous_hashes = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load or parse PDF hash cache from {PDF_HASH_CACHE_FILE}. Starting fresh.")
            previous_hashes = {}
    
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {directory_path} to load.")
        try:
            with open(PDF_HASH_CACHE_FILE, 'w') as f:
                json.dump({}, f, indent=4)
        except Exception as e:
            print(f"Error clearing PDF hash cache when no files found: {e}")
        return [], []

    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        current_file_hash = get_file_hash(file_path) 

        if filename not in previous_hashes or previous_hashes[filename] != current_file_hash:
            print(f"Loading NEW/MODIFIED PDF: {file_path}")
            new_modified_count += 1
        else:
            existing_count += 1

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:  
                        text += page_text + "\n"

                documents.append(text)
                document_metadata.append({
                    "source": file_path,
                    "title": os.path.basename(file_path),
                    "length": len(text),
                    "pages": len(pdf_reader.pages)
                })

                processed_file_hashes[filename] = current_file_hash

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            if filename in processed_file_hashes:
                del processed_file_hashes[filename] 

    summary_message_parts = []
    if new_modified_count > 0:
        summary_message_parts.append(f"{new_modified_count} new/modified PDF file{'s' if new_modified_count > 1 else ''}")
    if existing_count > 0:
        summary_message_parts.append(f"{existing_count} existing PDF file{'s' if existing_count > 1 else ''}")

    if summary_message_parts:
        print(f"Summary: Loaded {' and '.join(summary_message_parts)}.")
    else:
        print("Summary: No PDF files were loaded successfully.")

    files_currently_in_dir_set = {os.path.basename(f) for f in pdf_files}
    files_to_remove_from_cache = [f_name for f_name in previous_hashes if f_name not in files_currently_in_dir_set]

    final_hashes_to_save = {**previous_hashes, **processed_file_hashes}
    for f_name in files_to_remove_from_cache:
        if f_name in final_hashes_to_save:
            del final_hashes_to_save[f_name]

    # Save the updated hashes back to the cache (still direct JSON for this specific case)
    try:
        with open(PDF_HASH_CACHE_FILE, 'w') as f:
            json.dump(final_hashes_to_save, f, indent=4)
    except Exception as e:
        print(f"Error saving PDF hash cache: {e}")

    return documents, document_metadata


def chunk_documents(documents: List[str], chunk_size: int = 100, overlap: int = 20) -> Tuple[List[str], List[int]]:
    """Split longer documents into smaller chunks with overlap"""
    chunked_docs = []
    doc_mapping = []  

    for doc_idx, doc in enumerate(documents):
        if len(doc) <= chunk_size:
            chunked_docs.append(doc)
            doc_mapping.append(doc_idx)
        else:
            for i in range(0, len(doc), chunk_size - overlap):
                chunk = doc[i:i + chunk_size]
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
        if metadata["length"] > very_large_doc_threshold:
            chunk_size = 800
            overlap = 100
        elif metadata["length"] > large_doc_threshold:
            chunk_size = 500
            overlap = 75
        else:
            chunk_size = default_chunk_size
            overlap = default_overlap

        if len(doc) <= chunk_size:
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
            chunks = []
            chunk_start_indices = list(range(0, len(doc), chunk_size - overlap))

            for i, start_idx in enumerate(chunk_start_indices):
                end_idx = min(start_idx + chunk_size, len(doc))
                chunk = doc[start_idx:end_idx]

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

    top_indices = np.argsort(similarity_scores)[::-1][:top_n_summaries]

    relevant_docs_info = []
    for idx in top_indices:
        score = similarity_scores[idx]
        if score > 0: 
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