import os
import glob
import shutil
import PyPDF2
from typing import List, Dict, Tuple


def create_pdf_directory(pdf_dir: str = "./pdfs") -> str:
    """Create PDF directory and move any PDFs from root directory"""
    os.makedirs(pdf_dir, exist_ok=True)
    print(f"PDF directory created at: {os.path.abspath(pdf_dir)}")
    
    # Check for PDFs in current directory that need to be moved
    pdf_files_in_root = glob.glob("*.pdf")
    if pdf_files_in_root:
        print(f"Found {len(pdf_files_in_root)} PDFs in root directory, moving to {pdf_dir}...")
        for pdf in pdf_files_in_root:
            destination = os.path.join(pdf_dir, pdf)
            shutil.move(pdf, destination)
            print(f"  - Moved {pdf}")
    
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
