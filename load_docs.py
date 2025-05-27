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

print("Starting RAG Demo with GPT-4...")

# Create PDF directory
pdf_dir = "./pdfs"
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


# List all PDFs in the pdf directory
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
if pdf_files:
    print(f"\nFound {len(pdf_files)} PDF files ready for processing:")
    for pdf in pdf_files:
        print(f"  - {os.path.basename(pdf)}")
else:
    print(f"\nNo PDF files found in {pdf_dir}. Please upload some PDFs using the file browser.")


# 1. Document Collection from PDFs
def load_pdfs_from_directory(directory_path):
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