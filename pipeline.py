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

from load_docs import load_pdfs_from_directory
from retrieval import chunk_documents, adaptive_chunking, retrieve_relevant_docs, process_retrieved_chunks 
from augmentation import create_augmented_prompt
from generation import generate_answer_gpt4

def rag_pipeline_pdf(query, api_key, pdf_directory="./pdfs", k=3):
    """Complete RAG pipeline with PDF support"""
    print(f"\n{'='*70}")
    print(f"RUNNING COMPLETE RAG PIPELINE FOR: '{query}'")
    print(f"{'='*70}")

    pipeline_start = time.time()

    # Global variables needed for the pipeline
    global documents, document_metadata, chunked_docs, doc_mapping, chunk_metadata
    global vectorizer, document_vectors

    # Step 1: Load PDFs if not already loaded
    if 'documents' not in globals() or len(documents) == 0:
        print(f"Loading documents from {pdf_directory}...")
        documents, document_metadata = load_pdfs_from_directory(pdf_directory)

        # Check if any documents were loaded
        if len(documents) == 0:
            return {"error": f"No PDF documents found in {pdf_directory}"}

        # Step 2: Chunk the documents
        chunked_docs, doc_mapping, chunk_metadata = adaptive_chunking(
            documents, document_metadata, default_chunk_size=300, default_overlap=50
        )

        # Step 3: Create document embeddings
        vectorizer = TfidfVectorizer()
        document_vectors = vectorizer.fit_transform(chunked_docs)
        print(f"Created document embeddings with shape {document_vectors.shape}")

    # Step 4: Retrieve relevant chunks
    retrieved_docs, retrieval_time = retrieve_relevant_docs(query, k=k*2)

    # Step 5: Process and merge chunks from the same document
    processed_docs = process_retrieved_chunks(retrieved_docs, max_chunks_per_doc=2)[:k]
    processing_time = time.time() - pipeline_start - retrieval_time
    print(f"Processed {len(retrieved_docs)} chunks into {len(processed_docs)} document groups")

    # Step 6: Create augmented prompt
    prompt, prompt_time = create_augmented_prompt(query, processed_docs)

    # Step 7: Generate answer
    answer, generation_time = generate_answer_gpt4(prompt, api_key=api_key)

    # Calculate total pipeline time
    total_time = time.time() - pipeline_start

    # Display results
    print(f"\n=== FINAL RESULT ===")
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print(f"\nPerformance Metrics:")
    print(f"- Retrieval time: {retrieval_time:.4f}s")
    print(f"- Processing time: {processing_time:.4f}s")
    print(f"- Prompt creation time: {prompt_time:.4f}s")
    print(f"- Generation time: {generation_time:.4f}s")
    print(f"- Total pipeline time: {total_time:.4f}s")

    # Return complete result
    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "processed_docs": processed_docs,
        "prompt": prompt,
        "answer": answer,
        "metrics": {
            "retrieval_time": retrieval_time,
            "processing_time": processing_time,
            "prompt_time": prompt_time,
            "generation_time": generation_time,
            "total_time": total_time
        }
    }