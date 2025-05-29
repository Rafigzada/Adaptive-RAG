import time
import sys
import os
from typing import Dict, List, Any

# Add current directory to Python path to find local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_docs import load_pdfs_from_directory, adaptive_chunking
from retrieval import DocumentRetriever
from augmentation import create_augmented_prompt, create_chat_messages, format_context_with_citations
from generation import AnswerGenerator

class RAGPipeline:
    def __init__(self, api_key: str):
        self.retriever = DocumentRetriever()
        self.generator = AnswerGenerator(api_key)
        self.documents = []
        self.document_metadata = []
        self.is_initialized = False
    
    def initialize_documents(self, pdf_directory: str = "./pdfs"):
        """Initialize the pipeline by loading and processing documents"""
        print(f"Loading documents from {pdf_directory}...")
        self.documents, self.document_metadata = load_pdfs_from_directory(pdf_directory)
        
        if len(self.documents) == 0:
            raise ValueError(f"No PDF documents found in {pdf_directory}")
        
        # Chunk the documents
        chunked_docs, doc_mapping, chunk_metadata = adaptive_chunking(
            self.documents, self.document_metadata, 
            default_chunk_size=300, default_overlap=50
        )
        
        # Create embeddings
        self.retriever.create_embeddings(chunked_docs, doc_mapping, chunk_metadata)
        self.is_initialized = True
        print("Pipeline initialization completed!")
    
    def query(self, query: str, k: int = 3, use_chat_format: bool = True) -> Dict[str, Any]:
        """Process a single query through the complete RAG pipeline"""
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize_documents() first.")
        
        print(f"\n{'='*70}")
        print(f"RUNNING COMPLETE RAG PIPELINE FOR: '{query}'")
        print(f"{'='*70}")
        
        pipeline_start = time.time()
        
        # Step 1: Retrieve relevant chunks
        retrieved_docs, retrieval_time = self.retriever.retrieve_relevant_docs(query, k=k*2)
        
        # Step 2: Process and merge chunks from the same document
        processed_docs = self.retriever.process_retrieved_chunks(retrieved_docs, max_chunks_per_doc=2)[:k]
        processing_time = time.time() - pipeline_start - retrieval_time
        print(f"Processed {len(retrieved_docs)} chunks into {len(processed_docs)} document groups")
        
        # Step 3: Create augmented prompt
        if use_chat_format:
            context = format_context_with_citations(processed_docs)
            messages = create_chat_messages(query, context)
            prompt_time = 0.001  # Minimal time for message formatting
            
            # Step 4: Generate answer using chat format
            answer, generation_time = self.generator.generate_answer_with_messages(messages)
        else:
            prompt, prompt_time = create_augmented_prompt(query, processed_docs)
            
            # Step 4: Generate answer using simple prompt
            answer, generation_time = self.generator.generate_answer_gpt4(prompt)
        
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
            "answer": answer,
            "metrics": {
                "retrieval_time": retrieval_time,
                "processing_time": processing_time,
                "prompt_time": prompt_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
        }
    
    def batch_query(self, queries: List[str], k: int = 3) -> List[Dict[str, Any]]:
        """Process multiple queries through the pipeline"""
        results = []
        for query in queries:
            result = self.query(query, k=k)
            results.append(result)
            print(f"\n{'='*70}\n")
        return results


# Convenience function for backwards compatibility
def rag_pipeline_pdf(query: str, api_key: str, pdf_directory: str = "./pdfs", k: int = 3) -> Dict[str, Any]:
    """Complete RAG pipeline with PDF support - standalone function"""
    print(f"\n{'='*70}")
    print(f"RUNNING COMPLETE RAG PIPELINE FOR: '{query}'")
    print(f"{'='*70}")

    pipeline_start = time.time()

    # Initialize components
    retriever = DocumentRetriever()
    generator = AnswerGenerator(api_key)

    # Step 1: Load PDFs
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
    retriever.create_embeddings(chunked_docs, doc_mapping, chunk_metadata)

    # Step 4: Retrieve relevant chunks
    retrieved_docs, retrieval_time = retriever.retrieve_relevant_docs(query, k=k*2)

    # Step 5: Process and merge chunks from the same document
    processed_docs = retriever.process_retrieved_chunks(retrieved_docs, max_chunks_per_doc=2)[:k]
    processing_time = time.time() - pipeline_start - retrieval_time
    print(f"Processed {len(retrieved_docs)} chunks into {len(processed_docs)} document groups")

    # Step 6: Create augmented prompt
    prompt, prompt_time = create_augmented_prompt(query, processed_docs)

    # Step 7: Generate answer
    answer, generation_time = generator.generate_answer_gpt4(prompt)

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