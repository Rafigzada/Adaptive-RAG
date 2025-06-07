import time
import sys
import os
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

# Add current directory to Python path to find local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from updated modules
from load_docs import load_pdfs_from_directory, adaptive_chunking, find_relevant_documents
from retrieval import DocumentRetriever
from augmentation import create_augmented_prompt, create_chat_messages, format_context_with_citations
from generation import AnswerGenerator
from summarization import DocumentSummarizer # NEW IMPORT
from ranking import DocumentReRanker # NEW IMPORT


class RAGPipeline:
    def __init__(self, api_key: str, answer_model_name: str = 'gemini-1.5-flash'):
        self.retriever = DocumentRetriever()
        self.generator = AnswerGenerator(api_key, answer_model_name)
        self.api_key = api_key
        self.documents = []
        self.document_metadata = []
        self.document_summaries = []
        self.summary_vectorizer = None
        self.summary_vectors = None
        self.is_initialized = False

        # Initialize the new summarizer and re-ranker modules
        self.document_summarizer = DocumentSummarizer(api_key, answer_model_name) # NEW
        self.document_reranker = DocumentReRanker(api_key, answer_model_name)     # NEW

    def initialize_documents(self, pdf_directory: str = "./pdfs"):
        """Initialize the pipeline by loading, processing, and summarizing documents"""
        print(f"\n--- Initializing RAG Pipeline ---")
        print(f"Loading documents from {pdf_directory}...")
        self.documents, self.document_metadata = load_pdfs_from_directory(pdf_directory)

        if len(self.documents) == 0:
            raise ValueError(f"No PDF documents found in {pdf_directory}. Please ensure files are in '{os.path.abspath(pdf_directory)}'")

        # Step 1: Generate summaries for all documents using the new DocumentSummarizer
        self.document_summaries = self.document_summarizer.generate_summaries( # UPDATED CALL
            self.documents, self.document_metadata
        )

        # New Step: Initialize and fit TF-IDF vectorizer for summaries ONCE
        if self.document_summaries:
            summary_texts = [doc_summary["summary"] for doc_summary in self.document_summaries]
            self.summary_vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.summary_vectors = self.summary_vectorizer.fit_transform(summary_texts)
            print(f"\nFitted summary TF-IDF vectorizer (with stop words, n-grams). \nSummary vectors shape: {self.summary_vectors.shape}")
        else:
            print("No summaries generated, skipping summary TF-IDF vectorizer setup.")

        # Step 2: Chunk ALL documents initially
        chunked_docs, doc_mapping, chunk_metadata = adaptive_chunking(
            self.documents, self.document_metadata,
            default_chunk_size=500, default_overlap=50
        )

        # Step 3: Create embeddings for all chunks using the DocumentRetriever's TF-IDF
        self.retriever.create_embeddings(chunked_docs, doc_mapping, chunk_metadata)

        self.is_initialized = True
        print("\n--- RAG Pipeline Initialization Completed! ---")

    # Removed the static re_rank_retrieved_documents method from here

    def query(self, query: str, k: int = 3, use_chat_format: bool = True, rerank_top_k: int = 3) -> Dict[str, Any]:
        """Process a single query through the complete RAG pipeline"""
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize_documents() first.")

        print(f"\n{'='*70}")
        print(f"RUNNING COMPLETE RAG PIPELINE FOR: '{query}'")
        print(f"{'='*70}")

        pipeline_start = time.time()

        # Step 1: Pre-filter documents based on query and summaries using the stored TF-IDF
        pre_filtered_relevant_docs_info = find_relevant_documents(
            query, self.document_summaries, self.summary_vectorizer, self.summary_vectors, top_n_summaries=5
        )
        pre_filter_time = time.time() - pipeline_start

        pre_filtered_doc_indices = {doc_info['doc_idx'] for doc_info in pre_filtered_relevant_docs_info}

        # Step 2: Retrieve relevant chunks from the main chunk index.
        retrieved_chunks_all, retrieval_time_full = self.retriever.retrieve_relevant_docs(query, k=k*5)

        retrieved_docs_from_prefiltered = [
            chunk_data for chunk_data in retrieved_chunks_all
            if chunk_data.get("metadata", {}).get("original_doc_idx") in pre_filtered_doc_indices
        ]

        if not retrieved_docs_from_prefiltered:
            print("  - No chunks found from pre-filtered documents. Falling back to top-K from all documents.")
            retrieved_docs_for_processing = retrieved_chunks_all[:k*2]
        else:
            retrieved_docs_for_processing = retrieved_docs_from_prefiltered


        # Step 3: Process and merge chunks from the same document (from the filtered list)
        processed_docs_for_rerank = self.retriever.process_retrieved_chunks(
            retrieved_docs_for_processing, max_chunks_per_doc=2
        )
        processed_docs_for_rerank = processed_docs_for_rerank[:15] # Limit for re-ranking efficiency
        processing_time = (time.time() - pipeline_start) - pre_filter_time - retrieval_time_full
        print("---------------------------------------------------------------")
        print(f"\nProcess and merge chunks from the same document (from the filtered list).")
        print(f"Processed {len(retrieved_docs_for_processing)} chunks into {len(processed_docs_for_rerank)} document groups for re-ranking.")

        # --- Print: Document Groups for Re-ranking ---
        print("\n--- Document Groups Prepared for Re-ranking ---")
        if processed_docs_for_rerank:
            for i, doc_group in enumerate(processed_docs_for_rerank):
                print(f"  Group {i+1}:")
                print(f"    Title: {doc_group.get('metadata', {}).get('title', 'N/A')}")
                print(f"    Source: {os.path.basename(doc_group.get('metadata', {}).get('source', 'N/A'))}")
                print(f"    Content Preview: {doc_group.get('content', '')[:200]}...")
        else:
            print("  No document groups prepared for re-ranking.")
        print("---------------------------------------------------------------")


        # Step 4: Re-rank the processed documents using the DocumentReRanker instance
        re_ranking_start_time = time.time()
        final_retrieved_docs = self.document_reranker.rerank_documents( # UPDATED CALL
            query, processed_docs_for_rerank, rerank_top_k
        )
        reranking_time = time.time() - re_ranking_start_time
        print(f"Re-ranking completed in {reranking_time:.4f} seconds.")

        # --- Print: Re-ranked Documents (before final context slicing) ---
        print("\n\n--- Re-ranked Documents (before final context slicing) ---")
        if final_retrieved_docs:
            for i, doc in enumerate(final_retrieved_docs):
                print(f"    Rank {i+1} (Rerank Score: {doc.get('rerank_score', 'N/A'):.2f}):")
                print(f"    Title: {doc.get('metadata', {}).get('title', 'N/A')}")
                print(f"    Source: {os.path.basename(doc.get('metadata', {}).get('source', 'N/A'))}")
                print(f"    Content Preview: {doc.get('content', '')[:300]}...")
        else:
            print("  No documents were successfully re-ranked.")
        print("-------------------------------------------------------")


        # Step 5: Create augmented prompt using the re-ranked documents (only take top k as specified by query param)
        final_context_docs = final_retrieved_docs[:k]
        print(f"Using {len(final_context_docs)} documents for final context.")

        # --- Print: Actual Final Context Documents Sent to LLM ---
        print("\n\n--- Actual Final Context Documents Sent to LLM ---")
        if final_context_docs:
            for i, doc in enumerate(final_context_docs):
                print(f"\n  Context Doc {i+1}:")
                print(f"    Title: {doc.get('metadata', {}).get('title', 'N/A')}")
                print(f"    Source: {os.path.basename(doc.get('metadata', {}).get('source', 'N/A'))}")
                print(f"    Content Preview: {doc.get('content', '')[:300]}...")
        else:
            print("  No documents sent as final context to the LLM.")
        print("--------------------------------------------------")

        if use_chat_format:
            context = format_context_with_citations(final_context_docs)
            messages = create_chat_messages(query, context)
            prompt_time = 0.001
            answer, generation_time = self.generator.generate_answer_with_messages(messages)
        else:
            prompt, prompt_time = create_augmented_prompt(query, final_context_docs)
            answer, generation_time = self.generator.generate_answer_gemini(prompt)

        total_time = time.time() - pipeline_start

        print(f"\n=== FINAL RESULT ===")
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print(f"\nPerformance Metrics:")
        print(f"- Pre-filter (Summary TF-IDF) time: {pre_filter_time:.4f}s")
        print(f"- Retrieval (Chunk TF-IDF) time: {retrieval_time_full:.4f}s")
        print(f"- Processing (Merging) time: {processing_time:.4f}s")
        print(f"- Re-ranking time: {reranking_time:.4f}s")
        print(f"- Prompt creation time: {prompt_time:.4f}s")
        print(f"- Generation time: {generation_time:.4f}s")
        print(f"- Total pipeline time: {total_time:.4f}s")

        return {
            "query": query,
            "pre_filtered_docs_info": pre_filtered_relevant_docs_info,
            "retrieved_chunks_raw": retrieved_chunks_all,
            "processed_docs_before_rerank": processed_docs_for_rerank,
            "final_retrieved_docs": final_retrieved_docs,
            "answer": answer,
            "metrics": {
                "pre_filter_time": pre_filter_time,
                "retrieval_time": retrieval_time_full,
                "processing_time": processing_time,
                "reranking_time": reranking_time,
                "prompt_time": prompt_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
        }

    def batch_query(self, queries: List[str], k: int = 3, rerank_top_k: int = 3) -> List[Dict[str, Any]]:
        results = []
        for query in queries:
            result = self.query(query, k=k, rerank_top_k=rerank_top_k)
            results.append(result)
            print(f"\n{'='*70}\n")
        return results


# Convenience function for backwards compatibility (updated)
def rag_pipeline_pdf(query: str, api_key: str, pdf_directory: str = "./pdfs", k: int = 3, rerank_top_k: int = 3) -> Dict[str, Any]:
    """Complete RAG pipeline with PDF support - standalone function"""
    print(f"\n{'='*70}")
    print(f"RUNNING COMPLETE RAG PIPELINE FOR: '{query}' (Standalone Function)")
    print(f"{'='*70}")

    pipeline_start = time.time()

    # Initialize components
    retriever = DocumentRetriever()
    generator = AnswerGenerator(api_key)
    summarizer = DocumentSummarizer(api_key) # NEW
    reranker = DocumentReRanker(api_key)     # NEW


    # Step 1: Load PDFs
    print(f"Loading documents from {pdf_directory}...")
    documents, document_metadata = load_pdfs_from_directory(pdf_directory)

    if len(documents) == 0:
        return {"error": f"No PDF documents found in {pdf_directory}"}

    # Step 2: Generate summaries for all documents using the Summarizer
    summarization_start_time = time.time()
    document_summaries = summarizer.generate_summaries(documents, document_metadata) # UPDATED CALL
    summarization_time = time.time() - summarization_start_time

    summary_vectorizer = None
    summary_vectors = None
    if document_summaries:
        summary_texts = [doc_summary["summary"] for doc_summary in document_summaries]
        summary_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        summary_vectors = summary_vectorizer.fit_transform(summary_texts)
        print(f"Fitted summary TF-IDF vectorizer. Summary vectors shape: {summary_vectors.shape}")
    else:
        print("No summaries generated, skipping summary TF-IDF vectorizer setup.")


    # Step 3: Chunk ALL documents
    chunked_docs, doc_mapping, chunk_metadata = adaptive_chunking(
        documents, document_metadata, default_chunk_size=300, default_overlap=50
    )

    # Step 4: Create document embeddings for chunks using DocumentRetriever's TF-IDF
    retriever.create_embeddings(chunked_docs, doc_mapping, chunk_metadata)
    embedding_creation_time = time.time() - pipeline_start - summarization_time

    # Step 5: Pre-filter documents based on query and summaries using the fitted TF-IDF
    pre_filtered_relevant_docs_info = find_relevant_documents(
        query, document_summaries, summary_vectorizer, summary_vectors, top_n_summaries=5
    )
    pre_filter_time = time.time() - pipeline_start - summarization_time - embedding_creation_time
    pre_filtered_doc_indices = {doc_info['doc_idx'] for doc_info in pre_filtered_relevant_docs_info}


    # Step 6: Retrieve relevant chunks from the main chunk index
    retrieved_chunks_all, retrieval_time_full = retriever.retrieve_relevant_docs(query, k=k*10)

    retrieved_docs_from_prefiltered = [
        chunk_data for chunk_data in retrieved_chunks_all
        if chunk_data.get("metadata", {}).get("original_doc_idx") in pre_filtered_doc_indices
    ]

    if not retrieved_docs_from_prefiltered:
        print("  - No chunks found from pre-filtered documents. Falling back to top-K from all documents.")
        retrieved_docs_for_processing = retrieved_chunks_all[:k*2]
    else:
        retrieved_docs_for_processing = retrieved_docs_from_prefiltered


    # Step 7: Process and merge chunks from the same document
    processed_docs_for_rerank = retriever.process_retrieved_chunks(retrieved_docs_for_processing, max_chunks_per_doc=2)
    processed_docs_for_rerank = processed_docs_for_rerank[:15]
    processing_time = time.time() - pipeline_start - summarization_time - embedding_creation_time - pre_filter_time - retrieval_time_full
    print(f"Processed {len(retrieved_docs_for_processing)} chunks into {len(processed_docs_for_rerank)} document groups for re-ranking.")

    # --- Print: Document Groups for Re-ranking (for standalone function) ---
    print("\n--- Document Groups Prepared for Re-ranking (Standalone Function) ---")
    if processed_docs_for_rerank:
        for i, doc_group in enumerate(processed_docs_for_rerank):
            print(f"  Group {i+1}:")
            print(f"    Title: {doc_group.get('metadata', {}).get('title', 'N/A')}")
            print(f"    Source: {os.path.basename(doc_group.get('metadata', {}).get('source', 'N/A'))}")
            print(f"    Content Preview: {doc_group.get('content', '')[:200]}...")
    else:
        print("  No document groups prepared for re-ranking.")
    print("---------------------------------------------")


    # Step 8: Re-rank the processed documents using the ReRanker instance
    re_ranking_start_time = time.time()
    final_retrieved_docs = reranker.rerank_documents( # UPDATED CALL
        query, processed_docs_for_rerank, rerank_top_k
    )
    reranking_time = time.time() - re_ranking_start_time
    print(f"Re-ranking completed in {reranking_time:.4f} seconds.")

    # --- Print: Re-ranked Documents (before final context slicing - for standalone function) ---
    print("\n--- Re-ranked Documents (before final context slicing - Standalone Function) ---")
    if final_retrieved_docs:
        for i, doc in enumerate(final_retrieved_docs):
            print(f"\n  Rank {i+1} (Rerank Score: {doc.get('rerank_score', 'N/A'):.2f}):")
            print(f"    Title: {doc.get('metadata', {}).get('title', 'N/A')}")
            print(f"    Source: {os.path.basename(doc.get('metadata', {}).get('source', 'N/A'))}")
            print(f"    Content Preview: {doc.get('content', '')[:300]}...")
    else:
        print("  No documents were successfully re-ranked.")
    print("-------------------------------------------------------")


    # Step 9: Create augmented prompt using the re-ranked documents
    final_context_docs = final_retrieved_docs[:k]
    print(f"Using {len(final_context_docs)} documents for final context.")

    # --- Print: Actual Final Context Documents Sent to LLM (for standalone function) ---
    print("\n--- Actual Final Context Documents Sent to LLM (Standalone Function) ---")
    if final_context_docs:
        for i, doc in enumerate(final_context_docs):
            print(f"\n  Context Doc {i+1}:")
            print(f"    Title: {doc.get('metadata', {}).get('title', 'N/A')}")
            print(f"    Source: {os.path.basename(doc.get('metadata', {}).get('source', 'N/A'))}")
            print(f"    Content Preview: {doc.get('content', '')[:300]}...")
    else:
        print("  No documents sent as final context to the LLM.")
    print("--------------------------------------------------")


    prompt, prompt_time = create_augmented_prompt(query, final_context_docs)

    # Step 10: Generate answer
    answer, generation_time = generator.generate_answer_gemini(prompt)

    total_time = time.time() - pipeline_start

    print(f"\n=== FINAL RESULT ===")
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print(f"\nPerformance Metrics:")
    print(f"- Document Loading & Summarization time: {summarization_time:.4f}s")
    print(f"- Embedding Creation time (Chunks): {embedding_creation_time:.4f}s")
    print(f"- Pre-filter (Summary TF-IDF) time: {pre_filter_time:.4f}s")
    print(f"- Retrieval (Chunk TF-IDF) time: {retrieval_time_full:.4f}s")
    print(f"- Processing (Merging) time: {processing_time:.4f}s")
    print(f"- Re-ranking time: {reranking_time:.4f}s")
    print(f"- Prompt creation time: {prompt_time:.4f}s")
    print(f"- Generation time: {generation_time:.4f}s")
    print(f"- Total pipeline time: {total_time:.4f}s")

    return {
        "query": query,
        "pre_filtered_docs_info": pre_filtered_relevant_docs_info,
        "retrieved_chunks_raw": retrieved_chunks_all,
        "processed_docs_before_rerank": processed_docs_for_rerank,
        "final_retrieved_docs": final_retrieved_docs,
        "prompt": prompt,
        "answer": answer,
        "metrics": {
            "summarization_time": summarization_time,
            "embedding_creation_time": embedding_creation_time,
            "pre_filter_time": pre_filter_time,
            "retrieval_time": retrieval_time_full,
            "processing_time": processing_time,
            "reranking_time": reranking_time,
            "prompt_time": prompt_time,
            "generation_time": generation_time,
            "total_time": total_time
        }
    }