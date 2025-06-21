# pipeline.py
import time
import sys
import os
from typing import Dict, List, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # Explicitly import here for summary vectorizer

# Add current directory to Python path to find local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import CorpusInitializer from initializer.py
from initializer import CorpusInitializer

# Import caching functions from the new caching.py
from caching import load_cache, save_cache, clear_rag_cache, get_vectorizer_params_hash

# Import other existing modules
from load_docs import find_relevant_documents
from retrieval import DocumentRetriever
from augmentation import create_augmented_prompt, create_chat_messages, format_context_with_citations
from generation import AnswerGenerator
from summarization import DocumentSummarizer
from ranking import DocumentReRanker


class RAGPipeline:
    def __init__(self, api_key: str, answer_model_name: str = 'gemini-1.5-flash'):
        self.retriever = DocumentRetriever()
        self.generator = AnswerGenerator(api_key, answer_model_name)
        self.api_key = api_key
        
        # Initialize the CorpusInitializer
        self.corpus_initializer = CorpusInitializer()
        # Pass the retriever instance to CorpusInitializer so it can set the retriever's chunk data
        self.corpus_initializer.retriever_instance = self.retriever 

        self.document_summaries: List[Dict] = [] # These will only be generated if the condition is met
        self.summary_vectorizer: TfidfVectorizer = None
        self.summary_vectors: Any = None # Sparse matrix

        self.is_initialized = False
        self.document_summarizer = DocumentSummarizer(api_key, answer_model_name)
        self.document_reranker = DocumentReRanker(api_key, answer_model_name)

    def initialize_wikipedia(self):
        """Initialize the pipeline using Wikipedia articles via CorpusInitializer."""
        self.corpus_initializer.initialize_wikipedia_corpus(self.retriever)
        self.is_initialized = True

    def initialize_documents(self, pdf_directory: str = "./pdfs"):
        """Initialize the pipeline by loading, processing, and summarizing documents via CorpusInitializer."""
        self.corpus_initializer.initialize_pdf_corpus(self.retriever, pdf_directory)
        self.is_initialized = True
        
    def clear_cache(self):
        """Clears the entire RAG cache using the utility function from caching.py."""
        clear_rag_cache()


    def query(self, query: str, k: int = 3, use_chat_format: bool = True, rerank_top_k: int = 3, direct_retrieval_only: bool = False) -> Dict[str, Any]:
        """
        Process a single query through the complete RAG pipeline.
        
        Args:
            query (str): The user's query.
            k (int): The number of final relevant documents to use for answer generation.
            use_chat_format (bool): Whether to format the prompt as chat messages.
            rerank_top_k (int): The number of documents to keep after re-ranking.
            direct_retrieval_only (bool): If True, bypasses initial full document relevance
                                          check and summarization steps, going directly
                                          to chunk retrieval. Useful for benchmarking where
                                          the initial corpus is known to be relevant (e.g., SQuAD).
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize_documents() or initialize_wikipedia() first.")

        print(f"\n{'='*70}")
        print(f"RUNNING COMPLETE RAG PIPELINE FOR: '{query}'")
        print(f"{'='*70}")

        pipeline_start = time.time()
        summarization_time = 0.0
        pre_filter_time = 0.0
        pre_filtered_relevant_docs_info = [] # Default empty list
        pre_filtered_doc_indices = set() # Default empty set

        # Conditional path for direct retrieval or full pipeline
        if not direct_retrieval_only:
            # Step 1: Initial full document relevance check using TF-IDF (always done)
            print("\n--- Initial Full Document Relevance Check (TF-IDF) ---")
            if self.corpus_initializer.full_doc_vectorizer is None or self.corpus_initializer.full_doc_vectors is None:
                raise RuntimeError("Full document TF-IDF vectorizer not initialized. Please run initialize_documents() or initialize_wikipedia().")
                
            query_vector_full_doc = self.corpus_initializer.full_doc_vectorizer.transform([query])
            full_doc_similarity_scores = cosine_similarity(query_vector_full_doc, self.corpus_initializer.full_doc_vectors).flatten()

            relevant_full_doc_count = np.sum(full_doc_similarity_scores > 1e-9)
            print(f"  Found {relevant_full_doc_count} full documents with positive TF-IDF similarity.")

            top_n_full_docs_for_display = 5
            top_full_doc_indices = np.argsort(full_doc_similarity_scores)[::-1][:top_n_full_docs_for_display]
            if any(full_doc_similarity_scores[idx] > 0 for idx in top_full_doc_indices):
                for rank, idx in enumerate(top_full_doc_indices):
                    score = full_doc_similarity_scores[idx]
                    if score > 0:
                        doc_title = self.corpus_initializer.document_metadata[idx].get("title", f"Document {idx + 1}")
                        doc_source = self.corpus_initializer.document_metadata[idx].get("source", "N/A")
                        print(f"    Rank {rank+1}: Doc '{doc_title}' (Source: {os.path.basename(doc_source)}) - Score: {score:.4f}")
            else:
                print("    - No significant similarity found with full documents using TF-IDF for display.")
            print("--------------------------------------------------")

            use_summary_prefilter = False # Flag to control subsequent steps

            if relevant_full_doc_count < 5:
                print("\n  **Less than 5 relevant full documents found. Bypassing Document Summarization and Summary Pre-filter.**")
            else:
                print("\n  **5 or more relevant full documents found. Proceeding with Document Summarization and Summary Pre-filter.**")
                use_summary_prefilter = True

                summarization_start_time = time.time()
                self.document_summaries = self.document_summarizer.generate_summaries(
                    self.corpus_initializer.documents, self.corpus_initializer.document_metadata
                )
                
                tfidf_params_for_summaries = self.corpus_initializer.get_tfidf_params()
                current_doc_identifier = self.corpus_initializer.get_doc_identifier()
                tfidf_params_hash = self.corpus_initializer.get_tfidf_params_hash()

                cache_source_type = "pdf" if self.corpus_initializer.document_metadata and \
                                             self.corpus_initializer.document_metadata[0].get('source', '').lower().endswith('.pdf') else "wikipedia"
                
                loaded_summary_tfidf = load_cache(
                    cache_name=f"{cache_source_type}_summary_tfidf",
                    sub_dir="vectors",
                    identifier=current_doc_identifier,
                    params_hash=tfidf_params_hash
                )
                
                if loaded_summary_tfidf:
                    self.summary_vectorizer, self.summary_vectors = loaded_summary_tfidf
                    print(f"\nLoaded cached summary TF-IDF vectorizer. Summary vectors shape: {self.summary_vectors.shape}")
                elif self.document_summaries:
                    summary_texts = [doc_summary["summary"] for doc_summary in self.document_summaries]
                    
                    self.summary_vectorizer = TfidfVectorizer(**tfidf_params_for_summaries)
                    self.summary_vectors = self.summary_vectorizer.fit_transform(summary_texts)
                    print(f"\nFitted summary TF-IDF vectorizer. Summary vectors shape: {self.summary_vectors.shape}")
                    
                    save_cache(
                        (self.summary_vectorizer, self.summary_vectors),
                        cache_name=f"{cache_source_type}_summary_tfidf",
                        sub_dir="vectors",
                        identifier=current_doc_identifier,
                        params_hash=tfidf_params_hash
                    )
                else:
                    print("\nNo summaries generated or loaded, summary TF-IDF vectorizer setup skipped.")
                
                summarization_time = time.time() - summarization_start_time

                if self.summary_vectorizer is not None and \
                   self.summary_vectors is not None and \
                   self.summary_vectors.shape[0] > 0 and \
                   self.document_summaries:
                    pre_filter_start_time = time.time()
                    pre_filtered_relevant_docs_info = find_relevant_documents(
                        query, self.document_summaries, self.summary_vectorizer, self.summary_vectors, top_n_summaries=5
                    )
                    pre_filter_time = time.time() - pre_filter_start_time
                    pre_filtered_doc_indices = {doc_info['doc_idx'] for doc_info in pre_filtered_relevant_docs_info}
                else:
                    print("  - Summary vectorizer/vectors or summaries not available. Skipping summary-based pre-filter.")
                    pre_filtered_doc_indices = set()
                    use_summary_prefilter = False # Fallback
        else: # direct_retrieval_only is True
            print("\n--- Bypassing Full Document Relevance Check and Summarization (direct_retrieval_only=True) ---")
            use_summary_prefilter = False # Ensure this is False when bypassing

        # Step 2: Retrieve relevant chunks from the main chunk index
        retrieval_start_time = time.time()
        # When direct_retrieval_only is True, we retrieve a larger pool of chunks
        # as there's no pre-filtering from document summaries.
        retrieve_k_multiplier = 10 if not direct_retrieval_only else 20 # Retrieve more if no pre-filter
        retrieved_chunks_all, _ = self.retriever.retrieve_relevant_docs(query, k=k*retrieve_k_multiplier)
        retrieval_time_full = time.time() - retrieval_start_time

        # Conditional logic for processing retrieved chunks based on the decision
        processing_start_time = time.time()
        if not use_summary_prefilter:
            # If summarization was bypassed (either by logic or direct_retrieval_only),
            # directly use top chunks from overall retrieval
            print("  - Summarization bypassed or direct retrieval. Using top chunks from overall retrieval for processing.")
            retrieved_docs_for_processing = retrieved_chunks_all[:k*5] # Take a larger pool for re-ranking
            processing_message_prefix = f"Processed {len(retrieved_docs_for_processing)} chunks (from overall retrieval) into"
        else:
            # If summarization was used, filter chunks by its results
            retrieved_docs_from_prefiltered = [
                chunk_data for chunk_data in retrieved_chunks_all
                if chunk_data.get("metadata", {}).get("original_doc_idx") in pre_filtered_doc_indices
            ]
            if not retrieved_docs_from_prefiltered:
                print("  - No chunks found from summary-pre-filtered documents. Falling back to top-K from all chunks.")
                retrieved_docs_for_processing = retrieved_chunks_all[:k*2] # Fallback if specific pre-filter yielded no chunks
            else:
                retrieved_docs_for_processing = retrieved_docs_from_prefiltered
            processing_message_prefix = f"Processed {len(retrieved_docs_for_processing)} chunks (after summary pre-filter) into"

        # Step 3: Process and merge chunks from the selected list
        processed_docs_for_rerank = self.retriever.process_retrieved_chunks(
            retrieved_docs_for_processing, max_chunks_per_doc=3
        )
        # Ensure we don't send too many to the expensive re-ranker
        processed_docs_for_rerank = processed_docs_for_rerank[:15] # Limit the number sent to re-ranker
        processing_time = time.time() - processing_start_time
        print("---------------------------------------------------------------")
        print(f"\n{processing_message_prefix} {len(processed_docs_for_rerank)} document groups for re-ranking.")

        # --- Print: Document Groups for Re-ranking ---
        print("\n--- Document Groups Prepared for Re-ranking ---")
        if processed_docs_for_rerank:
            for i, doc_group in enumerate(processed_docs_for_rerank):
                print(f"  Group {i+1}:")
                print(f"    Title: {doc_group.get('metadata', {}).get('title', 'N/A')}")
                print(f"    Source: {os.path.basename(doc_group.get('metadata', {}).get('source', 'N/A'))}")
                print(f"    Content Preview: {doc_group.get('content', '')[:20]}...")
                print(f"    Length of Content: {len(doc_group.get('content'))}")
        else:
            print("  No document groups prepared for re-ranking.")
        print("---------------------------------------------------------------")


        # Step 4: Re-rank the processed documents using the DocumentReRanker instance
        re_ranking_start_time = time.time()
        final_retrieved_docs = self.document_reranker.rerank_documents(
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
                print(f"    Content Preview: {doc.get('content', '')[:30]}...")
                print(f"    Length of Content: {len(doc.get('content'))}")
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
                print(f"    Content Preview: {doc.get('content', '')[:30]}...")
                print(f"    Length of Content: {len(doc.get('content'))}")
        else:
            print("  No documents sent as final context to the LLM.")
        print("--------------------------------------------------")

        if use_chat_format:
            context = format_context_with_citations(final_context_docs)
            messages = create_chat_messages(query, context)
            prompt_time = 0.001 # Minimal time for simple formatting
            answer, generation_time = self.generator.generate_answer_with_messages(messages)
        else:
            prompt, prompt_time = create_augmented_prompt(query, final_context_docs)
            answer, generation_time = self.generator.generate_answer_gemini(prompt)

        total_time = time.time() - pipeline_start

        print(f"\n=== FINAL RESULT ===")
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print(f"\nPerformance Metrics:")
        # Adjust metrics based on whether steps were bypassed
        if not direct_retrieval_only:
            print(f"- Initial Full Doc Relevance Check time: {retrieval_start_time - pipeline_start:.4f}s")
            print(f"- Summarization (if used) time: {summarization_time:.4f}s")
            print(f"- Pre-filter (Summary TF-IDF if used) time: {pre_filter_time:.4f}s")
        else:
            print("- Initial Full Doc Relevance Check: Bypassed")
            print("- Summarization: Bypassed")
            print("- Pre-filter: Bypassed")

        print(f"- Retrieval (Chunk TF-IDF) time: {retrieval_time_full:.4f}s")
        print(f"- Processing (Merging chunks) time: {processing_time:.4f}s")
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
                "initial_full_doc_check_time": (retrieval_start_time - pipeline_start) if not direct_retrieval_only else 0.0,
                "summarization_time": summarization_time,
                "pre_filter_time": pre_filter_time,
                "retrieval_time": retrieval_time_full,
                "processing_time": processing_time,
                "reranking_time": reranking_time,
                "prompt_time": prompt_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
        }

    def batch_query(self, queries: List[str], k: int = 3, rerank_top_k: int = 3, direct_retrieval_only: bool = False) -> List[Dict[str, Any]]:
        """Processes a batch of queries."""
        results = []
        for query in queries:
            result = self.query(query, k=k, rerank_top_k=rerank_top_k, direct_retrieval_only=direct_retrieval_only)
            print(f"\n{'='*70}\n")
            results.append(result)
        return results


def rag_pipeline_pdf(query: str, api_key: str, pdf_directory: str = "./pdfs", k: int = 3, rerank_top_k: int = 3) -> Dict[str, Any]:
    """
    Complete RAG pipeline with PDF support - now using the RAGPipeline class.
    This function acts as a convenient entry point for PDF-based RAG.
    """
    print(f"\n{'='*70}")
    print(f"RUNNING PDF RAG PIPELINE FOR: '{query}' (via RAGPipeline class)")
    print(f"{'='*70}")

    pipeline_function_start = time.time()

    try:
        # Initialize the RAGPipeline instance
        rag_pipeline_instance = RAGPipeline(api_key=api_key)
        
        # Initialize documents specifically for PDFs using the RAGPipeline's method
        rag_pipeline_instance.initialize_documents(pdf_directory=pdf_directory)

        # Run the query through the pipeline (direct_retrieval_only defaults to False for PDFs)
        result = rag_pipeline_instance.query(query=query, k=k, rerank_top_k=rerank_top_k, direct_retrieval_only=False)
        
        total_pipeline_function_time = time.time() - pipeline_function_start
        # Add a metric for the total time of this wrapper function
        result["metrics"]["total_pdf_pipeline_function_time"] = total_pipeline_function_time 

        return result

    except Exception as e:
        print(f"An error occurred during PDF pipeline execution: {e}")
        # Return a dictionary with error information
        return {"error": str(e), "query": query}