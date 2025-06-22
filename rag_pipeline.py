# pipeline.py
import time
import sys
import os
from typing import Dict, List, Any, Tuple
import numpy as np 


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from initializer import CorpusInitializer
from src.caching.caching import clear_rag_cache
from src.retrieval.retrieval import DocumentRetriever
from src.augmentation.augmentation import create_augmented_prompt, create_chat_messages, format_context_with_citations
from src.generation.generation import AnswerGenerator
from src.generation.summarization import DocumentSummarizer
from src.generation.ranking import DocumentReRanker


class RAGPipeline:
    def __init__(self,
                 api_key: str,
                 answer_model_name: str = 'gemini-1.5-flash',
                 full_doc_relevance_threshold: float = 1.5,  
                 summary_relevance_threshold: float = 1.5    
                 ):
        print("Initializing RAGPipeline components...")
        self.retriever = DocumentRetriever()
        self.generator = AnswerGenerator(api_key, answer_model_name)
        self.api_key = api_key

        self.full_doc_relevance_threshold = full_doc_relevance_threshold
        self.summary_relevance_threshold = summary_relevance_threshold

        self.corpus_initializer = CorpusInitializer()
        self.corpus_initializer.retriever_instance = self.retriever

        self.document_summarizer = DocumentSummarizer(self.generator) 
        self.document_reranker = DocumentReRanker(self.generator)   

        self.is_initialized = False
        print("RAGPipeline components initialized.")


    def initialize_wikipedia(self):
        """Initialize the pipeline using Wikipedia articles via CorpusInitializer."""
        print("\n--- Initializing Wikipedia documents and creating embeddings ---")
        init_start = time.time()
        self.corpus_initializer.initialize_wikipedia_corpus(self.retriever)
        self.is_initialized = True
        init_end = time.time()
        print(f"--- Wikipedia initialization complete in {init_end - init_start:.2f} seconds ---")


    def initialize_documents(self, pdf_directory: str = "./pdfs"):
        """Initialize the pipeline by loading, processing, and summarizing documents via CorpusInitializer."""
        print("\n--- Initializing documents and creating embeddings ---")
        init_start = time.time()
        self.corpus_initializer.initialize_pdf_corpus(self.retriever, pdf_directory)
        self.is_initialized = True
        init_end = time.time()
        print(f"--- Document initialization complete in {init_end - init_start:.2f} seconds ---")


    def clear_cache(self):
        """Clears the entire RAG cache using the utility function from caching.py.
            Also includes a basic mechanism to clear ChromaDB collections."""
        print("\n--- Clearing RAG Pipeline Cache ---")
        clear_rag_cache()
        # Clear all ChromaDB collections managed by this retriever
        try:
            # List collections using the client directly
            collections_to_delete = self.retriever.chroma_client.list_collections()
            for collection_info in collections_to_delete:
                print(f"Deleting ChromaDB collection: {collection_info.name}")
                self.retriever.chroma_client.delete_collection(name=collection_info.name)
            print("ChromaDB collections cleared.")
        except Exception as e:
            print(f"Error clearing ChromaDB collections: {e}")
        print("--- Cache Cleared ---")

    def query(self, query: str, k: int = 4, use_chat_format: bool = True, rerank_top_k: int = 4, direct_retrieval_only: bool = False) -> Dict[str, Any]:
        """
        Process a single query through the complete RAG pipeline.
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize_documents() or initialize_wikipedia() first.")

        print(f"\n{'='*70}")
        print(f"RUNNING COMPLETE RAG PIPELINE FOR: '{query}'")
        print(f"{'='*70}")

        pipeline_start = time.time()
        summarization_time = 0.0
        pre_filter_time = 0.0
        
        filtered_relevant_full_docs = []
        filtered_relevant_summaries = []
        
        # This will store the document indices to be used for chunk filtering
        doc_indices_for_chunk_retrieval = set() 
        use_summary_prefilter = False

        # Conditional path for direct retrieval or full pipeline
        if not direct_retrieval_only:
            # Step 1: Initial full document relevance check using ChromaDB
            print("\n--- Initial Full Document Relevance Check (ChromaDB) ---")
            initial_full_doc_retrieve_k = 10
            
            # This returns a list of dicts: [{'content': 'text', 'metadata': {}, 'score': 0.x}]
            retrieved_full_docs_raw, _ = self.retriever.retrieve_relevant_docs(query, k=initial_full_doc_retrieve_k, collection_type="full_doc")

            # Filter full documents by relevance threshold
            for i, doc_info in enumerate(retrieved_full_docs_raw):
                doc_distance = doc_info.get('score', float('inf'))
                doc_text = doc_info.get('content', 'N/A')
                doc_meta = doc_info.get('metadata', {})

                if doc_distance <= self.full_doc_relevance_threshold:
                    filtered_relevant_full_docs.append(doc_info)
            
            print(f"\nRetrieved {len(retrieved_full_docs_raw)} full documents from ChromaDB (raw).")
            print(f"Filtered to {len(filtered_relevant_full_docs)} relevant full documents (distance <= {self.full_doc_relevance_threshold}).")

            # --- Conditional Logic based on filtered_relevant_full_docs count ---
            if len(filtered_relevant_full_docs) >= 5:
                use_summary_prefilter = True
                print(f"\n  **{len(filtered_relevant_full_docs)} relevant full documents found via ChromaDB. Proceeding with Document Summarization and Summary Pre-filter.**")

                summarization_start_time = time.time()
                # Generate summaries first (this will use the rag_cache if available)
                self.corpus_initializer.document_summaries = self.document_summarizer.generate_summaries(
                    self.corpus_initializer.documents, self.corpus_initializer.document_metadata
                )
                summarization_time = time.time() - summarization_start_time
                print(f"  Document summarization completed in {summarization_time:.2f} seconds.")

                # Add summaries to ChromaDB
                self.retriever.create_summary_embeddings(
                    self.corpus_initializer.document_summaries, self.corpus_initializer.doc_identifier
                )

                # Step 2: Summary pre-filtering using ChromaDB
                pre_filter_start_time = time.time()
                initial_summary_retrieve_k = 10
                retrieved_summaries_raw, _ = self.retriever.retrieve_relevant_docs(query, k=initial_summary_retrieve_k, collection_type="summary")
                
                # Filter summaries by relevance threshold
                for i, sum_info in enumerate(retrieved_summaries_raw):
                    sum_distance = sum_info.get('score', float('inf'))
                    sum_text = sum_info.get('content', 'N/A')
                    sum_meta = sum_info.get('metadata', {})

                    if sum_distance <= self.summary_relevance_threshold:
                        filtered_relevant_summaries.append(sum_info)
                
                pre_filter_time = time.time() - pre_filter_start_time
                print(f"  Summary pre-filtering completed in {pre_filter_time:.2f} seconds.")
                
                print(f"Retrieved {len(retrieved_summaries_raw)} summaries from ChromaDB (raw).")
                print(f"Filtered to {len(filtered_relevant_summaries)} relevant summaries (distance <= {self.summary_relevance_threshold}).")

                if filtered_relevant_summaries:
                    filtered_relevant_summaries.sort(key=lambda x: x['score'])
                    top_5_summary_docs = filtered_relevant_summaries[:7] 
                    doc_indices_for_chunk_retrieval = {
                        summary.get("metadata", {}).get("original_doc_idx")
                        for summary in top_5_summary_docs
                        if summary.get("metadata", {}).get("original_doc_idx") is not None
                    }
                    print(f"\n  Summary pre-filter selected {len(doc_indices_for_chunk_retrieval)} unique documents (from top 5 summaries) for chunk retrieval.")
                else:
                    print("  - No sufficiently relevant summaries found for pre-filtering. Falling back to initial full document filter for chunks.")
                    use_summary_prefilter = False # Fallback if summaries yielded nothing after filtering
                    # Fallback to initial full document indices if summary filtering failed
                    doc_indices_for_chunk_retrieval = {
                        doc_info.get("metadata", {}).get("original_doc_idx")
                        for doc_info in filtered_relevant_full_docs
                        if doc_info.get("metadata", {}).get("original_doc_idx") is not None
                    }
                    print(f"  - Using {len(doc_indices_for_chunk_retrieval)} unique document indices from initial full document check for chunk retrieval.")

            else: # Less than 5 relevant full documents found in initial check
                use_summary_prefilter = False
                print("\n  **Fewer than 5 sufficiently relevant full documents found via ChromaDB. Bypassing Document Summarization and Summary Pre-filter.**")
                # In this case, use the indices of the filtered relevant full documents directly for chunk retrieval
                doc_indices_for_chunk_retrieval = {
                    doc_info.get("metadata", {}).get("original_doc_idx")
                    for doc_info in filtered_relevant_full_docs
                    if doc_info.get("metadata", {}).get("original_doc_idx") is not None
                }
                print(f"  - Using {len(doc_indices_for_chunk_retrieval)} unique document indices from initial full document check for chunk retrieval.")

        else: # direct_retrieval_only is True
            print("\n--- Bypassing Full Document Relevance Check and Summarization (direct_retrieval_only=True) ---")
            use_summary_prefilter = False
            # In direct retrieval, we don't have pre-filtered docs to guide, so we retrieve broadly for chunks
            # and let the re-ranker handle it. No specific doc_indices_for_chunk_retrieval from previous steps.
            doc_indices_for_chunk_retrieval = set()
            print("  - Chunk retrieval will not be pre-filtered by document indices in direct_retrieval_only mode.")


        # Step 3: Retrieve relevant chunks from ChromaDB
        retrieval_start_time = time.time()
        initial_retrieve_k = 30

        chunk_retrieval_where_clause = None
        if doc_indices_for_chunk_retrieval:
            chunk_retrieval_where_clause = {"original_doc_idx": {"$in": list(doc_indices_for_chunk_retrieval)}}
            print(f"\n--- Retrieving chunks with filter (original_doc_idx IN {len(doc_indices_for_chunk_retrieval)} docs) ---")
            print(f"  Filtering chunks to documents with indices: {list(doc_indices_for_chunk_retrieval)}")
        else:
            print("\n--- Retrieving chunks without document index filter ---")


        retrieved_chunks_all, _ = self.retriever.retrieve_relevant_docs(
            query, 
            k=initial_retrieve_k, 
            collection_type="chunk",
            where_clause=chunk_retrieval_where_clause 
        )
        retrieval_time_full = time.time() - retrieval_start_time
        print(f"\n--- Retrieved {len(retrieved_chunks_all)} chunks from ChromaDB ---")

        # Step 4: Process and merge chunks (now, retrieved_chunks_all already contains only desired chunks)
        processing_start_time = time.time()
        
        
        if not retrieved_chunks_all:
            print("  - No chunks were retrieved after applying document index filter. Cannot proceed with processing.")
            # Early exit or return empty results if no chunks
            return {
                "query": query,
                "answer": "No relevant documents or chunks found to answer the query.",
                "metrics": {}
            }

        processing_message_prefix = f"Processed {len(retrieved_chunks_all)} chunks (from " + \
                                    ("summary pre-filtered documents" if use_summary_prefilter and doc_indices_for_chunk_retrieval else \
                                     "initial relevant full documents" if doc_indices_for_chunk_retrieval else \
                                     "all relevant chunks retrieved directly") + ") into"

        processed_docs_for_rerank = self.retriever.process_retrieved_chunks(
            retrieved_chunks_all, max_chunks_per_doc=3
        )
        # Ensure we don't send too many to the expensive re-ranker
        processed_docs_for_rerank = processed_docs_for_rerank[:15]
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
                print(f"    Content Preview: {doc_group.get('content', '')[:100]}...")
                print(f"    Length of Content: {len(doc_group.get('content'))}")
        else:
            print("  No document groups prepared for re-ranking.")
        print("---------------------------------------------------------------")


        # Step 5: Re-rank the processed documents using the DocumentReRanker instance
        re_ranking_start_time = time.time()
        final_retrieved_docs = self.document_reranker.rerank_documents(
            query, processed_docs_for_rerank, rerank_top_k
        )
        reranking_time = time.time() - re_ranking_start_time
        print(f"\nRe-ranking completed in {reranking_time:.4f} seconds.")

        # --- Print: Re-ranked Documents (before final context slicing) ---
        print("\n\n--- Re-ranked Documents (before final context slicing) ---")
        if final_retrieved_docs:
            for i, doc in enumerate(final_retrieved_docs):
                print(f"    Rank {i+1} (Rerank Score: {doc.get('rerank_score', 'N/A'):.2f}):")
                print(f"    Title: {doc.get('metadata', {}).get('title', 'N/A')}")
                print(f"    Source: {os.path.basename(doc.get('metadata', {}).get('source', 'N/A'))}")
                print(f"    Content Preview: {doc.get('content', '')[:100]}...")
                print(f"    Length of Content: {len(doc.get('content'))}")
        else:
            print("  No documents were successfully re-ranked.")
        print("-------------------------------------------------------")


        # Step 6: Create augmented prompt using the re-ranked documents (only take top k as specified by query param)
        final_context_docs = final_retrieved_docs[:k]
        print(f"Using {len(final_context_docs)} documents for final context.")

        # --- Print: Actual Final Context Documents Sent to LLM ---
        print("\n\n--- Actual Final Context Documents Sent to LLM ---")
        if final_context_docs:
            for i, doc in enumerate(final_context_docs):
                print(f"\n  Context Doc {i+1}:")
                print(f"    Title: {doc.get('metadata', {}).get('title', 'N/A')}")
                print(f"    Source: {os.path.basename(doc.get('metadata', {}).get('source', 'N/A'))}")
                print(f"    Content Preview: {doc.get('content', '')[:100]}...")
                print(f"    Length of Content: {len(doc.get('content'))}")
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
        initial_full_doc_check_duration = retrieval_start_time - pipeline_start if not direct_retrieval_only else 0.0

        if not direct_retrieval_only:
            print(f"- Initial Full Doc Retrieval & Filter time: {initial_full_doc_check_duration:.4f}s")
            print(f"- Summarization generation time: {summarization_time:.4f}s")
            print(f"- Summary Pre-filter (ChromaDB) time: {pre_filter_time:.4f}s")
        else:
            print("- Initial Full Doc Retrieval & Filter: Bypassed")
            print("- Summarization: Bypassed")
            print("- Summary Pre-filter: Bypassed")

        print(f"- Chunk Retrieval (ChromaDB) time: {retrieval_time_full:.4f}s")
        print(f"- Processing (Merging chunks) time: {processing_time:.4f}s")
        print(f"- Re-ranking time: {reranking_time:.4f}s")
        print(f"- Prompt creation time: {prompt_time:.4f}s")
        print(f"- Generation time: {generation_time:.4f}s")
        print(f"- Total pipeline time: {total_time:.4f}s")

        return {
            "query": query,
            "pre_filtered_docs_info": filtered_relevant_summaries,
            "retrieved_chunks_raw": retrieved_chunks_all, # This now correctly holds the filtered chunks from ChromaDB
            "processed_docs_before_rerank": processed_docs_for_rerank,
            "final_retrieved_docs": final_retrieved_docs,
            "answer": answer,
            "metrics": {
                "initial_full_doc_check_time": initial_full_doc_check_duration,
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

    def batch_query(self, queries: List[str], k: int = 3, rerank_top_k: int = 4, direct_retrieval_only: bool = False) -> List[Dict[str, Any]]:
        """Processes a batch of queries."""
        results = []
        for query in queries:
            result = self.query(query, k=k, rerank_top_k=rerank_top_k, direct_retrieval_only=direct_retrieval_only)
            print(f"\n{'='*70}\n")
            results.append(result)
        return results


def rag_pipeline_pdf(query: str, api_key: str, pdf_directory: str = "./pdfs", k: int = 3, rerank_top_k: int = 4) -> Dict[str, Any]:
    """
    Complete RAG pipeline with PDF support - now using the RAGPipeline class.
    This function acts as a convenient entry point for PDF-based RAG.
    """
    print(f"\n{'='*70}")
    print(f"RUNNING PDF RAG PIPELINE FOR: '{query}' (via RAGPipeline class)")
    print(f"{'='*70}")

    pipeline_function_start = time.time()

    try:
        rag_pipeline_instance = RAGPipeline(api_key=api_key,
                                            full_doc_relevance_threshold=1.5, # Default, can be overridden if needed
                                            summary_relevance_threshold=1.5) # Default, can be overridden if needed


        rag_pipeline_instance.initialize_documents(pdf_directory=pdf_directory)

        result = rag_pipeline_instance.query(query=query, k=k, rerank_top_k=rerank_top_k, direct_retrieval_only=False)

        total_pipeline_function_time = time.time() - pipeline_function_start

        result["metrics"]["total_pdf_pipeline_function_time"] = total_pipeline_function_time

        return result

    except Exception as e:
        print(f"An error occurred during PDF pipeline execution: {e}")
        # Return a dictionary with error information
        return {"error": str(e), "query": query}