import os
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any


class DocumentRetriever:
    def __init__(self):
        self.vectorizer = None
        self.document_vectors = None
        self.chunked_docs: List[str] = [] # Initialize with type hint
        self.doc_mapping: Dict[int, List[int]] = {} # Initialize with correct type hint
        self.chunk_metadata: List[Dict] = [] # Initialize with type hint
    
    # Corrected method signature to accept tfidf_params
    def create_embeddings(self, 
                          chunked_docs: List[str], 
                          doc_mapping: Dict[int, List[int]], # Corrected type hint
                          chunk_metadata: List[Dict],
                          tfidf_params: Dict[str, Any]): # ADDED THIS ARGUMENT
        """
        Create document embeddings using TF-IDF.
        Accepts tfidf_params to ensure consistent vectorizer configuration.
        """
        if not chunked_docs:
            print("No chunked documents provided to create embeddings.")
            self.chunked_docs = []
            self.doc_mapping = {}
            self.chunk_metadata = []
            self.vectorizer = None
            self.document_vectors = None
            return

        self.chunked_docs = chunked_docs
        self.doc_mapping = doc_mapping
        self.chunk_metadata = chunk_metadata
        
        # Use the provided tfidf_params for vectorizer initialization
        self.vectorizer = TfidfVectorizer(**tfidf_params)
        self.document_vectors = self.vectorizer.fit_transform(chunked_docs)
        print(f"Created document embeddings with shape {self.document_vectors.shape}")
    
    # Corrected return type for retrieve_relevant_docs
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve the k most relevant document chunks for a query using TF-IDF similarity.
        Returns a tuple: (list of dictionaries for relevant chunks, list of their similarity scores).
        """
        print(f"\n=== RETRIEVAL PHASE for: '{query}' ===")
        start_time = time.time()

        if self.vectorizer is None or self.document_vectors is None:
            raise ValueError("Embeddings not created. Call create_embeddings() first.")

        # Convert query to vector
        query_vector = self.vectorizer.transform([query])

        # Calculate similarity with all document chunks
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()

        # Get top k document indices
        # Ensure k does not exceed the number of available documents
        num_docs = self.document_vectors.shape[0]
        actual_k = min(k, num_docs)
        top_indices = similarities.argsort()[-actual_k:][::-1]

        # Create result list
        results = []
        scores = [] # To store individual scores
        for i, idx in enumerate(top_indices):
            # Only include chunks with positive similarity
            if similarities[idx] <= 0:
                continue

            metadata = self.chunk_metadata[idx]
            # Use original_doc_idx from metadata
            original_doc_idx = metadata.get("original_doc_idx", -1) 
            source_title = metadata.get("title", f"Document {original_doc_idx + 1}")
            source_doc = metadata.get("source", "")

            results.append({
                "index": idx,
                "score": float(similarities[idx]), # Ensure score is float
                "content": self.chunked_docs[idx],
                "original_doc_idx": original_doc_idx,
                "metadata": metadata
            })
            scores.append(float(similarities[idx]))

            print(f"Retrieved chunk {i+1} (score: {similarities[idx]:.4f}):")
            print(f"- From: '{source_title}' (Source: {os.path.basename(source_doc)})")
            #print(f"- Content preview: {self.chunked_docs[idx][:150]}...")

        retrieval_time = time.time() - start_time
        print(f"Retrieval completed in {retrieval_time:.4f} seconds")

        return results, scores # Return both results and scores

    def process_retrieved_chunks(self, retrieved_chunks: List[Dict], max_chunks_per_doc: int = 3) -> List[Dict]:
        """
        Merge chunks from the same document and deduplicate results.
        Prioritizes chunks with higher scores and limits the number of chunks per original document.
        """
        processed_results = []
        doc_chunks = {}

        # Group chunks by original document
        for chunk in retrieved_chunks:
            # Ensure 'original_doc_idx' exists in the chunk for grouping
            orig_doc_idx = chunk.get("original_doc_idx")
            if orig_doc_idx is None:
                # Fallback or error handling if original_doc_idx is missing
                print(f"Warning: Chunk missing 'original_doc_idx', skipping for merging: {chunk.get('content', '')[:50]}...")
                continue # Skip chunks that can't be mapped

            if orig_doc_idx not in doc_chunks:
                doc_chunks[orig_doc_idx] = []
            doc_chunks[orig_doc_idx].append(chunk)

        # For each document, select the best chunks and merge them
        for doc_idx, chunks in doc_chunks.items():
            # Sort chunks by score in descending order
            sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)

            # Take top chunks based on limit
            top_chunks = sorted_chunks[:max_chunks_per_doc]

            if not top_chunks: # Should not happen if doc_chunks was populated, but good defensive check
                continue

            # Merge content
            # Joining with "[...]" provides a visual separation between merged chunks
            content = "\n[...]\n".join([chunk["content"] for chunk in top_chunks])

            # Combine metadata: use metadata from the highest-scoring chunk as base,
            # and add / override specific fields for the merged document.
            base_metadata = top_chunks[0].get("metadata", {}).copy()
            
            processed_results.append({
                "content": content,
                "metadata": {
                    **base_metadata, # Start with base metadata
                    "original_doc_idx": doc_idx, # Confirm original document index
                    "num_chunks_merged": len(top_chunks),
                    "merged_chunk_indices": [c["index"] for c in top_chunks], # List of original chunk indices
                    # You might also want to add average score or max score if useful
                    "merged_score": top_chunks[0]["score"] # Use the score of the top chunk for the merged result
                }
            })

        # Sort final processed results by their "merged_score" (which is the score of their top chunk)
        return sorted(processed_results, key=lambda x: x["metadata"].get("merged_score", 0), reverse=True)