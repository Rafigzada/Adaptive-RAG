import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any


class DocumentRetriever:
    def __init__(self):
        self.vectorizer = None
        self.document_vectors = None
        self.chunked_docs = None
        self.doc_mapping = None
        self.chunk_metadata = None
    
    def create_embeddings(self, chunked_docs: List[str], doc_mapping: List[int], chunk_metadata: List[Dict]):
        """Create document embeddings using TF-IDF"""
        self.chunked_docs = chunked_docs
        self.doc_mapping = doc_mapping
        self.chunk_metadata = chunk_metadata
        
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(chunked_docs)
        print(f"Created document embeddings with shape {self.document_vectors.shape}")
    
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> Tuple[List[Dict], float]:
        """Retrieve the k most relevant document chunks for a query using TF-IDF similarity"""
        print(f"\n=== RETRIEVAL PHASE for: '{query}' ===")
        start_time = time.time()

        if self.vectorizer is None or self.document_vectors is None:
            raise ValueError("Embeddings not created. Call create_embeddings() first.")

        # Convert query to vector
        query_vector = self.vectorizer.transform([query])

        # Calculate similarity with all document chunks
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()

        # Get top k document indices
        top_indices = similarities.argsort()[-k:][::-1]

        # Create result list
        results = []
        for i, idx in enumerate(top_indices):
            # Add chunk metadata if available
            metadata = self.chunk_metadata[idx]
            source_title = metadata.get("title", f"Document {self.doc_mapping[idx] + 1}")
            source_doc = metadata.get("source", "")

            results.append({
                "index": idx,
                "score": float(similarities[idx]),
                "content": self.chunked_docs[idx],
                "original_doc_idx": self.doc_mapping[idx],
                "metadata": metadata
            })
            print(f"Retrieved chunk {i+1} (score: {similarities[idx]:.4f}):")
            print(f"- From: {source_title}")
            print(f"- Content preview: {self.chunked_docs[idx][:150]}...")

        retrieval_time = time.time() - start_time
        print(f"Retrieval completed in {retrieval_time:.4f} seconds")

        return results, retrieval_time

    def process_retrieved_chunks(self, retrieved_chunks: List[Dict], max_chunks_per_doc: int = 2) -> List[Dict]:
        """Merge chunks from the same document and deduplicate results"""
        processed_results = []
        doc_chunks = {}

        # Group chunks by original document
        for chunk in retrieved_chunks:
            orig_doc_idx = chunk["original_doc_idx"]
            if orig_doc_idx not in doc_chunks:
                doc_chunks[orig_doc_idx] = []
            doc_chunks[orig_doc_idx].append(chunk)

        # For each document, select the best chunks
        for doc_idx, chunks in doc_chunks.items():
            # Sort chunks by score
            sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)

            # Take top chunks based on limit
            top_chunks = sorted_chunks[:max_chunks_per_doc]

            # Create merged content
            content = "\n[...]\n".join([chunk["content"] for chunk in top_chunks])

            # Add processed result
            processed_results.append({
                "index": top_chunks[0]["index"],  # Use index of highest scoring chunk
                "score": top_chunks[0]["score"],  # Use highest score
                "content": content,
                "original_doc_idx": doc_idx,
                "metadata": top_chunks[0].get("metadata", {}),
                "num_chunks_merged": len(top_chunks)
            })

        # Sort final results by score
        return sorted(processed_results, key=lambda x: x["score"], reverse=True)