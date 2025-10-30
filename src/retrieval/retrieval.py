import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Define a persistent directory for ChromaDB
CHROMA_DB_DIR = "./chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

class DocumentRetriever:
    """
    Manages document chunking, embedding using a Sentence Transformer model,
    and retrieval using ChromaDB.
    
    ENHANCEMENT: Added simple query expansion with crossover-like functionality
    """
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the DocumentRetriever with a ChromaDB client and an embedding function.
        Args:
            embedding_model_name (str): The name of the Sentence Transformer model to use.
                                        "all-MiniLM-L6-v2" is a good default for speed/accuracy.
        """
        print(f"Initializing ChromaDB client and embedding model: {embedding_model_name}...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )

        self.chunk_collection = None
        self.full_doc_collection = None
        self.summary_collection = None

        # Base names for collections; identifier will be appended
        self.chunk_collection_base_name = "rag_document_chunks"
        self.full_doc_collection_base_name = "rag_full_documents"
        self.summary_collection_base_name = "rag_summaries"

    def _get_or_create_collection(self, base_name: str, identifier: str):
        """Helper to get or create a ChromaDB collection with a unique name."""
        collection_name = f"{base_name}_{identifier}"
        print(f"Attempting to get or create ChromaDB collection: '{collection_name}'")
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Successfully connected to/created ChromaDB collection '{collection_name}'.")
            return collection
        except Exception as e:
            print(f"Error connecting to or creating ChromaDB collection '{collection_name}': {e}")
            return None

    def create_chunk_embeddings(self, chunked_docs: List[str], doc_mapping: List[int], chunk_metadata: List[Dict], cache_identifier: str):
        """
        Creates a ChromaDB collection and adds document chunks to it.
        """
        self.chunk_collection = self._get_or_create_collection(self.chunk_collection_base_name, cache_identifier)

        if self.chunk_collection:
            if self.chunk_collection.count() == 0 or self.chunk_collection.count() != len(chunked_docs):
                print(f"Collection '{self.chunk_collection.name}' is empty or outdated. Adding {len(chunked_docs)} chunks...")
                ids = [f"chunk_{i}" for i in range(len(chunked_docs))]
                self.chunk_collection.add(
                    documents=chunked_docs,
                    metadatas=chunk_metadata,
                    ids=ids
                )
                print(f"Added {len(chunked_docs)} documents to ChromaDB collection '{self.chunk_collection.name}'.")
            else:
                print(f"Collection '{self.chunk_collection.name}' already contains {self.chunk_collection.count()} documents. Skipping re-embedding.")

    def create_full_document_embeddings(self, documents: List[str], metadatas: List[Dict], cache_identifier: str):
        """
        Creates a ChromaDB collection and adds full documents to it.
        """
        self.full_doc_collection = self._get_or_create_collection(self.full_doc_collection_base_name, cache_identifier)

        if self.full_doc_collection:
            if self.full_doc_collection.count() == 0 or self.full_doc_collection.count() != len(documents):
                print(f"Collection '{self.full_doc_collection.name}' is empty or outdated. Adding {len(documents)} full documents...")
                ids = [f"doc_{i}" for i in range(len(documents))]
                # Metadatas for full docs might be slightly different than for chunks
                self.full_doc_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Added {len(documents)} full documents to ChromaDB collection '{self.full_doc_collection.name}'.")
            else:
                print(f"Collection '{self.full_doc_collection.name}' already contains {self.full_doc_collection.count()} documents. Skipping re-embedding.")

    def create_summary_embeddings(self, summaries: List[Dict], cache_identifier: str):
        """
        Creates a ChromaDB collection and adds document summaries to it.
        Ensures metadata is non-empty and handles potential None values.
        """
        self.summary_collection = self._get_or_create_collection(self.summary_collection_base_name, cache_identifier)

        if self.summary_collection:
            summary_texts = [s["summary"] for s in summaries]

            processed_summary_metadatas = []
            for i, s in enumerate(summaries):
                current_metadata = s.get("metadata", {}).copy() # This will now correctly grab the nested dict

                if "original_doc_idx" in current_metadata:
                    try:
                        current_metadata["original_doc_idx"] = int(current_metadata["original_doc_idx"])
                    except (ValueError, TypeError):
                        current_metadata["original_doc_idx"] = i # Fallback if conversion fails
                else:
                    current_metadata["original_doc_idx"] = i # Default if not found (shouldn't happen with the change above)

                # Clean metadata values: remove None, and ensure values are acceptable types for ChromaDB
                keys_to_delete = []
                for k, v in current_metadata.items():
                    if v is None:
                        keys_to_delete.append(k)
                    elif not isinstance(v, (bool, int, float, str)):
                        # Convert other types (like lists, objects if they somehow get in) to strings.
                        current_metadata[k] = str(v)
                
                for k in keys_to_delete:
                    del current_metadata[k]

                # IMPORTANT: Guarantee metadata is never empty (ChromaDB requirement)
                # Ensure 'original_doc_idx' is always present as a minimum
                if not current_metadata: # If all useful metadata somehow got removed
                    current_metadata = {"original_doc_idx": i} # Re-add original_doc_idx
                elif "original_doc_idx" not in current_metadata: # Ensure it's always there
                    current_metadata["original_doc_idx"] = i

                processed_summary_metadatas.append(current_metadata) # Append the cleaned current_metadata

            ids = [f"summary_{j}" for j in range(len(summaries))]

            if self.summary_collection.count() == 0 or self.summary_collection.count() != len(summaries):
                print(f"Collection '{self.summary_collection.name}' is empty or outdated. Adding {len(summaries)} summaries...")
                self.summary_collection.add(
                    documents=summary_texts,
                    metadatas=processed_summary_metadatas, # Use the processed_summary_metadatas
                    ids=ids
                )
                print(f"Added {len(summaries)} summaries to ChromaDB collection '{self.summary_collection.name}'.")
            else:
                print(f"Collection '{self.summary_collection.name}' already contains {self.summary_collection.count()} documents. Skipping re-embedding.")

    def expand_query_with_crossover(self, query: str) -> List[str]:
        """
        SIMPLE IMPROVEMENT: Expand query using crossover-like technique
        Creates variations of the original query by combining with common search patterns
        """
        # Base query patterns for crossover
        patterns =[
            "information about",
            "details on", 
            "research on",
            "data regarding"
        ]
        
        # Simple synonyms for expansion
        synonyms = {
            "find": ["search", "locate"],
            "document": ["paper", "file"], 
            "information": ["data", "details"],
            "about": ["on", "regarding"]
        }
        
        expanded_queries = [query]  # Start with original
        
        # Pattern crossover: combine query with patterns
        query_words = query.lower().split()
        for pattern in patterns[:2]:  # Limit to 2 patterns
            if len(query_words) > 1:
                # Remove first word if it's a search verb, then add pattern
                if query_words[0] in ["find", "search", "get", "locate"]:
                    new_query = f"{pattern} {' '.join(query_words[1:])}"
                else:
                    new_query = f"{pattern} {query}"
                expanded_queries.append(new_query)
        
        # Synonym crossover: replace one word with synonym
        for word in query_words[:2]:  # Limit to first 2 words
            if word in synonyms:
                synonym = synonyms[word][0]  # Take first synonym
                new_query = query.replace(word, synonym, 1)
                expanded_queries.append(new_query)
        
        return expanded_queries[:4]  # Return max 4 queries (original + 3 variants)

    def retrieve_relevant_docs(self, query: str, k: int = 5, collection_type: str = "chunk", 
                             where_clause: Optional[Dict] = None, use_expansion: bool = True) -> Tuple[List[Dict], List[float]]:
        """
        ENHANCED: Retrieves the most relevant documents with optional query expansion
        
        Args:
            query (str): The user's query.
            k (int): The number of top relevant results to retrieve.
            collection_type (str): Specifies which collection to query ('chunk', 'full_doc', 'summary').
            where_clause (Optional[Dict]): A dictionary representing a WHERE clause for metadata filtering.
            use_expansion (bool): Whether to use query expansion with crossover technique.

        Returns:
            Tuple[List[Dict], List[float]]: Retrieved items and their similarity scores.
        """
        collection = None
        if collection_type == "chunk":
            collection = self.chunk_collection
        elif collection_type == "full_doc":
            collection = self.full_doc_collection
        elif collection_type == "summary":
            collection = self.summary_collection
        else:
            print(f"Invalid collection_type: {collection_type}")
            return [], []

        if collection is None:
            print(f"ChromaDB '{collection_type}' collection is not initialized. Cannot perform retrieval.")
            return [], []

        # IMPROVEMENT: Use query expansion if enabled
        if use_expansion:
            queries = self.expand_query_with_crossover(query)
            print(f"Expanding query '{query}' to {len(queries)} variants")
        else:
            queries = [query]

        all_retrieved_items = []
        all_scores = []

        # Retrieve for each query variant
        for q in queries:
            print(f"\n--- Retrieving relevant {collection_type}s for: '{q}' (k={k//len(queries) + 1}) ---")
            if where_clause:
                print(f"  Applying WHERE clause: {where_clause}")
            try:
                results = collection.query(
                    query_texts=[q],
                    n_results=max(1, k//len(queries) + 1),  # Distribute k across queries
                    where=where_clause,
                    include=['documents', 'metadatas', 'distances']
                )

                if results and results['documents']:
                    for i in range(len(results['documents'][0])):
                        item_content = results['documents'][0][i]
                        item_meta = results['metadatas'][0][i]
                        item_score = results['distances'][0][i]

                        all_retrieved_items.append({
                            "content": item_content,
                            "metadata": item_meta,
                            "score": item_score
                        })
                        all_scores.append(item_score)

            except Exception as e:
                print(f"Error during ChromaDB retrieval for '{collection_type}' with query '{q}': {e}")

        # Remove duplicates and sort by score
        seen_content = set()
        unique_items = []
        unique_scores = []
        
        for item, score in zip(all_retrieved_items, all_scores):
            content_key = item["content"][:100]  # Use first 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_items.append(item)
                unique_scores.append(score)

        # Sort by score and limit to k
        sorted_pairs = sorted(zip(unique_items, unique_scores), key=lambda x: x[1])
        final_items = [item for item, _ in sorted_pairs[:k]]
        final_scores = [score for _, score in sorted_pairs[:k]]

        # Print results
        for i, item in enumerate(final_items):
            item_title = item['metadata'].get("title", f"{collection_type.capitalize()} {item['metadata'].get('original_doc_idx', 'N/A')}")
            item_source = item['metadata'].get("source", "N/A")
            print(f"  {collection_type.capitalize()} {i+1}: '{item['content'][:50]}...' \n    (Doc: '{item_title}', \n    Source: {os.path.basename(item_source)}) - \n    Score (Distance): {final_scores[i]:.4f}")

        return final_items, final_scores

    def process_retrieved_chunks(self, retrieved_chunks: List[Dict], max_chunks_per_doc: int = 3) -> List[Dict]:
        """
        Processes and potentially merges relevant chunks from the same original document,
        ensuring unique documents are represented and limiting chunks per document.
        This prepares the documents for re-ranking.

        Args:
            retrieved_chunks (List[Dict]): A list of dictionaries, each representing a retrieved chunk.
                                            Each dict should have 'content' and 'metadata'.
            max_chunks_per_doc (int): Maximum number of chunks to use per original document.

        Returns:
            List[Dict]: A list of dictionaries, each representing a consolidated document
                        (or a group of chunks from the same document) with 'content' and 'metadata'.
        """
        print(f"\n--- Processing retrieved chunks for re-ranking (max_chunks_per_doc={max_chunks_per_doc}) ---")
        # Group chunks by their original document ID
        grouped_chunks = {}
        for chunk in retrieved_chunks:
            original_doc_idx = chunk['metadata'].get('original_doc_idx')
            if original_doc_idx is not None:
                if original_doc_idx not in grouped_chunks:
                    grouped_chunks[original_doc_idx] = {
                        "content_parts": [],
                        "metadata": chunk['metadata']
                    }
                grouped_chunks[original_doc_idx]["content_parts"].append({
                    "content": chunk["content"],
                    "chunk_idx": chunk['metadata'].get('chunk_idx')
                })

        # Sort chunks within each document group by their original chunk index
        for doc_idx in grouped_chunks:
            grouped_chunks[doc_idx]["content_parts"].sort(key=lambda x: x["chunk_idx"])

        processed_docs = []
        for doc_idx, doc_data in grouped_chunks.items():
            # Take only the top N chunks as specified
            selected_chunks_for_doc = doc_data["content_parts"][:max_chunks_per_doc]

            # Concatenate content from selected chunks
            combined_content = "\n\n".join([c["content"] for c in selected_chunks_for_doc])

            # Use the metadata from one of the chunks (they should be consistent for the same doc)
            # Ensure 'original_doc_idx' is still in the metadata for future reference
            metadata = doc_data["metadata"]

            processed_docs.append({
                "content": combined_content,
                "metadata": metadata
            })
            print(f"  Consolidated document {len(processed_docs)}: Title='{metadata.get('title', 'N/A')}' (Source: {os.path.basename(metadata.get('source', 'N/A'))}) with {len(selected_chunks_for_doc)} chunks. Content Length: {len(combined_content)}")

        print(f"--- Finished processing. Total {len(processed_docs)} documents prepared for re-ranking. ---")
        return processed_docs