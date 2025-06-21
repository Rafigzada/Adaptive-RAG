import os
import hashlib # Still needed for local hashing in CorpusInitializer
from typing import Dict, List, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

# Import caching functions from the new caching.py file
# Note the change: 'load_cached_data' and 'save_cached_data' are now 'load_cache' and 'save_cache'
from caching import get_doc_identifier, get_vectorizer_params_hash, load_cache, save_cache, VECTOR_CACHE_DIR

# Assuming these are available from load_docs.py and retrieval.py
from load_docs import load_wikipedia_articles, load_pdfs_from_directory, adaptive_chunking
from retrieval import DocumentRetriever # Needed to set its internal data during initialization


# --- Corpus Initialization and Management Class ---
class CorpusInitializer:
    def __init__(self):
        self.documents: List[str] = []
        self.document_metadata: List[Dict] = []
        self.full_doc_vectorizer: TfidfVectorizer = None
        self.full_doc_vectors: Any = None # Sparse matrix (from TF-IDF)
        self.doc_identifier: str = "" # Identifier for the current loaded document set

        # The DocumentRetriever instance from the RAGPipeline will be set here.
        # This manager will prepare the data that the retriever then uses internally.
        self.retriever_instance: DocumentRetriever = None

        # Common TF-IDF parameters for consistency across full docs and chunks
        self.tfidf_params = {
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.85,
        }
        # Use get_vectorizer_params_hash from caching.py
        self.tfidf_params_hash = get_vectorizer_params_hash(self.tfidf_params)


    def _prepare_full_document_vectors(self, cache_prefix: str):
        """Helper to create or load TF-IDF vectors for full documents."""
        # Use get_doc_identifier from caching.py
        self.doc_identifier = get_doc_identifier(self.documents, self.document_metadata)
        
        # Use load_cache from caching.py, specifying the cache name, subdirectory, identifier, and params hash
        loaded_data = load_cache(
            cache_name=f"{cache_prefix}_full_doc_vectors", # A unique name for this cache type
            sub_dir="vectors", # Stored in the 'vectors' subdirectory
            identifier=self.doc_identifier,
            params_hash=self.tfidf_params_hash
        )
        
        if loaded_data:
            self.full_doc_vectorizer, self.full_doc_vectors = loaded_data
            print(f"\nLoaded cached full document TF-IDF vectorizer. \nFull doc vectors shape: {self.full_doc_vectors.shape}")
        else:
            # If no cached data or cache is invalid, compute new vectors
            self.full_doc_vectorizer = TfidfVectorizer(**self.tfidf_params)
            self.full_doc_vectors = self.full_doc_vectorizer.fit_transform(self.documents)
            print(f"\nFitted full document TF-IDF vectorizer ({cache_prefix}). \nFull doc vectors shape: {self.full_doc_vectors.shape}")
            
            # Use save_cache from caching.py
            save_cache(
                (self.full_doc_vectorizer, self.full_doc_vectors), # Tuple of data to save
                cache_name=f"{cache_prefix}_full_doc_vectors",
                sub_dir="vectors",
                identifier=self.doc_identifier,
                params_hash=self.tfidf_params_hash
            )

    def _prepare_chunk_vectors(self, cache_prefix: str, default_chunk_size: int, default_overlap: int):
        """Helper to create or load chunk TF-IDF vectors for the retriever."""
        if self.retriever_instance is None:
            raise ValueError("DocumentRetriever instance must be set on CorpusInitializer before preparing chunk vectors.")

        # Chunk the documents (always re-chunk if documents might have changed, as chunking logic isn't cached)
        chunked_docs, doc_mapping, chunk_metadata = adaptive_chunking(
            self.documents, self.document_metadata,
            default_chunk_size=default_chunk_size,
            default_overlap=default_overlap
        )

        # Attempt to load cached chunk vectors and vectorizer
        # Use load_cache from caching.py
        loaded_data = load_cache(
            cache_name=f"{cache_prefix}_chunk_vectors", # A unique name for this cache type
            sub_dir="vectors", # Stored in the 'vectors' subdirectory
            identifier=self.doc_identifier,
            params_hash=self.tfidf_params_hash
        )
        
        if loaded_data:
            cached_vectorizer, cached_vectors = loaded_data
            # If loaded from cache, manually set the retriever's attributes
            self.retriever_instance.vectorizer = cached_vectorizer
            self.retriever_instance.document_vectors = cached_vectors
            self.retriever_instance.chunked_docs = chunked_docs # Re-assign as chunking is re-run
            self.retriever_instance.doc_mapping = doc_mapping # Re-assign
            self.retriever_instance.chunk_metadata = chunk_metadata # Re-assign
            print(f"\nLoaded cached {cache_prefix} chunk embeddings with shape {self.retriever_instance.document_vectors.shape}")
        else:
            # If no cached data, create embeddings (which also sets vectorizer/vectors on retriever)
            # The create_embeddings method on DocumentRetriever should handle its own TF-IDF fit
            self.retriever_instance.create_embeddings(chunked_docs, doc_mapping, chunk_metadata, self.tfidf_params)
            
            # Use save_cache from caching.py
            save_cache(
                (self.retriever_instance.vectorizer, self.retriever_instance.document_vectors),
                cache_name=f"{cache_prefix}_chunk_vectors",
                sub_dir="vectors",
                identifier=self.doc_identifier,
                params_hash=self.tfidf_params_hash
            )


    def initialize_wikipedia_corpus(self, retriever_instance: DocumentRetriever, limit: int = 50):
        """Initializes the corpus with Wikipedia articles."""
        print("🔍 Loading Wikipedia articles...")
        self.documents, self.document_metadata = load_wikipedia_articles(limit=limit)

        if not self.documents: # Check if documents were actually loaded
            raise ValueError(f"No Wikipedia documents found with limit={limit}. Please check your connection or limit.")
            
        self.retriever_instance = retriever_instance # Set the retriever instance
        self._prepare_full_document_vectors("wikipedia")
        self._prepare_chunk_vectors("wikipedia", default_chunk_size=300, default_overlap=50)
        print("✅ Wikipedia corpus initialization completed!")

    def initialize_pdf_corpus(self, retriever_instance: DocumentRetriever, pdf_directory: str = "./pdfs"):
        """Initializes the corpus with documents from a specified PDF directory."""
        print(f"\n--- Initializing PDF Corpus ---")
        print(f"Loading documents from {pdf_directory}...")
        self.documents, self.document_metadata = load_pdfs_from_directory(pdf_directory)

        if not self.documents: # Check if documents were actually loaded
            raise ValueError(f"No PDF documents found in {pdf_directory}. Please ensure files are in '{os.path.abspath(pdf_directory)}'")
            
        self.retriever_instance = retriever_instance # Set the retriever instance
        self._prepare_full_document_vectors("pdf")
        self._prepare_chunk_vectors("pdf", default_chunk_size=500, default_overlap=50)
        print("\n--- PDF Corpus Initialization Completed! ---")

    # These methods provide access to common parameters, useful for consistency
    def get_tfidf_params(self):
        return self.tfidf_params
        
    def get_tfidf_params_hash(self):
        return self.tfidf_params_hash
        
    def get_doc_identifier(self):
        return self.doc_identifier