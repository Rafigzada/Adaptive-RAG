import os
import hashlib
from typing import Dict, List, Any, Tuple


from src.caching.caching import get_doc_identifier
from src.loading_documents.load_docs import load_wikipedia_articles, load_pdfs_from_directory, adaptive_chunking
from src.retrieval.retrieval import DocumentRetriever


class CorpusInitializer:
    def __init__(self):
        self.documents: List[str] = []
        self.document_metadata: List[Dict] = []
        self.doc_identifier: str = "" 

        self.retriever_instance: DocumentRetriever = None

        self.document_summaries: List[Dict] = []


    def _prepare_full_document_embeddings(self, cache_prefix: str):
        """Helper to create embeddings for full documents in ChromaDB."""
        if self.retriever_instance is None:
            raise ValueError("DocumentRetriever instance must be set on CorpusInitializer before preparing full document embeddings.")

        print(f"\n--- Preparing full document embeddings for {cache_prefix} corpus ---")
        self.retriever_instance.create_full_document_embeddings(self.documents, self.document_metadata, self.doc_identifier)
        print(f"ChromaDB embeddings prepared for {len(self.documents)} full documents.")

    def _prepare_summary_embeddings(self, cache_prefix: str):
         """Helper to create embeddings for document summaries in ChromaDB."""
         if self.retriever_instance is None:
             raise ValueError("DocumentRetriever instance must be set on CorpusInitializer before preparing summary embeddings.")

         if not self.document_summaries:
             print(f"No summaries available for {cache_prefix} corpus. Skipping summary embedding preparation.")
             return

         print(f"\n--- Preparing summary embeddings for {cache_prefix} corpus ---")

         self.retriever_instance.create_summary_embeddings(self.document_summaries, self.doc_identifier)
         print(f"ChromaDB embeddings prepared for {len(self.document_summaries)} summaries.")

    def _prepare_chunk_vectors(self, cache_prefix: str, default_chunk_size: int, default_overlap: int):
        """
        Helper to chunk documents and prepare embeddings using the DocumentRetriever (ChromaDB).
        """
        if self.retriever_instance is None:
            raise ValueError("DocumentRetriever instance must be set on CorpusInitializer before preparing chunk vectors.")

        print(f"\n--- Chunking documents for {cache_prefix} corpus ---")
        chunked_docs, doc_mapping, chunk_metadata = adaptive_chunking(
            self.documents, self.document_metadata,
            default_chunk_size=default_chunk_size,
            default_overlap=default_overlap
        )
        print(f"Chunked {len(self.documents)} documents into {len(chunked_docs)} chunks.")

        self.retriever_instance.create_chunk_embeddings(
            chunked_docs, doc_mapping, chunk_metadata, self.doc_identifier # Using doc_identifier for collection naming
        )
        print(f"ChromaDB embeddings prepared for {len(chunked_docs)} chunks.")


    def initialize_wikipedia_corpus(self, retriever_instance: DocumentRetriever, limit: int = 50):
        """Initializes the corpus with Wikipedia articles."""
        print("\n🔍 Loading Wikipedia articles...")
        self.documents, self.document_metadata = load_wikipedia_articles(limit=limit)

        if not self.documents: # Check if documents were actually loaded
            raise ValueError(f"No Wikipedia documents found with limit={limit}. Please check your your connection or limit.")

        self.retriever_instance = retriever_instance # Set the retriever instance
        self.doc_identifier = get_doc_identifier(self.documents, self.document_metadata) # Set identifier here

        # Prepare all embeddings using ChromaDB
        self._prepare_full_document_embeddings("wikipedia")

        self._prepare_chunk_vectors("wikipedia", default_chunk_size=300, default_overlap=50)
        print("✅ Wikipedia corpus initialization completed!")

    def initialize_pdf_corpus(self, retriever_instance: DocumentRetriever, pdf_directory: str = "./pdfs"):
        """Initializes the corpus with documents from a specified PDF directory."""
        print(f"\n--- Initializing PDF Corpus ---")
        print(f"Loading documents from {pdf_directory}...")
        self.documents, self.document_metadata = load_pdfs_from_directory(pdf_directory)

        if not self.documents:
            raise ValueError(f"No PDF documents found in {pdf_directory}. Please ensure files are in '{os.path.abspath(pdf_directory)}'")

        self.retriever_instance = retriever_instance # Set the retriever instance
        self.doc_identifier = get_doc_identifier(self.documents, self.document_metadata) # Set identifier here

        # Prepare all embeddings using ChromaDB
        self._prepare_full_document_embeddings("pdf")

        self._prepare_chunk_vectors("pdf", default_chunk_size=500, default_overlap=50)
        print("\n--- PDF Corpus Initialization Completed! ---")