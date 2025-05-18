





import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'modules')))
from document_collection import documents
from document_embedding import create_tfidf_embeddings
import time
from sklearn.metrics.pairwise import cosine_similarity

document_vectors, vectorizer=create_tfidf_embeddings(documents)

def retrieve_relevant_docs(query, k=3):
    """
    Retrieve the k most relevant documents for a query using TF-IDF similarity.

    Args:
    - query (str): The query string.
    - document_vectors (sparse matrix): The TF-IDF matrix of the documents.
    - documents (list of str): List of document strings.
    - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    - k (int): The number of relevant documents to retrieve.

    Returns:
    - results (list of dict): A list of the k most relevant documents with their scores and indices.
    - retrieval_time (float): Time taken for retrieval.
    """
    print(f"\n=== RETRIEVAL PHASE for: '{query}' ===")
    start_time = time.time()

    # Convert query to vector using the same vectorizer
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between query and all documents
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Get top k document indices based on similarity
    top_indices = similarities.argsort()[-k:][::-1]

    # Create result list with document index, similarity score, and content
    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            "index": idx,
            "score": float(similarities[idx]),
            "content": documents[idx]
        })
        print(f"Retrieved document {i+1} (score: {similarities[idx]:.4f}):")
        print(f"- {documents[idx][:100]}...")  # Show first 100 characters of the document

    retrieval_time = time.time() - start_time
    print(f"Retrieval completed in {retrieval_time:.4f} seconds")

    return results, retrieval_time





