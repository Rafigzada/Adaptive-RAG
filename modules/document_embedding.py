






from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_embeddings(documents):
    """
    Function to create document embeddings using TF-IDF.
    
    Args:
    - documents (list of str): List of text documents to create embeddings for.
    
    Returns:
    - document_vectors (sparse matrix): TF-IDF matrix representing document embeddings.
    """
    print("\nCreating document embeddings using TF-IDF...")
    
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Generate the TF-IDF document-term matrix
    document_vectors = vectorizer.fit_transform(documents)
    
    # Print the shape of the resulting matrix
    print(f"Created embeddings with shape {document_vectors.shape}")
    
    return document_vectors, vectorizer










