






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


    "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary.",
    "Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\".",
    "Next to the Main Building is the Basilica of the Sacred Heart.",
    "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.",
    "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858.",
    "At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."








