





import time

def create_augmented_prompt(query, retrieved_docs):
    """
    Create a prompt with retrieved context and citations.
    
    Args:
    - query (str): The user's query/question.
    - retrieved_docs (list of dict): The retrieved documents with their index, score, and content.

    Returns:
    - prompt (str): The structured prompt containing the context and the question.
    - prompt_time (float): Time taken to create the augmented prompt.
    """
    print(f"\n=== CONTEXT INTEGRATION PHASE ===")
    start_time = time.time()

    # Format context with citations (numbering the documents)
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(f"[{i+1}] {doc['content']}")

    context = "\n\n".join(context_parts)

    # Create structured prompt with the query and context
    prompt = f"""Question: {query}

Context information:
{context}

Answer the question based on the context information provided. Include citation numbers [1], [2], etc. when referencing specific documents.
"""

    prompt_time = time.time() - start_time
    print(f"Created prompt with {len(retrieved_docs)} documents in {prompt_time:.4f} seconds")
    print(f"Prompt length: {len(prompt)} characters")

    return prompt, prompt_time





