





import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'modules')))
import time
import google.generativeai as genai
from retrieval_system import retrieve_relevant_docs
from context_integration import create_augmented_prompt
from generation import generate_answer_gemini


def rag_pipeline_gemini(query, api_key, k=3):
    """
    Complete RAG pipeline with Gemini for generation.

    Args:
        query (str): The query/question to be answered.
        api_key (str): Your Google Gemini API key.
        k (int): The number of relevant documents to retrieve (default is 3).

    Returns:
        dict: A dictionary containing the results and performance metrics.
    """
    print(f"\n{'='*70}")
    print(f"RUNNING COMPLETE RAG PIPELINE FOR: '{query}'")
    print(f"{'='*70}")

    # Start the pipeline timer
    pipeline_start = time.time()

    # Step 1: Retrieve relevant documents
    retrieved_docs, retrieval_time = retrieve_relevant_docs(query, k=k) # Assuming this function is defined elsewhere

    # Step 2: Create augmented prompt using the retrieved documents
    prompt, prompt_time = create_augmented_prompt(query, retrieved_docs) # Assuming this function is defined elsewhere

    # Step 3: Generate an answer using Gemini with the augmented prompt
    answer, generation_time = generate_answer_gemini(prompt, api_key=api_key)

    # Calculate total pipeline time
    total_time = time.time() - pipeline_start

    # Display results
    print(f"\n=== FINAL RESULT ===")
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    print(f"\nPerformance Metrics:")
    print(f"- Retrieval time: {retrieval_time:.4f}s")
    print(f"- Prompt creation time: {prompt_time:.4f}s")
    print(f"- Generation time: {generation_time:.4f}s")
    print(f"- Total pipeline time: {total_time:.4f}s")

    # Return the result in a structured format
    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "prompt": prompt,
        "answer": answer,
        "metrics": {
            "retrieval_time": retrieval_time,
            "prompt_time": prompt_time,
            "generation_time": generation_time,
            "total_time": total_time
        }
    }





