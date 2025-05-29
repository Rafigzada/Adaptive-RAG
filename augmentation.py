import time
from typing import List, Dict, Tuple


def create_augmented_prompt(query: str, retrieved_docs: List[Dict]) -> Tuple[str, float]:
    """Create a prompt with retrieved context and citations"""
    print(f"\n=== CONTEXT INTEGRATION PHASE ===")
    start_time = time.time()

    # Format context with citations, noting original document sources
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("title", f"Document {i+1}")

        context_parts.append(f"[{i+1}] From {source}:\n{doc['content']}")

    context = "\n\n".join(context_parts)

    # Create structured prompt with clear instructions
    prompt = f"""Question: {query}

Context information:
{context}

Answer the question based on the context information provided above. Include citation numbers [1], [2], etc. when referencing specific information from the context.
If the context doesn't contain enough information to fully answer the question, acknowledge this limitation in your response.
Present your answer in a clear and concise manner.
"""

    prompt_time = time.time() - start_time
    print(f"Created prompt with {len(retrieved_docs)} chunks in {prompt_time:.4f} seconds")
    print(f"Prompt length: {len(prompt)} characters")

    return prompt, prompt_time


def format_context_with_citations(retrieved_docs: List[Dict]) -> str:
    """Format retrieved documents with proper citations"""
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("title", f"Document {i+1}")
        
        # Add chunk information if available
        chunk_info = ""
        if not metadata.get("is_full_doc", True):
            chunk_info = f" (Chunk {metadata.get('chunk_idx', 0) + 1})"
        
        context_parts.append(f"[{i+1}] From {source}{chunk_info}:\n{doc['content']}")
    
    return "\n\n".join(context_parts)


def create_system_prompt() -> str:
    """Create a system prompt for the AI assistant"""
    return """You are a helpful assistant that answers questions based on provided context.
Always cite your sources using the provided citation numbers [1], [2], etc.
If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
Provide clear, concise, and well-structured responses."""


def create_chat_messages(query: str, context: str, system_prompt: str = None) -> List[Dict[str, str]]:
    """Create chat messages for the API call"""
    if system_prompt is None:
        system_prompt = create_system_prompt()
    
    user_message = f"""Question: {query}

Context information:
{context}

Answer the question based on the context information provided above. Include citation numbers [1], [2], etc. when referencing specific information from the context.
If the context doesn't contain enough information to fully answer the question, acknowledge this limitation in your response.
Present your answer in a clear and concise manner."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
