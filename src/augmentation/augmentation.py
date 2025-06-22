import time
import os
import hashlib
from typing import List, Dict, Tuple


def score_documents(retrieved_docs: List[Dict], query: str) -> List[Tuple[Dict, float]]:
    """Mock scoring of retrieved docs based on length or custom criteria"""
    scored = []
    for doc in retrieved_docs:
        
        score = len(doc['content'])  # Example: longer content is ranked higher
        scored.append((doc, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def truncate_context(context_parts: List[str], max_length: int = 4000) -> List[str]:

    total_length = 0
    truncated = []
    for part in context_parts:
        part_len = len(part)
        if total_length + part_len > max_length:
            break
        truncated.append(part)
        total_length += part_len
    return truncated


def create_augmented_prompt(query: str, retrieved_docs: List[Dict], max_chunks: int = 5, max_chars: int = 4000) -> Tuple[str, float]:
    """Create a prompt with retrieved context and citations, enhanced with scoring and truncation"""
    print(f"\n=== CONTEXT INTEGRATION PHASE ===")
    start_time = time.time()

    # Score and rank documents
    scored_docs = score_documents(retrieved_docs, query)

    

    # Remove duplicates using content hash
    seen_hashes = set()
    unique_docs = []
    for doc, score in scored_docs:
        content_hash = hashlib.md5(doc['content'].encode()).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
        if len(unique_docs) >= max_chunks:
            break

    # Format context with citations
    context_parts = []
    for i, doc in enumerate(unique_docs):
        metadata = doc.get("metadata", {})
        source = metadata.get("title", f"Document {i+1}")
        
        if not metadata.get("is_full_doc", True):
            chunk_info = f" (Chunk {metadata.get('chunk_idx', 0) + 1})"
        context_parts.append(f"[{i+1}] From {source}{chunk_info}:\n{doc['content']}")

    # Truncate if necessary
    context_parts = truncate_context(context_parts, max_chars)
    context = "\n\n".join(context_parts)


    # Final prompt
    prompt = f"""Question: {query}

Context information:
{context}

Answer the question based on the context information provided above. Include citation numbers [1], [2], etc. when referencing specific information from the context.
If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
Present your answer in a clear and concise manner.
"""

    prompt_time = time.time() - start_time
    print(f"Created prompt with {len(context_parts)} chunks in {prompt_time:.4f} seconds")
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
    """
    Create chat messages for the API call, adapted for Gemini.
    The system prompt is now prepended to the user message.
    """
    if system_prompt is None:
        system_prompt = create_system_prompt()

    combined_user_content = (
        f"{system_prompt}\n\n"
        f"Question: {query}\n\n"
        f"Context information:\n{context}\n\n"
        f"Answer the question based on the context information provided above. "
        f"Include citation numbers [1], [2], etc. when referencing specific information from the context.\n"
        f"If the context doesn't contain enough information to fully answer the question, acknowledge this limitation in your response.\n"
        f"Present your answer in a clear and concise manner."
    )

    return [
        {"role": "user", "parts": [{"text": combined_user_content}]}  # Gemini expects 'parts' with 'text'
    ]
