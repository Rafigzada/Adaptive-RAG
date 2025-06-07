import google.generativeai as genai
from typing import List, Dict, Any
import os
import time

class DocumentReRanker:
    """
    A class to handle the re-ranking of retrieved documents using an LLM.
    """
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.reranker_model = genai.GenerativeModel(model_name)
        print(f"Document Re-ranker model initialized ({model_name}).")

    def rerank_documents(self, query: str, retrieved_docs: List[Dict], top_k_rerank: int = 3) -> List[Dict]:
        """
        Re-ranks the retrieved documents using a Generative Model to assess relevance.
        This provides a more semantic and contextual re-ranking.
        """
        if not retrieved_docs:
            print("  - No documents to re-rank. Returning empty list.")
            return []

        print(f"\n--- Re-ranking {len(retrieved_docs)} Retrieved Documents ---")
        reranked_scores = []

        batch_size = 3 # Process documents in batches to manage context window and get scores
        
        for i in range(0, len(retrieved_docs), batch_size):
            batch_docs = retrieved_docs[i:i + batch_size] # Use the actual document objects
            
            print(f"    --- Sending Batch {i//batch_size + 1} to Re-ranker (Documents in this batch) ---")
            batch_candidates_text = []
            for j, doc in enumerate(batch_docs):
                title = doc.get('metadata', {}).get('title', 'Unknown Title')
                source = os.path.basename(doc.get('metadata', {}).get('source', 'Unknown Source'))
                content_preview = doc.get('content', '')[:500] # Limit content for prompt
                
                batch_candidates_text.append(f"Document {j+1} (Title: {title}, Source: {source}): {content_preview}...")
                print(f"    Document {j+1} in batch: '{title}' (Source: {source})")
            print("  -------------------------------------------------------------")

            rerank_prompt = f"Given the query: \"{query}\"\n\n"
            rerank_prompt += "Please rate the relevance of the following documents to the query on a scale of 1 to 5 (1 = Not relevant, 5 = Highly relevant). Provide only the document number and its score, for example 'Document 1: 5'. List them from most to least relevant. If a document is completely irrelevant, give it a score of 0.\n\n"
            for candidate_text in batch_candidates_text:
                rerank_prompt += f"{candidate_text}\n\n"
            rerank_prompt += "Relevance Scores (Document Number: Score) and ordered list (most relevant first):"

            try:
                response = self.reranker_model.generate_content(
                    rerank_prompt,
                    generation_config={
                        "max_output_tokens": 150,
                        "temperature": 0.1,
                    }
                )
                rerank_output = response.text.strip()
                print(f"\n  - LLM Re-ranker Output(Score given by LLM on a scale of 1 to 5) for batch {i//batch_size + 1}:\n{rerank_output}\n") # Debug print

                batch_scores_parsed = []
                lines = rerank_output.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('Document') and ':' in line:
                        try:
                            parts = line.split(':')
                            doc_num_part = parts[0].replace('Document', '').strip()
                            score_part = parts[1].strip()
                            
                            doc_num = int(doc_num_part)
                            score = float(score_part)
                            if 1 <= doc_num <= len(batch_docs): # Ensure doc_num corresponds to an actual document in the batch
                                batch_scores_parsed.append({'doc_idx_in_batch': doc_num - 1, 'score': score})
                        except (ValueError, IndexError) as e:
                            print(f"    - Warning: Could not parse line '{line}': {e}")
                            continue
                
                batch_scores_parsed.sort(key=lambda x: x['score'], reverse=True)

                for entry in batch_scores_parsed:
                    original_doc_index_in_batch = entry['doc_idx_in_batch']
                    doc_to_add = batch_docs[original_doc_index_in_batch].copy() # Copy to avoid modifying original
                    doc_to_add['rerank_score'] = entry['score']
                    
                    # Ensure we don't add duplicates (though with doc groups, this is less common)
                    if not any(d.get('metadata', {}).get('original_doc_idx') == doc_to_add.get('metadata', {}).get('original_doc_idx') and
                               d.get('metadata', {}).get('chunk_idx') == doc_to_add.get('metadata', {}).get('chunk_idx') # Use chunk_idx for uniqueness within same doc
                               for d in reranked_scores):
                        reranked_scores.append(doc_to_add)

            except genai.types.BlockedPromptException as e:
                print(f"  - WARNING: Re-ranking prompt was blocked for batch {i//batch_size + 1}: {e}")
            except Exception as e:
                print(f"  - ERROR during re-ranking batch {i//batch_size + 1}: {e}")
            
            time.sleep(0.1) # Small delay for rate limiting

        # Final sort of all collected re-ranked documents by their new 'rerank_score'
        reranked_scores.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        print(f"  - Total documents collected for re-ranking: {len(reranked_scores)}")
        print(f"Re-ranked to top {top_k_rerank} documents.")
        return reranked_scores[:top_k_rerank]