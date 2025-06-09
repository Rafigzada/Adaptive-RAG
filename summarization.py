import google.generativeai as genai
from typing import List, Dict, Any
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """
    A class to handle the generation of document summaries using an LLM.
    """
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.summary_model = genai.GenerativeModel(model_name)
        logger.info(f"Document Summarizer model initialized ({model_name}).")
        print(f"\nDocument Summarizer model initialized ({model_name}).")

    def generate_summaries(self, documents: List[str], document_metadata: List[Dict]) -> List[Dict]:
        """
        Generates concise summaries for a list of documents using the LLM.
        """
        print(f"\nGenerating summaries for {len(documents)} documents...")
        document_summaries = []
        for i, doc_text in enumerate(documents):
            doc_idx = i
            metadata = document_metadata[doc_idx]
            original_source = metadata.get("source", f"document_{doc_idx}")
            original_title = metadata.get("title", f"Document {doc_idx}")

            prompt = f"Summarize the following document , focusing on its main topic and key points. Do not include introductory phrases like 'This document discusses' or 'The text is about'. Just the summary.\n\nDocument Title: {original_title}\n\nDocument Content:\n{doc_text[:8000]}..."

            if i == 0: # Only print for the first document to avoid excessive output
                print("\n--- DEBUG: Document Content Sent to LLM (First Document Only) ---")
                print(f"Document Source: {original_source}")
                print(f"Content (first 200 chars): {doc_text[:200]}{'...' if len(doc_text) > 200 else ''}")
                print("---------------------------------------------------------------\n")
            
            try:
                response = self.summary_model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": 800, # Concise summary
                        "temperature": 0.1,
                    }
                )
                summary_text = response.text.strip()

                if i == 0: # Only print for the first document processed
                    print("\n--- TEST OUTPUT: First Document Summary Preview ---")
                    print(f"Document Source: {original_source}")
                    print(f"Summary : {summary_text}")
                    print("---------------------------------------------------\n")
                
                document_summaries.append({
                    "doc_idx": doc_idx,
                    "summary": summary_text,
                    "original_source": original_source,
                    "original_title": original_title,
                    "original_doc_text_preview": doc_text[:500] # For debugging
                })
                print(f"  - Summary for '{original_title}' generated.")
            except genai.types.BlockedPromptException:
                print(f"  - Warning: Summarization for '{original_title}' was blocked due to safety concerns.")
                document_summaries.append({
                    "doc_idx": doc_idx,
                    "summary": "[Blocked due to safety concerns]",
                    "original_source": original_source,
                    "original_title": original_title,
                    "original_doc_text_preview": doc_text[:500]
                })
            except Exception as e:
                print(f"  - Error generating summary for '{original_title}': {e}")
                document_summaries.append({
                    "doc_idx": doc_idx,
                    "summary": "[Error generating summary]",
                    "original_source": original_source,
                    "original_title": original_title,
                    "original_doc_text_preview": doc_text[:500]
                })
            time.sleep(0.1) # Small delay to avoid hitting rate limits

        print("Summary generation completed.")
        return document_summaries