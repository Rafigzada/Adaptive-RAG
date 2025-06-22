#!/usr/bin/env python3
"""
Main script to run the RAG pipeline
"""

import os
import sys
import pandas as pd

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

# Import all required modules
from rag_pipeline import RAGPipeline, rag_pipeline_pdf
from src.loading_documents.load_docs import pdf_directory, list_pdf_files

def main():
    """Main function to run the RAG demo"""
    print("Starting RAG Demo with Gemini...")
    
    # Configuration
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
    PDF_DIR = "./pdfs"
    
    if not OPENAI_API_KEY:
        print("Please set your OpenAI API key in the OPENAI_API_KEY variable")
        return
    
    # Setup PDF directory
    pdf_directory(PDF_DIR)
    pdf_files = list_pdf_files(PDF_DIR)
    
    #if not pdf_files:
    #    print("No PDF files found. Please add some PDFs to the pdfs directory.")
    #    return

    
    # Sample questions to demonstrate
    test_queries = [
        "What are the main components of Retrieval-Augmented Generation?"
        
    ]
    
    # Option 1: Using the class-based approach (recommended)
    print("\n" + "="*70)
    print("USING CLASS-BASED RAG PIPELINE")
    print("="*70)
    
    try:
        # Initialize pipeline
        rag = RAGPipeline(api_key=OPENAI_API_KEY)
        rag.initialize_documents(PDF_DIR)
        
        # Run queries
        class_results = []
        for query in test_queries:
            result = rag.query(query, k=4)
            class_results.append(result)
            print("\n" + "="*70 + "\n")
        
        # Display summary
        display_results_summary(class_results, "Class-based Pipeline Results")
        
    except Exception as e:
        print(f"Error with class-based pipeline: {e}")

def display_results_summary(results, title):
    """Display a summary of the results"""
    if not results:
        print("No results to display")
        return
    
    print(f"\n=== {title.upper()} ===")
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
            
        summary_data.append({
            "Query": result["query"],
            "Answer": result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"],
            "Retrieval Time (s)": result["metrics"]["retrieval_time"],
            "Processing Time (s)": result["metrics"]["processing_time"],
            "Prompt Time (s)": result["metrics"]["prompt_time"],
            "Generation Time (s)": result["metrics"]["generation_time"],
            "Total Time (s)": result["metrics"]["total_time"]
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Calculate average times
        avg_times = {
            "Avg Retrieval": summary_df["Retrieval Time (s)"].mean(),
            "Avg Processing": summary_df["Processing Time (s)"].mean(),
            "Avg Prompt": summary_df["Prompt Time (s)"].mean(),
            "Avg Generation": summary_df["Generation Time (s)"].mean(),
            "Avg Total": summary_df["Total Time (s)"].mean()
        }
        
        print(f"\nAverage Performance Metrics:")
        for metric, value in avg_times.items():
            print(f"- {metric}: {value:.4f}s")
    

def interactive_mode():
    """Run the pipeline in interactive mode"""
    print("Starting Interactive RAG Mode...")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        print("No API key provided. Exiting.")
        return
    
    PDF_DIR = input("\nEnter PDF directory path (default: ./pdfs): ").strip() or "./pdfs"

    # --- Get model name input ---
    model_input = input("\nEnter Gemini model name (default: gemini-1.5-flash-latest): ").strip()
    
    selected_answer_model = model_input if model_input else 'gemini-1.5-flash-latest' 

    # Setup
    pdf_directory(PDF_DIR)
    pdf_files = list_pdf_files(PDF_DIR)
    
    if not pdf_files:
        print("No PDF files found. Please add some PDFs to the directory.")
        return
    
    # Initialize pipeline
    try:
        rag = RAGPipeline(api_key=OPENAI_API_KEY, answer_model_name=selected_answer_model)
        rag.initialize_documents(pdf_directory=PDF_DIR)
        print("Pipeline initialized successfully!")
        
        # Interactive loop
        while True:
            print("\n" + "-"*50)
            query = input("Enter your question (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            try:
                result = rag.query(query, k=3)
                print(f"\nAnswer: {result['answer']}")
            except Exception as e:
                print(f"Error processing query: {e}")
                
    except Exception as e:
        print(f"Error initializing pipeline: {e}")



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
