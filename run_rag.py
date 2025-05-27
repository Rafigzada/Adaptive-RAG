from rag_implementation import rag_pipeline_pdf

OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with your actual API key

result = rag_pipeline_pdf(
    "What are the main components of Retrieval-Augmented Generation?", 
    api_key=OPENAI_API_KEY,
    pdf_directory="./pdfs"
)

print(result["answer"])