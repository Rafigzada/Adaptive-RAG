import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, Markdown
import PyPDF2

def generate_answer_gpt4(prompt, api_key):
    """Generate an answer using GPT-4 via OpenAI API"""
    print(f"\n=== GENERATION PHASE ===")
    start_time = time.time()

    # Setup OpenAI client
    client = openai.OpenAI(api_key=api_key)

    try:
        # Make API call
        print("Calling GPT-4 API...")
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # You can change to other models as needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Increased for more detailed responses
            temperature=0.2  # Lower temperature for more factual responses
        )

        # Extract and return answer
        answer = response.choices[0].message.content.strip()
        generation_time = time.time() - start_time
        print(f"Generated answer in {generation_time:.4f} seconds")
        return answer, generation_time

    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return f"Error generating response: {str(e)}", time.time() - start_time