




import time
import google.generativeai as genai

def generate_answer_gemini(prompt, api_key):
    """
    Generate an answer using Google's Gemini API.

    Args:
        prompt (str): The prompt that includes the context and the query.
        api_key (str): Your Google Gemini API key.

    Returns:
        answer (str): The generated answer from Gemini.
        generation_time (float): Time taken to generate the answer.
    """
    print(f"\n=== GENERATION PHASE ===")
    start_time = time.time()

    # Setup Gemini API
    genai.configure(api_key=api_key)

    # Specify the model
    model_name = 'gemini-2.0-flash'  # Use a constant for the model name
    try:
        # Make API call to Gemini
        print(f"Calling Gemini API with model: {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        # Extract and return the answer from the response
        if response.text:
            answer = response.text
        else:
            answer = "No response from the model."

        generation_time = time.time() - start_time
        print(f"Generated answer in {generation_time:.4f} seconds")
        return answer, generation_time

    except Exception as e:
        print(f"Error with Gemini API: {e}")
        answer = f"Error generating response: {str(e)}"
        generation_time = time.time() - start_time
        return answer, generation_time


