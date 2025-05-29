import time
import openai
from typing import List, Dict, Tuple


class AnswerGenerator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_answer_gpt4(self, prompt: str, model: str = "gpt-4-turbo-preview", 
                           max_tokens: int = 500, temperature: float = 0.2) -> Tuple[str, float]:
        """Generate an answer using GPT-4 via OpenAI API"""
        print(f"\n=== GENERATION PHASE ===")
        start_time = time.time()

        try:
            # Make API call
            print(f"Calling {model} API...")
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Extract and return answer
            answer = response.choices[0].message.content.strip()
            generation_time = time.time() - start_time
            print(f"Generated answer in {generation_time:.4f} seconds")
            return answer, generation_time

        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return f"Error generating response: {str(e)}", time.time() - start_time

    def generate_answer_with_messages(self, messages: List[Dict[str, str]], 
                                    model: str = "gpt-4-turbo-preview",
                                    max_tokens: int = 500, 
                                    temperature: float = 0.2) -> Tuple[str, float]:
        """Generate an answer using pre-formatted chat messages"""
        print(f"\n=== GENERATION PHASE ===")
        start_time = time.time()

        try:
            print(f"Calling {model} API...")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer = response.choices[0].message.content.strip()
            generation_time = time.time() - start_time
            print(f"Generated answer in {generation_time:.4f} seconds")
            return answer, generation_time

        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return f"Error generating response: {str(e)}", time.time() - start_time

    def generate_streaming_answer(self, messages: List[Dict[str, str]], 
                                model: str = "gpt-4-turbo-preview",
                                max_tokens: int = 500, 
                                temperature: float = 0.2):
        """Generate a streaming answer"""
        print(f"\n=== STREAMING GENERATION PHASE ===")
        start_time = time.time()

        try:
            print(f"Starting streaming from {model}...")
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end='', flush=True)
                    full_response += content

            generation_time = time.time() - start_time
            print(f"\nStreaming completed in {generation_time:.4f} seconds")
            return full_response, generation_time

        except Exception as e:
            print(f"Error with OpenAI streaming API: {e}")
            return f"Error generating response: {str(e)}", time.time() - start_time
