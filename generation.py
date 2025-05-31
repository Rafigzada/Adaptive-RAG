import time
import google.generativeai as genai
from typing import List, Dict, Tuple
import requests
import google.api_core.exceptions


class AnswerGenerator:
    def __init__(self, api_key: str, default_model: str = "gemini-1.5-flash-latest"):
        """
        Initializes the AnswerGenerator with a Gemini API key and configures the SDK.
        """
        self.api_key = api_key
        self.default_model = default_model
        genai.configure(api_key=self.api_key)

    def _make_gemini_api_call(self, messages: List[Dict[str, str]],
                              model: str,
                              max_tokens: int,
                              temperature: float) -> Tuple[str, float]:
        """
        Internal helper method to make a call to the Gemini API using the SDK.
        """
        print(f"\n=== GENERATION PHASE (Gemini SDK) ===")
        start_time = time.time()
        
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            print(f"Calling Gemini API with SDK (model: {model})...")
            model_instance = genai.GenerativeModel(model_name=model)

            response = model_instance.generate_content(
                messages,
                generation_config=generation_config
            )

            answer = response.text.strip()

            generation_time = time.time() - start_time
            print(f"Generated answer in {generation_time:.4f} seconds")
            return answer, generation_time

        except google.api_core.exceptions.GoogleAPIError as e:
            print(f"Error with Gemini API (SDK): {e}")
            return f"Error generating response: {str(e)}", time.time() - start_time
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return f"Error generating response: {str(e)}", time.time() - start_time


    def generate_answer_gemini(self, prompt: str, model: str = None,
                               max_tokens: int = 500, temperature: float = 0.2) -> Tuple[str, float]:
        """
        Generate an answer using the Gemini API via SDK, maintaining the original function name.
        Note: This function now uses 'gemini-1.5-flash-latest' by default, despite 'gpt4' in its name.
        """
        # Use the provided model or fallback to the instance's default model
        actual_model = model if model is not None else self.default_model

        messages = [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
        return self._make_gemini_api_call(messages, actual_model, max_tokens, temperature)

    def generate_answer_with_messages(self, messages: List[Dict[str, str]],
                                      model: str = None,
                                      max_tokens: int = 500,
                                      temperature: float = 0.2) -> Tuple[str, float]:
        """
        Generate an answer using pre-formatted chat messages via the Gemini API SDK.
        The 'messages' list should be in the format expected by the Gemini API
        (e.g., {"role": "user", "parts": [{"text": "..."}]}).
        """
        # Use the provided model or fallback to the instance's default model
        actual_model = model if model is not None else self.default_model

        return self._make_gemini_api_call(messages, actual_model, max_tokens, temperature)

    def generate_streaming_answer(self, messages: List[Dict[str, str]],
                                      model: str = None,
                                      max_tokens: int = 500,
                                      temperature: float = 0.2) -> Tuple[str, float]:
        """
        Generate a non-streaming answer using the Gemini API SDK, maintaining the original function name.
        Note: The Gemini SDK's generate_content method for non-streaming calls returns the full response.
        """
        # Use the provided model or fallback to the instance's default model
        actual_model = model if model is not None else self.default_model

        return self._make_gemini_api_call(messages, actual_model, max_tokens, temperature)