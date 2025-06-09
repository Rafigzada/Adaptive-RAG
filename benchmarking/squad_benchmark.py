import os
import sys
from datasets import load_dataset
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_pipeline import RAGPipeline


def load_squad_samples(n: int = 20):
    dataset = load_dataset("squad", split=f"validation[:{n}]")
    return dataset



def run_squad_benchmark(api_key: str, top_k: int = 3, sample_size: int = 20) -> List[Dict]:
    # Load questions and answers
    squad_data = load_dataset("squad", split=f"validation[:{sample_size}]")

    # Init RAG
    rag = RAGPipeline(api_key=api_key)
    rag.initialize_wikipedia()  # Uses Wikipedia

    results = []

    for i, example in enumerate(squad_data):
        question = example["question"]
        expected_answers = example["answers"]["text"]  # list of acceptable answers
        context = example["context"]

        try:
            result = rag.query(question, k=top_k)
            generated = result["answer"]

            matched = any(ans.lower() in generated.lower() for ans in expected_answers)

            results.append({
                "question": question,
                "generated": generated,
                "expected": expected_answers,
                "match": matched
            })

            print(f"[{i+1}/{sample_size}] ✅ {'✔' if matched else '✘'} {question}")
        except Exception as e:
            print(f"[{i+1}/{sample_size}] ❌ Error: {e}")

    return results

def summarize_results(results: List[Dict]):
    correct = sum(1 for r in results if r["match"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print(f"\n🎯 Benchmark Summary:")
    print(f"- Total samples: {total}")
    print(f"- Correct answers: {correct}")
    print(f"- Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")

    if not API_KEY:
        print("Set your OPENAI_API_KEY in the .env file")
    else:
        results = run_squad_benchmark(API_KEY, sample_size=20)
        summarize_results(results)