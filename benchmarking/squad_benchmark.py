# squad_benchmark.py
import os
import sys
from datasets import load_dataset
from typing import List, Dict

# Adjust path to ensure rag_pipeline is found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_pipeline import RAGPipeline


def load_squad_samples(n: int = 20):
    dataset = load_dataset("squad", split=f"validation[:{n}]")
    return dataset


def run_squad_benchmark(api_key: str, top_k: int = 3, sample_size: int = 20, direct_retrieval: bool = True) -> List[Dict]:
    """
    Runs a benchmark against the SQuAD dataset using the RAG pipeline.

    Args:
        api_key (str): Your API key for the generative model.
        top_k (int): The number of final relevant documents to use for answer generation.
        sample_size (int): The number of SQuAD samples to use for benchmarking.
        direct_retrieval (bool): If True, the RAG pipeline will bypass full document
                                  relevance check and summarization steps, going
                                  directly to chunk retrieval. Recommended for SQuAD.
    Returns:
        List[Dict]: A list of dictionaries containing benchmark results.
    """
    # Load questions and answers
    squad_data = load_dataset("squad", split=f"validation[:{sample_size}]")

    # Init RAG
    rag = RAGPipeline(api_key=api_key)
    rag.initialize_wikipedia()  # Uses Wikipedia as the corpus

    results = []

    print(f"\n--- Running SQuAD Benchmark with direct_retrieval={direct_retrieval} ---")

    for i, example in enumerate(squad_data):
        question = example["question"]
        expected_answers = example["answers"]["text"]  # list of acceptable answers
        # For SQuAD, the 'context' in the dataset is the *source document*
        # from which the answer comes. The RAG pipeline will retrieve chunks
        # from its own initialized corpus based on the question, not directly use this context.
        # context = example["context"] # This is not directly used by the RAG pipeline's query method

        try:
            # Pass the new direct_retrieval_only flag to the query method
            result = rag.query(question, k=top_k, direct_retrieval_only=direct_retrieval)
            generated = result["answer"]

            matched = any(ans.lower() in generated.lower() for ans in expected_answers)

            results.append({
                "question": question,
                "generated": generated,
                "expected": expected_answers,
                "match": matched,
                "metrics": result["metrics"] # Include metrics for analysis
            })

            print(f"[{i+1}/{sample_size}] ✅ {'✔' if matched else '✘'} {question}")
        except Exception as e:
            print(f"[{i+1}/{sample_size}] ❌ Error: {e}")
            results.append({
                "question": question,
                "generated": f"Error: {e}",
                "expected": expected_answers,
                "match": False,
                "metrics": {}
            })

    return results

def summarize_results(results: List[Dict]):
    correct = sum(1 for r in results if r["match"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    print(f"\n🎯 Benchmark Summary:")
    print(f"- Total samples: {total}")
    print(f"- Correct answers: {correct}")
    print(f"- Accuracy: {accuracy:.2%}")

    # Optional: print average times if metrics are available
    if results and "metrics" in results[0]:
        print("\nAverage Pipeline Metrics:")
        avg_metrics = {key: np.mean([r["metrics"].get(key, 0) for r in results if "metrics" in r])
                       for key in results[0]["metrics"] if isinstance(results[0]["metrics"].get(key), (int, float))}
        
        for key, value in avg_metrics.items():
            print(f"- {key.replace('_', ' ').title()}: {value:.4f}s")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY") # Ensure you're using GEMINI_API_KEY or your actual key env var

    if not API_KEY:
        print("Set your GEMINI_API_KEY in the .env file") # Updated env var name
    else:
        # Run benchmark with direct retrieval for SQuAD
        results = run_squad_benchmark(API_KEY, sample_size=20, direct_retrieval=True)
        summarize_results(results)

        # You can also run it with the full pipeline for comparison if desired
        # print("\n" + "="*70)
        # print("Running SQuAD Benchmark with FULL RAG Pipeline for comparison:")
        # print("="*70)
        # full_pipeline_results = run_squad_benchmark(API_KEY, sample_size=5, direct_retrieval=False)
        # summarize_results(full_pipeline_results)