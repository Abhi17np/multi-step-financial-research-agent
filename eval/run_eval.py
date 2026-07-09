from agent.orchestrator import retrieve_for_all_subquestions
from agent.synthesizer import synthesize
from eval.test_questions import TEST_QUESTIONS
from eval.custom_evaluator import score_faithfulness, score_answer_relevancy

import csv
import time

DELAY_BETWEEN_CALLS = 15  # seconds - stays under the 5 requests/minute free tier limit


def run_evaluation():
    results = []

    for item in TEST_QUESTIONS:
        query = item["question"]
        print(f"\nRunning: {query}")

        sub_results = retrieve_for_all_subquestions(query)
        time.sleep(DELAY_BETWEEN_CALLS)

        final_answer = synthesize(query, sub_results)
        time.sleep(DELAY_BETWEEN_CALLS)

        all_contexts = []
        for r in sub_results:
            all_contexts.extend([c["text"] for c in r["chunks"]])

        faithfulness = score_faithfulness(final_answer, all_contexts)
        time.sleep(DELAY_BETWEEN_CALLS)

        relevancy = score_answer_relevancy(query, final_answer)
        time.sleep(DELAY_BETWEEN_CALLS)

        print(f"  Faithfulness: {faithfulness.get('faithfulness_score')}")
        print(f"  Relevancy: {relevancy.get('relevancy_score')}")

        results.append({
            "question": query,
            "faithfulness_score": faithfulness.get("faithfulness_score"),
            "faithfulness_reason": faithfulness.get("reasoning"),
            "relevancy_score": relevancy.get("relevancy_score"),
            "relevancy_reason": relevancy.get("reasoning"),
        })

    with open("eval/eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    avg_faithfulness = sum(r["faithfulness_score"] for r in results if r["faithfulness_score"] is not None) / len(results)
    avg_relevancy = sum(r["relevancy_score"] for r in results if r["relevancy_score"] is not None) / len(results)

    print(f"\n=== Summary ===")
    print(f"Average Faithfulness: {avg_faithfulness:.2f}")
    print(f"Average Relevancy: {avg_relevancy:.2f}")
    print(f"Detailed results saved to eval/eval_results.csv")


if __name__ == "__main__":
    run_evaluation()