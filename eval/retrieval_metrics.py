from agent.orchestrator import retrieve_for_all_subquestions
from eval.test_questions import TEST_QUESTIONS
import re


def compute_system_metrics():
    results = []

    for item in TEST_QUESTIONS:
        query = item["question"]
        print(f"Running: {query}")

        sub_results = retrieve_for_all_subquestions(query)

        for r in sub_results:
            results.append({
                "question": query,
                "sub_question": r["original_sub_question"],
                "attempts": r["attempts"],
                "first_try_sufficient": r["attempts"] == 1
            })

    total = len(results)
    first_try_pass_rate = sum(1 for r in results if r["first_try_sufficient"]) / total
    avg_attempts = sum(r["attempts"] for r in results) / total

    print(f"\n=== System Metrics (across {total} sub-questions) ===")
    print(f"First-attempt verification pass rate: {first_try_pass_rate:.2%}")
    print(f"Average retries per sub-question: {avg_attempts:.2f}")

    return {
        "first_try_pass_rate": first_try_pass_rate,
        "avg_attempts": avg_attempts,
        "total_sub_questions": total
    }


if __name__ == "__main__":
    compute_system_metrics()