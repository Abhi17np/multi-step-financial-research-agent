from agent.planner import plan
from agent.verifier import verify
from retrieval.hybrid_retriever import hybrid_retrieve  # swapped from retrieval.retriever.retrieve
import config

MAX_RETRIES = config.MAX_VERIFIER_RETRIES


def retrieve_for_all_subquestions(original_query: str, top_k: int = 5) -> list[dict]:
    sub_questions = plan(original_query)

    results = []
    for sq in sub_questions:
        current_query = sq["question"]
        current_top_k = top_k
        ticker = sq["ticker"]

        attempt = 0
        chunks = []
        verification = None

        while attempt <= MAX_RETRIES:
            chunks = hybrid_retrieve(query=current_query, top_k=current_top_k, ticker_filter=ticker)
            verification = verify(current_query, chunks)

            if verification["sufficient"]:
                break

            attempt += 1
            if attempt > MAX_RETRIES:
                print(f"  [!] Max retries reached for '{sq['question']}' - proceeding with best-effort evidence")
                break

            print(f"  [retry {attempt}] Insufficient evidence for '{current_query}' - reformulating...")
            current_query = verification["reformulated_query"] or current_query
            current_top_k += 3

        results.append({
            "original_sub_question": sq["question"],
            "final_query_used": current_query,
            "ticker": ticker,
            "chunks": chunks,
            "verification": verification,
            "attempts": attempt + 1
        })

    return results


if __name__ == "__main__":
    test_query = "How did Salesforce's and Workday's AI strategies differ?"
    results = retrieve_for_all_subquestions(test_query)

    for r in results:
        print(f"\n[{r['ticker']}] Original: {r['original_sub_question']}")
        print(f"  Final query used: {r['final_query_used']}")
        print(f"  Attempts: {r['attempts']}")
        print(f"  Verified sufficient: {r['verification']['sufficient']}")
        print(f"  Chunks retrieved: {len(r['chunks'])}")