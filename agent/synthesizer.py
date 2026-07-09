from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)
MODEL_NAME = config.GEMINI_MODEL_NAME


def build_synthesis_prompt(original_query: str, sub_results: list[dict]) -> str:
    sections = []
    for r in sub_results:
        chunk_texts = "\n".join(
            f"  [{r['ticker']} - chunk {i+1}] {c['text']}"
            for i, c in enumerate(r["chunks"])
        )
        sections.append(
            f"Sub-question: {r['original_sub_question']}\nEvidence:\n{chunk_texts}"
        )

    evidence_block = "\n\n".join(sections)

    return f"""You are a financial research assistant. Answer the user's original question by synthesizing the evidence gathered for each sub-question below.

Rules:
- Directly answer the original question - compare/contrast if it's a comparison question.
- Only use the evidence provided, do not add outside knowledge.
- Cite the company ticker when making claims, e.g. (CRM) or (WDAY).
- Write a coherent, well-organized answer, not just a list of disconnected facts.

Original question: {original_query}

Evidence gathered:
{evidence_block}

Final synthesized answer:"""


def synthesize(original_query: str, sub_results: list[dict]) -> str:
    prompt = build_synthesis_prompt(original_query, sub_results)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


if __name__ == "__main__":
    from agent.orchestrator import retrieve_for_all_subquestions

    test_query = "How did Salesforce's and Workday's AI strategies differ?"
    sub_results = retrieve_for_all_subquestions(test_query)
    final_answer = synthesize(test_query, sub_results)

    print(f"Original question: {test_query}\n")
    print(f"Final Answer:\n{final_answer}")