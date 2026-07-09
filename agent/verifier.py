from google import genai
import config
import json

client = genai.Client(api_key=config.GEMINI_API_KEY)
MODEL_NAME = config.GEMINI_MODEL_NAME


def build_verifier_prompt(sub_question: str, chunks: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk['text']}"
        for i, chunk in enumerate(chunks)
    )

    return f"""You are an evidence verifier for a research system.

Sub-question: {sub_question}

Retrieved evidence:
{context_block}

Judge whether this evidence is SUFFICIENT to write a reasonable, factual answer to the sub-question.
Sufficient means: the evidence directly relates to the question and contains concrete, usable facts -
it does NOT need to cover every possible angle or philosophical depth of the topic.
Only mark insufficient if the evidence is genuinely off-topic, missing, or too vague to say anything concrete.

Return ONLY valid JSON, no markdown fences, matching this exact schema:
{{
  "sufficient": true or false,
  "reason": "short explanation",
  "reformulated_query": "a better search query to try if insufficient, else null"
}}

JSON:"""


def verify(sub_question: str, chunks: list[dict]) -> dict:
    prompt = build_verifier_prompt(sub_question, chunks)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"WARNING: Verifier parse failed: {e}")
        # Fail-safe: assume sufficient rather than looping forever on a parse error
        return {"sufficient": True, "reason": "parse error, defaulting to accept", "reformulated_query": None}


if __name__ == "__main__":
    from retrieval.retriever import retrieve

    sub_question = "What is Salesforce's AI strategy?"
    chunks = retrieve(sub_question, top_k=3, ticker_filter="CRM")

    result = verify(sub_question, chunks)
    print(json.dumps(result, indent=2))