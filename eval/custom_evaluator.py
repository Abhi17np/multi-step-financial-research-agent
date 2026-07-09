from google import genai
import config
import json

client = genai.Client(api_key=config.GEMINI_API_KEY)
MODEL_NAME = config.GEMINI_MODEL_NAME


def score_faithfulness(answer: str, contexts: list[str]) -> dict:
    """
    Checks if every claim in the answer is actually supported by the given contexts.
    Mirrors RAGAS's faithfulness metric: is the model hallucinating beyond its evidence?
    """
    context_block = "\n\n".join(contexts)

    prompt = f"""You are evaluating an AI system's answer for faithfulness to its source evidence.

Source evidence:
{context_block}

Generated answer:
{answer}

Judge: does the answer ONLY make claims that are supported by the evidence above?
Score from 0.0 (completely unsupported/hallucinated) to 1.0 (fully supported by evidence).

Return ONLY valid JSON, no markdown fences:
{{"faithfulness_score": 0.0 to 1.0, "reasoning": "short explanation"}}

JSON:"""

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return _parse_json(response.text)


def score_answer_relevancy(question: str, answer: str) -> dict:
    """
    Checks if the answer actually addresses the question asked.
    Mirrors RAGAS's answer relevancy metric.
    """
    prompt = f"""You are evaluating whether an answer is relevant to the question asked.

Question: {question}

Answer: {answer}

Judge: does the answer directly and completely address the question?
Score from 0.0 (off-topic/irrelevant) to 1.0 (fully relevant and on-point).

Return ONLY valid JSON, no markdown fences:
{{"relevancy_score": 0.0 to 1.0, "reasoning": "short explanation"}}

JSON:"""

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return _parse_json(response.text)


def _parse_json(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"WARNING: judge parse failed: {e}")
        return {"score": None, "reasoning": "parse error"}