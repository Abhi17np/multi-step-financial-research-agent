from google import genai
import config
import json

client = genai.Client(api_key=config.GEMINI_API_KEY)
MODEL_NAME = config.GEMINI_MODEL_NAME

VALID_TICKERS = ["CRM", "NOW", "WDAY", "ADBE"]


def build_planner_prompt(query: str) -> str:
    return f"""You are a query planning assistant for a financial research system.

Your job: break down the user's question into simpler sub-questions, each focused on ONE company.

Available companies (tickers): {", ".join(VALID_TICKERS)}
- CRM = Salesforce
- NOW = ServiceNow
- WDAY = Workday
- ADBE = Adobe

Rules:
- If the question is about only one company, return just ONE sub-question.
- If the question compares multiple companies, create one sub-question PER company.
- Each sub-question must be a complete, standalone question (don't just say "same for X" - rewrite it fully).
- ticker must be one of: {", ".join(VALID_TICKERS)}
- Return ONLY valid JSON, no other text, no markdown code fences, matching this exact schema:

{{
  "sub_questions": [
    {{"question": "...", "ticker": "..."}},
    {{"question": "...", "ticker": "..."}}
  ]
}}

User question: {query}

JSON:"""


def plan(query: str) -> list[dict]:
    prompt = build_planner_prompt(query)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    raw_text = response.text.strip()

    # Defensive cleanup - sometimes LLMs wrap JSON in ```json fences despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        parsed = json.loads(raw_text)
        return parsed["sub_questions"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"WARNING: Failed to parse planner output: {e}")
        print(f"Raw output was: {raw_text}")
        # Fallback: treat the whole query as a single sub-question, no ticker filter
        return [{"question": query, "ticker": None}]


if __name__ == "__main__":
    test_query = "How did Salesforce's and Workday's AI strategies differ?"
    sub_questions = plan(test_query)

    print(f"Original question: {test_query}\n")
    print("Decomposed into:")
    for sq in sub_questions:
        print(f"  - [{sq['ticker']}] {sq['question']}")