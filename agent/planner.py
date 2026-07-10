from google import genai
import config
import json

client = genai.Client(api_key=config.GEMINI_API_KEY)
MODEL_NAME = config.GEMINI_MODEL_NAME


def load_registry():
    with open("data/processed/documents_registry.json", "r") as f:
        return json.load(f)


REGISTRY = load_registry()
VALID_TICKERS = {entry["ticker"] for entry in REGISTRY}
VALID_YEARS_BY_TICKER = {}
for entry in REGISTRY:
    VALID_YEARS_BY_TICKER.setdefault(entry["ticker"], set()).add(entry["fiscal_year"])

# Build a compact company reference list for the prompt (ticker: company name)
COMPANY_LIST = "\n".join(f"- {e['ticker']} = {e['company']} (fiscal year {e['fiscal_year']} available)" for e in REGISTRY)


def build_planner_prompt(query: str) -> str:
    return f"""You are a query planning assistant for a financial research system.

Your job: break down the user's question into simpler sub-questions, each focused on ONE company and ONE fiscal year.

Companies and fiscal years available in this system:
{COMPANY_LIST}

IMPORTANT: This system ONLY has data for the companies/years listed above. If the user's question
mentions a company NOT in this list, or a fiscal year not available for that company, set that
sub-question's ticker to "UNSUPPORTED" instead of guessing.

Rules:
- If the question is about only one company, return just ONE sub-question.
- If the question compares multiple companies, create one sub-question PER company.
- If no fiscal year is mentioned, use the most recent year available for that company.
- Each sub-question must be a complete, standalone question.
- Return ONLY valid JSON, no markdown fences, matching this exact schema:

{{
  "sub_questions": [
    {{"question": "...", "ticker": "...", "fiscal_year": "..."}}
  ]
}}

User question: {query}

JSON:"""


def plan(query: str) -> list[dict]:
    prompt = build_planner_prompt(query)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").replace("json", "", 1).strip()

    try:
        parsed = json.loads(raw_text)
        return parsed["sub_questions"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"WARNING: Failed to parse planner output: {e}")
        return [{"question": query, "ticker": None, "fiscal_year": None}]


if __name__ == "__main__":
    test_query = "How did Apple's and Microsoft's approaches to AI differ?"
    sub_questions = plan(test_query)

    print(f"Original question: {test_query}\n")
    for sq in sub_questions:
        print(f"  [{sq['ticker']} FY{sq['fiscal_year']}] {sq['question']}")