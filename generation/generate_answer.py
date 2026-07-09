from google import genai
import config

client = genai.Client(api_key=config.GEMINI_API_KEY)
MODEL_NAME = config.GEMINI_MODEL_NAME


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source {i+1} - {chunk['ticker']}]\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = f"""You are a financial research assistant. Answer the question using ONLY the context provided below.

Rules:
- If the context does not contain enough information to answer, say "I don't have enough information to answer this."
- Do not use any outside knowledge beyond what's in the context.
- When you make a claim, mention which source number it came from, e.g. (Source 1).

Context:
{context_block}

Question: {query}

Answer:"""

    return prompt


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    prompt = build_prompt(query, retrieved_chunks)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    from retrieval.retriever import retrieve

    test_query = "What is Salesforce's AI strategy?"
    chunks = retrieve(test_query, top_k=5, ticker_filter="CRM")
    answer = generate_answer(test_query, chunks)

    print(f"Query: {test_query}\n")
    print(f"Answer:\n{answer}")