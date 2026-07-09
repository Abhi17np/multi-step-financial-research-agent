from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.orchestrator import retrieve_for_all_subquestions
from agent.synthesizer import synthesize

app = FastAPI(title="Multi-Step Financial Research Agent")

# CORS: allows the React frontend (running on a different port, e.g. localhost:5173)
# to actually call this API. Without this, browsers block the request by default.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; in production you'd restrict this to your actual frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(request: QueryRequest):
    """
    Runs the full agentic RAG pipeline and returns the complete trace:
    sub-questions, retrieval/verification per sub-question, and the final answer.
    """
    sub_results = retrieve_for_all_subquestions(request.question)
    final_answer = synthesize(request.question, sub_results)

    # Build a clean trace for the frontend - don't send raw chunk objects,
    # just what's useful to display
    trace = []
    for r in sub_results:
        trace.append({
            "sub_question": r["original_sub_question"],
            "ticker": r["ticker"],
            "final_query_used": r["final_query_used"],
            "attempts": r["attempts"],
            "verified_sufficient": r["verification"]["sufficient"],
            "verification_reason": r["verification"]["reason"],
            "top_chunks": [
                {"ticker": c["ticker"], "text": c["text"][:300], "distance": c["distance"]}
                for c in r["chunks"][:3]
            ]
        })

    return {
        "question": request.question,
        "trace": trace,
        "final_answer": final_answer
    }


@app.get("/")
def health_check():
    return {"status": "Research agent API is running"}