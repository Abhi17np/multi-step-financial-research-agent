import { useState } from "react";

const API_URL = "http://127.0.0.1:8000/query";

function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-1">Multi-Step Financial Research Agent</h1>
        <p className="text-slate-500 mb-6">Agentic RAG over Salesforce, ServiceNow, Workday, and Adobe 10-K filings</p>

        <div className="flex gap-3 mb-8">
          <input
            className="flex-1 border border-slate-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="e.g. How did Salesforce's and Workday's AI strategies differ?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />
          <button
            className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50"
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>

        {error && (
          <div className="bg-red-50 text-red-700 border border-red-200 rounded-lg p-4 mb-6">
            Error: {error}
          </div>
        )}

        {result && (
          <div className="space-y-6">
            {/* Reasoning trace */}
            <div>
              <h2 className="text-lg font-semibold mb-3 text-slate-700">Reasoning Trace</h2>
              <div className="grid gap-4 md:grid-cols-2">
                {result.trace.map((step, i) => (
                  <div key={i} className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold px-2 py-1 bg-slate-100 rounded text-slate-600">
                        {step.ticker}
                      </span>
                      <span
                        className={`text-xs font-medium px-2 py-1 rounded ${
                          step.verified_sufficient
                            ? "bg-green-100 text-green-700"
                            : "bg-amber-100 text-amber-700"
                        }`}
                      >
                        {step.verified_sufficient ? "Verified" : "Uncertain"} · {step.attempts} attempt{step.attempts > 1 ? "s" : ""}
                      </span>
                    </div>
                    <p className="font-medium text-sm mb-2">{step.sub_question}</p>
                    <p className="text-xs text-slate-500 mb-3">{step.verification_reason}</p>
                    <details className="text-xs text-slate-600">
                      <summary className="cursor-pointer font-medium text-slate-500">
                        Top retrieved evidence ({step.top_chunks.length})
                      </summary>
                      <div className="mt-2 space-y-2">
                        {step.top_chunks.map((chunk, j) => (
                          <div key={j} className="bg-slate-50 rounded p-2">
                            {chunk.text}...
                          </div>
                        ))}
                      </div>
                    </details>
                  </div>
                ))}
              </div>
            </div>

            {/* Final answer */}
            <div>
              <h2 className="text-lg font-semibold mb-3 text-slate-700">Final Answer</h2>
              <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm whitespace-pre-line leading-relaxed">
                {result.final_answer}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;