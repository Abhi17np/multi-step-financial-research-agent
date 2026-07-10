from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import json
import config
from retrieval.retriever import retrieve as dense_retrieve

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    ALL_CHUNKS = json.load(f)

tokenized_corpus = [chunk["text"].lower().split() for chunk in ALL_CHUNKS]
bm25 = BM25Okapi(tokenized_corpus)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def bm25_search(query: str, top_k: int, ticker_filter: str = None, fiscal_year_filter: str = None) -> list[dict]:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = [
        (chunk, scores[i]) for i, chunk in enumerate(ALL_CHUNKS)
        if (ticker_filter is None or chunk["ticker"] == ticker_filter)
        and (fiscal_year_filter is None or chunk["fiscal_year"] == fiscal_year_filter)
    ]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_k]]


def reciprocal_rank_fusion(dense_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    scores = {}
    chunk_lookup = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_lookup[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_lookup[cid] = chunk

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [chunk_lookup[cid] for cid in ranked_ids]


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored[:top_k]]


def hybrid_retrieve(query: str, top_k: int = 5, ticker_filter: str = None, fiscal_year_filter: str = None, candidate_pool_size: int = 20) -> list[dict]:
    dense_results = dense_retrieve(query, top_k=candidate_pool_size, ticker_filter=ticker_filter, fiscal_year_filter=fiscal_year_filter)
    bm25_results = bm25_search(query, top_k=candidate_pool_size, ticker_filter=ticker_filter, fiscal_year_filter=fiscal_year_filter)

    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    reranked = rerank(query, fused[:candidate_pool_size], top_k=top_k)

    return reranked


if __name__ == "__main__":
    test_query = "What is Apple's approach to AI?"
    results = hybrid_retrieve(test_query, top_k=3, ticker_filter="AAPL")
    print(f"Query: {test_query}\n")
    for r in results:
        print(f"[{r['ticker']}] {r['text'][:200]}...")
        print("---")