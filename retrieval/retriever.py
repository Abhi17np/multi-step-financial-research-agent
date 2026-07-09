from sentence_transformers import SentenceTransformer
import chromadb
import config

model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=config.VECTOR_STORE_DIR)
collection = client.get_collection(config.COLLECTION_NAME)


def retrieve(query: str, top_k: int = 5, ticker_filter: str = None) -> list[dict]:
    """
    Embeds the query and searches ChromaDB for the top_k most similar chunks.
    Optional ticker_filter restricts search to one company only (e.g. 'CRM'),
    useful later when the planner generates company-specific sub-questions.
    """
    query_embedding = model.encode([query]).tolist()

    where_clause = {"ticker": ticker_filter} if ticker_filter else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_clause
    )

    retrieved_chunks = []
    for i in range(len(results["ids"][0])):
        retrieved_chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "ticker": results["metadatas"][0][i]["ticker"],
            "distance": results["distances"][0][i]  # lower = more similar
        })

    return retrieved_chunks


if __name__ == "__main__":
    # Quick manual test
    test_query = "What is Salesforce's AI strategy?"
    results = retrieve(test_query, top_k=3)

    print("\n--- Same query, filtered to CRM only ---")
    filtered_results = retrieve(test_query, top_k=3, ticker_filter="CRM")
    for r in filtered_results:
        print(f"[{r['ticker']}] (distance: {r['distance']:.4f})")
        print(r['text'][:300])
        print("---")