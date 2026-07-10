from sentence_transformers import SentenceTransformer
import chromadb
import config

model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=config.VECTOR_STORE_DIR)
collection = client.get_collection(config.COLLECTION_NAME)


def retrieve(query: str, top_k: int = 5, ticker_filter: str = None, fiscal_year_filter: str = None) -> list[dict]:
    """
    Embeds the query and searches ChromaDB for the top_k most similar chunks.
    Supports filtering by ticker and/or fiscal_year - combined via $and when both given.
    """
    query_embedding = model.encode([query]).tolist()

    conditions = []
    if ticker_filter:
        conditions.append({"ticker": ticker_filter})
    if fiscal_year_filter:
        conditions.append({"fiscal_year": fiscal_year_filter})

    if len(conditions) == 2:
        where_clause = {"$and": conditions}
    elif len(conditions) == 1:
        where_clause = conditions[0]
    else:
        where_clause = None

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
            "fiscal_year": results["metadatas"][0][i].get("fiscal_year"),
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


if __name__ == "__main__":
    test_query = "What is Apple's approach to AI?"
    results = retrieve(test_query, top_k=3, ticker_filter="AAPL")
    print(f"Query: {test_query}\n")
    for r in results:
        print(f"[{r['ticker']} FY{r['fiscal_year']}] (distance: {r['distance']:.4f})")
        print(r['text'][:300])
        print("---")