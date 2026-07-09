# add_tables_to_vectorstore.py (new file in ingestion/)

from sentence_transformers import SentenceTransformer
import chromadb
import json
import config

def add_tables():
    with open("data/processed/table_chunks.json", "r", encoding="utf-8") as f:
        table_chunks = json.load(f)

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=config.VECTOR_STORE_DIR)
    collection = client.get_collection(config.COLLECTION_NAME)

    texts = [c["text"] for c in table_chunks]
    ids = [c["id"] for c in table_chunks]
    metadatas = [{"ticker": c["ticker"], "type": "table"} for c in table_chunks]

    print(f"Embedding {len(table_chunks)} table chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas)
    print(f"Added. New total chunks in collection: {collection.count()}")

if __name__ == "__main__":
    add_tables()