from sentence_transformers import SentenceTransformer
import chromadb
import json

CHUNKS_PATH = "data/processed/chunks.json"
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "research_agent_10k"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_and_store():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)

    # Recreate collection fresh each time we re-run ingestion, so old/stale
    # data doesn't linger if chunking logic changes later
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet, that's fine

    collection = client.create_collection(COLLECTION_NAME)

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [{"ticker": chunk["ticker"], "chunk_index": chunk["chunk_index"]} for chunk in chunks]

    print("Generating embeddings (this may take a couple minutes)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    print("Storing in ChromaDB...")
    # Chroma has a max batch size for adds, so we insert in batches to be safe
    BATCH_SIZE = 500
    for i in range(0, len(chunks), BATCH_SIZE):
        collection.add(
            ids=ids[i:i+BATCH_SIZE],
            embeddings=embeddings[i:i+BATCH_SIZE].tolist(),
            documents=texts[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE]
        )
        print(f"  Stored batch {i}-{min(i+BATCH_SIZE, len(chunks))}")

    print(f"\nDone. Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    embed_and_store()