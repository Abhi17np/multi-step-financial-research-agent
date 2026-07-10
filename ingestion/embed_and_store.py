from sentence_transformers import SentenceTransformer
import chromadb
import json
import config

CHUNKS_PATH = "data/processed/chunks.json"
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "research_agent_10k"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 64  # embedding batch size - keeps memory usage reasonable at scale
CHROMA_INSERT_BATCH = 500  # Chroma's per-call insert limit


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

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]
    metadatas = [
        {
            "ticker": chunk["ticker"],
            "company": chunk["company"],
            "fiscal_year": chunk["fiscal_year"],
            "doc_id": chunk["doc_id"],
            "chunk_index": chunk["chunk_index"]
        }
        for chunk in chunks
    ]

    print(f"Generating embeddings for {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    print("This will take a while at this scale - progress bar shows overall status.")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)

    print("Storing in ChromaDB...")
    total = len(chunks)
    for i in range(0, total, CHROMA_INSERT_BATCH):
        end = min(i + CHROMA_INSERT_BATCH, total)
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"  Stored {end}/{total}")

    print(f"\nDone. Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    embed_and_store()