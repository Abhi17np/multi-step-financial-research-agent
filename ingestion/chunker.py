from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

PROCESSED_DIR = "data/processed"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_registry():
    with open(f"{PROCESSED_DIR}/documents_registry.json", "r") as f:
        return json.load(f)


def chunk_document(text: str, entry: dict) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    raw_chunks = splitter.split_text(text)

    chunks_with_metadata = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks_with_metadata.append({
            "id": f"{entry['doc_id']}_{i}",
            "ticker": entry["ticker"],
            "company": entry["company"],
            "fiscal_year": entry["fiscal_year"],
            "doc_id": entry["doc_id"],
            "chunk_index": i,
            "text": chunk_text
        })
    return chunks_with_metadata


def process_all():
    registry = load_registry()
    all_chunks = []

    for entry in registry:
        path = f"{PROCESSED_DIR}/{entry['doc_id']}.txt"
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Skipping {entry['doc_id']}: cleaned text not found")
            continue

        chunks = chunk_document(text, entry)
        all_chunks.extend(chunks)
        print(f"{entry['doc_id']}: {len(chunks)} chunks")

    with open(f"{PROCESSED_DIR}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")


if __name__ == "__main__":
    process_all()