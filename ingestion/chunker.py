from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json

PROCESSED_DIR = "data/processed"
CHUNKS_DIR = "data/processed"  # chunks.json will live alongside cleaned text

TICKERS = ["CRM", "NOW", "WDAY", "ADBE"]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_document(text: str, ticker: str) -> list[dict]:
    """
    Splits one company's cleaned 10-K text into chunks.
    Each chunk keeps metadata (ticker, chunk index) so we know
    which company/source it came from later during retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]  # tries paragraph -> sentence -> word -> char
    )

    raw_chunks = splitter.split_text(text)

    chunks_with_metadata = []
    for i, chunk_text in enumerate(raw_chunks):
        chunks_with_metadata.append({
            "id": f"{ticker}_{i}",
            "ticker": ticker,
            "chunk_index": i,
            "text": chunk_text
        })

    return chunks_with_metadata


def process_all():
    all_chunks = []

    for ticker in TICKERS:
        path = f"{PROCESSED_DIR}/{ticker}_10k.txt"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_document(text, ticker)
        all_chunks.extend(chunks)
        print(f"{ticker}: {len(chunks)} chunks created")

    out_path = f"{CHUNKS_DIR}/chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    process_all()