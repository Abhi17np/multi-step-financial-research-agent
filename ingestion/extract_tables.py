import pandas as pd
import json
import os
import warnings
from io import StringIO

warnings.filterwarnings("ignore")

RAW_DIR = "data/raw"
TICKERS = ["CRM", "NOW", "WDAY", "ADBE"]

MAX_TABLES_PER_COMPANY = 30
MIN_TABLE_ROWS = 2
MAX_TABLE_CELLS = 500


def is_meaningful_table(df) -> bool:
    if len(df) < MIN_TABLE_ROWS or len(df.columns) < 2:
        return False

    total_cells = len(df) * len(df.columns)
    if total_cells > MAX_TABLE_CELLS:
        return False

    try:
        all_values = df.astype(str).values.flatten()
    except Exception:
        return False

    if any(len(str(v)) > 300 or "<" in str(v) for v in all_values):
        return False

    numeric_count = sum(
        1 for v in all_values
        if any(char.isdigit() for char in str(v)) and len(str(v).strip()) > 0
    )
    ratio = (numeric_count / len(all_values)) if len(all_values) > 0 else 0

    return ratio > 0.10


def extract_tables_from_html(html_path: str, ticker: str) -> list[dict]:
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    try:
        tables = pd.read_html(StringIO(html_content))
    except Exception as e:
        print(f"  Failed to parse tables for {ticker}: {type(e).__name__}")
        return []

    print(f"  Found {len(tables)} raw <table> tags, filtering...")

    table_chunks = []
    count = 0

    for i, df in enumerate(tables):
        if count >= MAX_TABLES_PER_COMPANY:
            break

        try:
            if not is_meaningful_table(df):
                continue

            markdown_table = df.to_markdown(index=False)

            if len(markdown_table) > 5000:
                continue

            table_chunks.append({
                "id": f"{ticker}_table_{i}",
                "ticker": ticker,
                "type": "table",
                "table_index": i,
                "text": f"[Financial table from {ticker} 10-K]\n{markdown_table}"
            })
            count += 1
        except Exception:
            continue

    return table_chunks


def process_all():
    all_table_chunks = []

    for ticker in TICKERS:
        html_path = f"{RAW_DIR}/{ticker}_10k.html"
        print(f"Extracting tables from {ticker}...")
        chunks = extract_tables_from_html(html_path, ticker)
        all_table_chunks.extend(chunks)
        print(f"  Extracted {len(chunks)} usable tables\n")

    out_path = "data/processed/table_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_table_chunks, f, indent=2)

    print(f"Total table chunks: {len(all_table_chunks)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    process_all()