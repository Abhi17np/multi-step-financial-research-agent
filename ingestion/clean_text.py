from bs4 import BeautifulSoup
import os
import re
import json

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def load_registry():
    with open(f"{PROCESSED_DIR}/documents_registry.json", "r") as f:
        return json.load(f)


def clean_html_to_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def trim_to_actual_filing(text: str) -> str:
    """
    Finds the real Item 1 Business section by checking the gap to the NEXT
    item heading (Item 1A or Item 2), not just checking total remaining length -
    a real business section has substantial content before the next item;
    a spurious/cross-reference match (e.g. near Item 11-14 boilerplate) does not.
    """
    MIN_SECTION_LENGTH = 5000  # a real Item 1 Business section is always well above this

    item1_matches = [m.start() for m in re.finditer(r"ITEM\s+1\.\s+BUSINESS", text, re.IGNORECASE)]
    if not item1_matches:
        return text

    next_item_pattern = re.compile(r"ITEM\s+1A\.|ITEM\s+2\.", re.IGNORECASE)

    best_match = None
    best_gap_size = 0

    for pos in item1_matches:
        next_match = next_item_pattern.search(text, pos + 20)  # skip ahead of the heading itself
        gap_end = next_match.start() if next_match else len(text)
        gap_size = gap_end - pos

        if gap_size > best_gap_size:
            best_gap_size = gap_size
            best_match = pos

    if best_match is not None and best_gap_size >= MIN_SECTION_LENGTH:
        return text[best_match:]

    return text  # fallback - no confidently-real section found, keep everything


def process_all():
    registry = load_registry()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for entry in registry:
        doc_id = entry["doc_id"]
        raw_path = f"{RAW_DIR}/{doc_id}.html"
        out_path = f"{PROCESSED_DIR}/{doc_id}.txt"

        if not os.path.exists(raw_path):
            print(f"Skipping {doc_id}: raw file not found")
            continue

        print(f"Cleaning {doc_id}...")
        with open(raw_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        clean_text = clean_html_to_text(html_content)
        trimmed_text = trim_to_actual_filing(clean_text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(trimmed_text)

        print(f"  Saved: {out_path} ({len(trimmed_text)} chars)")


if __name__ == "__main__":
    process_all()