from bs4 import BeautifulSoup
import os
import re

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

TICKERS = ["CRM", "NOW", "WDAY", "ADBE"]


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
    SEC HTML filings often have an XBRL metadata block glued to the front
    (dates, currency codes, us-gaap: tags) before the real human-readable
    document starts. We find the LAST occurrence of 'ITEM 1.' followed by
    'BUSINESS' (the real Part I opening), since the table of contents
    also mentions 'PART I' / 'ITEM 1.' earlier but that's just a listing.
    """
    # Find all positions where "ITEM 1." and "BUSINESS" appear close together
    matches = [m.start() for m in re.finditer(r"ITEM\s+1\.\s+BUSINESS", text, re.IGNORECASE)]

    if not matches:
        print("WARNING: Could not find 'ITEM 1. BUSINESS' marker — returning full text uncut")
        return text

    # The LAST match is the real section start (earlier ones are just TOC references)
    real_start = matches[-1]
    return text[real_start:]


def process_all():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for ticker in TICKERS:
        raw_path = f"{RAW_DIR}/{ticker}_10k.html"
        out_path = f"{PROCESSED_DIR}/{ticker}_10k.txt"

        print(f"Cleaning {ticker}...")
        with open(raw_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        clean_text = clean_html_to_text(html_content)
        trimmed_text = trim_to_actual_filing(clean_text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(trimmed_text)

        print(f"Saved: {out_path} ({len(trimmed_text)} characters, trimmed from {len(clean_text)})")


if __name__ == "__main__":
    process_all()