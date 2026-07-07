import requests
import json
import time
import os

# SEC EDGAR requires a descriptive User-Agent (your real name/email) or it blocks you
HEADERS = {
    "User-Agent": "Abhishek ST research-agent-project abhishekst1713@gmail.com"
}

TICKERS = {
    "CRM": "Salesforce",
    "NOW": "ServiceNow",
    "WDAY": "Workday",
    "ADBE": "Adobe"
}


def get_cik(ticker: str) -> str:
    """
    EDGAR identifies companies by CIK, not ticker.
    This maps ticker -> CIK using SEC's official ticker file.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    for entry in data.values():
        if entry["ticker"] == ticker:
            # CIK must be zero-padded to 10 digits for other API calls
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found")


def get_latest_10k_url(cik: str) -> str:
    """
    Fetches a company's filing history and finds the most recent 10-K.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    filings = resp.json()["filings"]["recent"]

    for i, form in enumerate(filings["form"]):
        if form == "10-K":
            accession = filings["accessionNumber"][i].replace("-", "")
            doc_name = filings["primaryDocument"][i]
            cik_int = cik.lstrip("0")
            return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc_name}"

    raise ValueError("No 10-K found")


def fetch_and_save(ticker: str, company_name: str):
    print(f"Fetching {ticker} ({company_name})...")
    cik = get_cik(ticker)
    doc_url = get_latest_10k_url(cik)

    resp = requests.get(doc_url, headers=HEADERS)
    resp.raise_for_status()

    os.makedirs("data/raw", exist_ok=True)
    out_path = f"data/raw/{ticker}_10k.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

    print(f"Saved: {out_path}")
    time.sleep(0.5)  # be polite to SEC's servers, avoid rate-limiting


if __name__ == "__main__":
    for ticker, name in TICKERS.items():
        fetch_and_save(ticker, name)