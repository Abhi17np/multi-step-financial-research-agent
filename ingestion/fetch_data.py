import requests
import json
import time
import os

HEADERS = {
    "User-Agent": "Abhishek ST research-agent-project abhishekst1713@gmail.com"
}

# Top 100 S&P 500 companies by market cap (verified live, as of July 2026)
# Source: slickcharts.com/sp500 - includes FAANG/MAANG and major large-caps across all sectors
TICKERS = {
    "NVDA": "NVIDIA", "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "AVGO": "Broadcom", "META": "Meta Platforms", "TSLA": "Tesla",
    "MU": "Micron Technology", "LLY": "Eli Lilly", "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase",
    "WMT": "Walmart", "AMD": "Advanced Micro Devices", "V": "Visa", "JNJ": "Johnson & Johnson",
    "XOM": "Exxon Mobil", "INTC": "Intel", "AMAT": "Applied Materials", "CSCO": "Cisco Systems",
    "MA": "Mastercard", "LRCX": "Lam Research", "ABBV": "AbbVie", "CAT": "Caterpillar",
    "BAC": "Bank of America", "ORCL": "Oracle", "COST": "Costco Wholesale", "UNH": "UnitedHealth Group",
    "GE": "General Electric", "KO": "Coca-Cola", "MS": "Morgan Stanley", "CVX": "Chevron",
    "PG": "Procter & Gamble", "HD": "Home Depot", "NFLX": "Netflix", "GS": "Goldman Sachs",
    "PLTR": "Palantir Technologies", "MRK": "Merck", "KLAC": "KLA Corp", "DELL": "Dell Technologies",
    "GEV": "GE Vernova", "PM": "Philip Morris International", "TXN": "Texas Instruments", "IBM": "IBM",
    "PANW": "Palo Alto Networks", "SNDK": "Sandisk", "WFC": "Wells Fargo", "RTX": "RTX Corp",
    "LIN": "Linde", "C": "Citigroup", "AXP": "American Express", "ANET": "Arista Networks",
    "MRVL": "Marvell Technology", "CRWD": "CrowdStrike", "QCOM": "Qualcomm", "STX": "Seagate Technology",
    "APH": "Amphenol", "WDC": "Western Digital", "MCD": "McDonald's", "TMUS": "T-Mobile US",
    "AMGN": "Amgen", "TMO": "Thermo Fisher Scientific", "ADI": "Analog Devices", "PEP": "PepsiCo",
    "NEE": "NextEra Energy", "SCHW": "Charles Schwab", "VZ": "Verizon Communications", "BA": "Boeing",
    "APP": "AppLovin", "UNP": "Union Pacific", "GILD": "Gilead Sciences", "DIS": "Walt Disney",
    "TJX": "TJX Companies", "GLW": "Corning", "ABT": "Abbott Laboratories", "WELL": "Welltower",
    "DE": "Deere & Co", "BLK": "BlackRock", "ETN": "Eaton", "UBER": "Uber Technologies",
    "BX": "Blackstone", "T": "AT&T", "ISRG": "Intuitive Surgical", "DHR": "Danaher",
    "PFE": "Pfizer", "BKNG": "Booking Holdings", "CB": "Chubb", "PGR": "Progressive Corp",
    "CRM": "Salesforce", "PLD": "Prologis", "COP": "ConocoPhillips", "CVS": "CVS Health",
    "SPGI": "S&P Global", "VRTX": "Vertex Pharmaceuticals", "SYK": "Stryker", "VRT": "Vertiv Holdings",
    "COF": "Capital One", "SBUX": "Starbucks", "FTNT": "Fortinet", "PH": "Parker-Hannifin"
}

# Note: BRK-B uses a hyphen (SEC's format), not the "BRK.B" display format used on some
# financial sites. A handful of these companies (foreign-domiciled, e.g. Linde) may file
# Form 20-F instead of 10-K, and will be gracefully skipped rather than crash the run.


def get_cik(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    for entry in data.values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found")


def get_latest_10k_url(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    filings = resp.json()["filings"]["recent"]

    for i, form in enumerate(filings["form"]):
        if form == "10-K":
            accession = filings["accessionNumber"][i].replace("-", "")
            doc_name = filings["primaryDocument"][i]
            filing_date = filings["filingDate"][i]
            cik_int = cik.lstrip("0")
            return {
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc_name}",
                "filing_date": filing_date
            }
    raise ValueError("No 10-K found (may file 20-F or other form)")


def fetch_and_save(ticker: str, company_name: str) -> dict | None:
    print(f"Fetching {ticker} ({company_name})...")
    try:
        cik = get_cik(ticker)
        filing = get_latest_10k_url(cik)
    except Exception as e:
        print(f"  Skipped {ticker}: {type(e).__name__}")
        return None

    try:
        resp = requests.get(filing["url"], headers=HEADERS)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Failed to download {ticker}: {type(e).__name__}")
        return None

    os.makedirs("data/raw", exist_ok=True)
    fiscal_year = filing["filing_date"][:4]
    doc_id = f"{ticker}_{fiscal_year}"
    out_path = f"data/raw/{doc_id}.html"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resp.text)

    print(f"  Saved: {out_path}")
    return {"ticker": ticker, "company": company_name, "fiscal_year": fiscal_year, "doc_id": doc_id}


if __name__ == "__main__":
    registry = []
    for ticker, name in TICKERS.items():
        result = fetch_and_save(ticker, name)
        if result:
            registry.append(result)
        time.sleep(0.3)

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/documents_registry.json", "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\nTotal documents successfully fetched: {len(registry)} / {len(TICKERS)} attempted")
    print("Registry saved to data/processed/documents_registry.json")