"""
Daily Tiingo news collector: builds a point-in-time news history going forward.

Each run fetches the last LOOKBACK_DAYS of articles for every ticker and appends any
new ones to monthly CSVs under data/news_live/, deduplicated by Tiingo article id.
Overlapping lookback windows mean a missed run (or a late-crawled article) is
picked up the next day.

First run: set LOOKBACK_DAYS=90 to backfill everything the Power plan allows
(~3 months of queryable news history). Daily runs after that can use the default 7.

Usage:
    TIINGO_API_KEY=... python scripts/collect_news.py
    LOOKBACK_DAYS=90 TIINGO_API_KEY=... python scripts/collect_news.py   # one-time backfill
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

TICKERS = [
    "NVDA", "AMD", "INTC", "ASML", "TSM", "AVGO", "MRVL", "ARM", "SMCI",
    "MSFT", "GOOGL", "AMZN", "META",
    "PLTR", "AI", "SOUN",
    "DELL", "ANET", "HPE",
]

API_URL = "https://api.tiingo.com/tiingo/news"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "news_live"
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))
PAGE_SIZE = 1000          # Tiingo maximum per request
MAX_PAGES = 10            # safety cap per ticker per run
COLUMNS = ["id", "ticker", "published_at", "crawled_at", "fetched_at",
           "source", "title", "description", "url", "tickers"]


def fetch_ticker(session, ticker, start, end):
    """All articles for one ticker in [start, end], paginated."""
    rows, fetched_at = [], datetime.now(timezone.utc).isoformat(timespec="seconds")
    for page in range(MAX_PAGES):
        resp = session.get(API_URL, params={
            "tickers": ticker.lower(),
            "startDate": start, "endDate": end,
            "limit": PAGE_SIZE, "offset": page * PAGE_SIZE,
            "sortBy": "publishedDate",
        }, timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"{ticker}: unexpected response {batch}")
        for a in batch:
            rows.append({
                "id": a.get("id"),
                "ticker": ticker,
                "published_at": a.get("publishedDate"),
                "crawled_at": a.get("crawlDate"),
                "fetched_at": fetched_at,   # when WE saw it: proof of point-in-time availability
                "source": a.get("source", ""),
                "title": (a.get("title") or "").strip(),
                "description": (a.get("description") or "").strip(),
                "url": a.get("url", ""),
                "tickers": ",".join(a.get("tickers") or []),
            })
        if len(batch) < PAGE_SIZE:
            break
        time.sleep(1)
    return rows


def main():
    key = os.getenv("TIINGO_API_KEY")
    if not key:
        sys.exit("TIINGO_API_KEY is not set")

    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
    end = today.isoformat()

    session = requests.Session()
    session.headers.update({"Authorization": f"Token {key}",
                            "Content-Type": "application/json"})

    new = []
    for t in TICKERS:
        try:
            rows = fetch_ticker(session, t, start, end)
            print(f"{t:6s} {len(rows):5d} articles")
            new.extend(rows)
        except Exception as e:                      # one bad ticker shouldn't kill the run
            print(f"{t:6s} FAILED: {e}")
        time.sleep(1)

    if not new:
        print("No articles fetched.")
        return

    df = pd.DataFrame(new, columns=COLUMNS)
    df["published_at"] = pd.to_datetime(df["published_at"], format="ISO8601", utc=True)

    # Guard against the clamping behavior we saw: drop anything outside the window
    lo = pd.Timestamp(start, tz="UTC")
    df = df[df["published_at"] >= lo]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    for month, part in df.groupby(df["published_at"].dt.strftime("%Y-%m")):
        path = OUT_DIR / f"{month}.csv"
        if path.exists():
            old = pd.read_csv(path, keep_default_na=False)
            old["published_at"] = pd.to_datetime(old["published_at"], format="ISO8601", utc=True)
            before = len(old)
            merged = pd.concat([old, part], ignore_index=True)
        else:
            before = 0
            merged = part
        # same article can be tagged to several of our tickers: keep one row per (id, ticker)
        merged = (merged.sort_values("fetched_at")
                        .drop_duplicates(subset=["id", "ticker"], keep="first")
                        .sort_values(["published_at", "ticker"]))
        merged.to_csv(path, index=False)
        added += len(merged) - before
        print(f"{path.name}: {len(merged)} rows (+{len(merged) - before})")

    print(f"Done. {added} new (article, ticker) rows; window {start} -> {end}.")


if __name__ == "__main__":
    main()
