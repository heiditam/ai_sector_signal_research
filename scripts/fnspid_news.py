"""
fnspid_news.py: historical news for the 19 tickers from the FNSPID dataset.

FNSPID (Dong et al., KDD 2024): ~15.7M financial news records, 1999-2023, license CC BY-NC 4.0
(non-commercial; cite the paper). The news file is very large (tens of GB), so this script
STREAMS it in chunks and keeps only your tickers.

Steps (run from the repo root, one at a time):

  1. Download (resumable; ~tens of GB, do it on Wi-Fi):
       mkdir -p data/fnspid
       curl -L -C - -o data/fnspid/nasdaq_exteral_data.csv \
         https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv
     (the misspelling "exteral" is the real file name)

  2. Filter to your tickers + check coverage:
       python scripts/fnspid_news.py filter
     -> data/fnspid/news_19.parquet  and a ticker x year coverage table

  3. Score headlines with FinBERT and build daily features (slow; run once):
       python scripts/fnspid_news.py score
     -> data/fnspid/news_daily_features.csv   (Date, Ticker, news_sentiment, news_count, pct_positive)

Then merge onto the panel with the SAME one-day lag as the filing features (FNSPID timestamps
are often date-only, so an article dated day t may have appeared after the close).

IMPORTANT: FNSPID ends in 2023, so 2024-2025 have no news. Either restrict news experiments to
folds whose test periods end by 2023, or treat news as missing (not zero) after 2023.
This is for the NEXT version: do not add it to this week's paper.
"""

import os
import sys
import pandas as pd
import numpy as np

TICKERS = ["NVDA", "AMD", "INTC", "ASML", "TSM", "AVGO", "MRVL", "ARM", "SMCI",
           "MSFT", "GOOGL", "AMZN", "META", "PLTR", "AI", "SOUN", "DELL", "ANET", "HPE"]
# Older symbols that may appear in news before renames (e.g., Facebook -> Meta in 2022)
ALIASES = {"FB": "META", "GOOG": "GOOGL"}

RAW = "data/fnspid/nasdaq_exteral_data.csv"
FILTERED = "data/fnspid/news_19.parquet"
DAILY = "data/fnspid/news_daily_features.csv"
START, END = "2019-01-01", "2025-12-31"


def filter_news(raw=RAW, out=FILTERED, chunksize=500_000):
    """Stream the big CSV and keep rows for our tickers in [START, END]."""
    keep_syms = set(TICKERS) | set(ALIASES)
    cols_wanted = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]
    header = pd.read_csv(raw, nrows=0).columns
    usecols = [c for c in cols_wanted if c in header]
    missing = set(cols_wanted) - set(usecols)
    if missing:
        print("Note: columns not in file:", missing, "| available:", list(header))

    parts, seen = [], 0
    for chunk in pd.read_csv(raw, usecols=usecols, chunksize=chunksize,
                             dtype=str, on_bad_lines="skip"):
        seen += len(chunk)
        sym = chunk["Stock_symbol"].str.upper().str.strip()
        chunk = chunk[sym.isin(keep_syms)].copy()
        if chunk.empty:
            continue
        chunk["Ticker"] = chunk["Stock_symbol"].str.upper().str.strip().replace(ALIASES)
        chunk["published"] = pd.to_datetime(chunk["Date"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["published"])
        chunk = chunk[(chunk["published"] >= START) & (chunk["published"] <= END + " 23:59:59")]
        parts.append(chunk.drop(columns=["Stock_symbol"]))
        print(f"\rscanned {seen:,} rows, kept {sum(len(p) for p in parts):,}", end="")
    print()

    news = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    news["Article_title"] = news["Article_title"].fillna("").str.strip()
    news = news[news["Article_title"].str.len() > 0]
    news = news.drop_duplicates(subset=["Ticker", "Article_title", "published"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    news.to_parquet(out, index=False)
    print(f"saved {len(news):,} articles -> {out}")
    coverage_report(news)
    return news


def coverage_report(news):
    """Articles per ticker per year, plus the share of timestamps that are date-only."""
    tab = news.groupby([news["Ticker"], news["published"].dt.year]).size().unstack(fill_value=0)
    tab = tab.reindex(TICKERS, fill_value=0)
    pd.set_option("display.width", 200)
    print("\nArticles per ticker per year:\n", tab.to_string())
    date_only = (news["published"].dt.strftime("%H:%M:%S") == "00:00:00").mean()
    print(f"\nShare of timestamps with no time of day: {date_only:.0%} "
          f"(these must be lagged a full trading day)")
    print("Last article date:", news["published"].max().date())


def score_and_aggregate(filtered=FILTERED, out=DAILY, batch_size=64):
    """FinBERT on headlines -> daily per-ticker features (same definitions as the notebook)."""
    from transformers import pipeline
    news = pd.read_parquet(filtered)
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                       truncation=True, max_length=128)
    titles = news["Article_title"].tolist()
    res = []
    for i in range(0, len(titles), batch_size):
        res.extend(finbert(titles[i:i + batch_size]))
        print(f"\rscored {min(i + batch_size, len(titles)):,}/{len(titles):,}", end="")
    print()
    news["label"] = [r["label"].lower() for r in res]
    news["score"] = [r["score"] if r["label"] == "positive" else
                     -r["score"] if r["label"] == "negative" else 0.0 for r in res]
    # Calendar date in US/Eastern so evening (UTC next-day) articles stay on the right day
    news["Date"] = news["published"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    daily = news.groupby(["Date", "Ticker"]).agg(
        news_sentiment=("score", "mean"),
        news_count=("score", "size"),
        pct_positive=("label", lambda x: (x == "positive").mean()),
    ).reset_index()
    daily.to_csv(out, index=False)
    print(f"saved {len(daily):,} ticker-days -> {out}")
    return daily


def merge_news(panel, daily_path=DAILY, news_end="2023-12-31"):
    """Attach news to a (Date, Ticker)-indexed panel.
    - Weekend/holiday articles roll forward to the next trading day.
    - One-day lag (date-only timestamps may be after the close).
    - After FNSPID ends (news_end), values are NaN (missing), not 0."""
    daily = pd.read_csv(daily_path, parse_dates=["Date"])
    cols = ["news_sentiment", "news_count", "pct_positive"]
    p = panel.reset_index()
    trading = np.sort(p["Date"].unique())
    # map each article date to the next trading day on or after it
    idx = np.searchsorted(trading, daily["Date"].values)
    daily = daily[idx < len(trading)].copy()
    daily["Date"] = trading[idx[idx < len(trading)]]
    daily = daily.groupby(["Date", "Ticker"]).agg(
        news_sentiment=("news_sentiment", "mean"),
        news_count=("news_count", "sum"),
        pct_positive=("pct_positive", "mean")).reset_index()
    p = p.merge(daily, on=["Date", "Ticker"], how="left")
    p["news_count"] = p["news_count"].fillna(0)
    p[["news_sentiment", "pct_positive"]] = p[["news_sentiment", "pct_positive"]].fillna(0)
    p = p.sort_values(["Ticker", "Date"])
    p[cols] = p.groupby("Ticker")[cols].shift(1)                     # one-day lag
    p.loc[p["Date"] > pd.Timestamp(news_end) + pd.Timedelta(days=1), cols] = np.nan
    return p.set_index(["Date", "Ticker"]).sort_index()


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "filter":
        filter_news()
    elif step == "score":
        score_and_aggregate()
    elif step == "coverage":
        coverage_report(pd.read_parquet(FILTERED))
    else:
        print(__doc__)
