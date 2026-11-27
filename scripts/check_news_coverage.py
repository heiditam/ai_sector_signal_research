"""
Check EODHD news coverage for 2015-2018 before extending the download.

    python scripts/check_news_coverage.py

- Run from the repository root. Requires EODHD_API_KEY (environment or .env).
- Picks 6 tickers spread across news volume (by their 2019 article count),
  downloads 2015-2018 for them, and compares with their cached 2019-2025 years.
- Downloads are saved to the same cache as download_news.py, so they are not
  wasted if you extend the download later. Uses roughly 25-40 API requests.
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from download_news import CACHE_DIR, PAUSE_YEAR, fetch_year, to_frame

NEW_YEARS = [2015, 2016, 2017, 2018]
OLD_YEARS = list(range(2019, 2026))
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95, 1.0]   # light to heavy news coverage


def load_year(ticker, year):
    path = f'{CACHE_DIR}/{ticker}_{year}.csv'
    return pd.read_csv(path) if os.path.exists(path) else None


def summarize(df, year):
    """Articles per year, and the share of weekdays with at least one article."""
    if df is None:
        return None, None
    days = pd.bdate_range(f'{year}-01-01', f'{year}-12-31')
    pub = pd.to_datetime(df['published'], errors='coerce', utc=True).dt.tz_convert(None).dt.normalize()
    return len(df), pub[pub.isin(days)].nunique() / len(days)


def main():
    from dotenv import load_dotenv
    load_dotenv('.env')
    key = os.environ.get('EODHD_API_KEY')
    if not key:
        sys.exit('Set the EODHD_API_KEY environment variable first.')

    # pick tickers spread across news volume, using their 2019 counts
    counts = {f.rsplit('_', 1)[0]: len(pd.read_csv(f'{CACHE_DIR}/{f}'))
              for f in os.listdir(CACHE_DIR) if f.endswith('_2019.csv')}
    counts = pd.Series(counts).sort_values()
    tickers = [counts.index[min(int(q * len(counts)), len(counts) - 1)] for q in QUANTILES]
    print('Sample (2019 articles):', ', '.join(f'{t} ({counts[t]})' for t in tickers), flush=True)

    for t in tickers:
        for y in NEW_YEARS:
            path = f'{CACHE_DIR}/{t}_{y}.csv'
            if os.path.exists(path):
                continue
            try:
                items = fetch_year(f'{t}.US', f'{y}-01-01', f'{y}-12-31', key)
            except Exception as e:
                sys.exit(f'{t} {y}: FAILED ({str(e).replace(key, "***")})')
            to_frame(items).to_csv(path, index=False)
            print(f'  {t} {y}: {len(items)} articles', flush=True)
            time.sleep(PAUSE_YEAR)

    rows = []
    for t in tickers:
        for y in NEW_YEARS + OLD_YEARS:
            n, share = summarize(load_year(t, y), y)
            rows.append({'Ticker': t, 'year': y, 'articles': n, 'days_with_news': share})
    out = pd.DataFrame(rows)

    print('\nArticles per year')
    print(out.pivot(index='Ticker', columns='year', values='articles').loc[tickers].to_string())
    print('\nShare of weekdays with at least one article')
    print(out.pivot(index='Ticker', columns='year', values='days_with_news').loc[tickers].round(2).to_string())


if __name__ == '__main__':
    main()
