"""
Step 1 of the news pipeline: download and cache EODHD news for every ticker.

    caffeinate -i python scripts/download_news.py > logs/download_news.log 2>&1

- Run from the repository root.
- Requires EODHD_API_KEY, from the environment or .env (never hard-code the key).
- Safe to stop and restart: each ticker-year is cached as a CSV, and cached
  years are skipped. Failed requests are NOT cached, so a rerun retries them.
- Scoring happens separately in score_sentiment.py.
"""
import os
import sys
import time

import pandas as pd
import requests

# ---------------------------------------------------------------- settings
PANEL_PATH = 'outputs/v3/data/features_df.csv'   # daily panel with Date and Ticker columns
CACHE_DIR = 'data/news_cache_eodhd'
START, END = '2019-01-01', '2026-12-31'

NEWS_URL = 'https://eodhd.com/api/news'
PAGE_SIZE = 1000          # articles per request
MAX_OFFSET = 10000        # EODHD returns 422 for offsets at or above this
PAUSE_PAGE = 0.5          # seconds between pages
PAUSE_YEAR = 1.0          # seconds between ticker-years
RETRIES = 3

COLUMNS = ['published', 'headline', 'summary', 'eod_polarity', 'n_symbols']


def fetch_page(symbol, start, end, offset, key):
    """One page of results. Retries transient errors; 404 means no articles."""
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(NEWS_URL, params={
                's': symbol, 'from': start, 'to': end,
                'limit': PAGE_SIZE, 'offset': offset,
                'api_token': key, 'fmt': 'json'}, timeout=60)
            if r.status_code == 404:
                return []
            if r.status_code in (429, 500, 502, 503, 504) and attempt < RETRIES:
                time.sleep(10 * attempt)          # back off on rate limits / server errors
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == RETRIES:
                raise
            time.sleep(5 * attempt)


def fetch_year(symbol, start, end, key):
    """All articles in [start, end]. EODHD rejects offsets >= MAX_OFFSET (422),
    so a range with more articles than that is split in half and fetched again."""
    rows, offset = [], 0
    while offset < MAX_OFFSET:
        batch = fetch_page(symbol, start, end, offset, key)
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE
        time.sleep(PAUSE_PAGE)

    d0, d1 = pd.Timestamp(start), pd.Timestamp(end)
    if d0 == d1:                                  # a single day over the cap: keep what we got
        return rows
    mid = d0 + (d1 - d0) / 2
    mid = mid.normalize()
    return (fetch_year(symbol, start, mid.strftime('%Y-%m-%d'), key)
            + fetch_year(symbol, (mid + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), end, key))


def to_frame(items):
    return pd.DataFrame([{
        'published':    item.get('date', ''),
        'headline':     str(item.get('title') or ''),
        'summary':      str(item.get('content') or '')[:500],
        'eod_polarity': (item.get('sentiment') or {}).get('polarity'),
        'n_symbols':    len(item.get('symbols') or []),
    } for item in items], columns=COLUMNS)


def main():
    from dotenv import load_dotenv
    load_dotenv('.env')
    key = os.environ.get('EODHD_API_KEY')
    if not key:
        sys.exit('Set the EODHD_API_KEY environment variable first.')
    os.makedirs(CACHE_DIR, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, usecols=['Date', 'Ticker'], parse_dates=['Date'])
    panel = panel[(panel['Date'] >= START) & (panel['Date'] <= END)]
    ranges = panel.groupby('Ticker')['Date'].agg(['min', 'max'])
    print(f'{len(ranges)} tickers, {START} to {END}', flush=True)

    failed, t0 = [], time.time()
    for i, (ticker, (d_min, d_max)) in enumerate(ranges.iterrows(), 1):
        symbol = f'{ticker}.US'                                  # e.g. NVDA.US, BRK-B.US
        first_year = d_min.replace(month=1, day=1)
        for y_start in pd.date_range(first_year, d_max, freq='YS'):
            path = f'{CACHE_DIR}/{ticker}_{y_start.year}.csv'
            if os.path.exists(path):
                continue
            y_from = max(y_start, d_min)
            y_to = min(y_start + pd.DateOffset(years=1) - pd.Timedelta(days=1), d_max)
            try:
                items = fetch_year(symbol, y_from.strftime('%Y-%m-%d'), y_to.strftime('%Y-%m-%d'), key)
            except Exception as e:
                msg = str(e).replace(key, '***')                     # keep the API key out of the log
                if getattr(getattr(e, 'response', None), 'status_code', None) == 402:
                    sys.exit(f'  {ticker} {y_start.year}: 402 Payment Required (API quota used up). '
                             f'Stopping; rerun once the quota resets.')
                failed.append(f'{ticker}_{y_start.year}')
                print(f'  {ticker} {y_start.year}: FAILED ({msg}); will retry on next run', flush=True)
                continue
            to_frame(items).to_csv(path, index=False)            # empty years are cached too
            print(f'  {ticker} {y_start.year}: {len(items)} articles', flush=True)
            time.sleep(PAUSE_YEAR)

        if i % 25 == 0:
            mins = (time.time() - t0) / 60
            print(f'{i}/{len(ranges)} tickers done ({mins:.0f} min elapsed)', flush=True)

    print(f'Finished in {(time.time() - t0) / 60:.0f} min. {len(failed)} failed: {failed}', flush=True)


if __name__ == '__main__':
    main()
