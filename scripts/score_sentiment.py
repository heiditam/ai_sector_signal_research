"""
Step 2 of the news pipeline: score cached EODHD headlines with FinBERT and
aggregate to one row per (Date, Ticker).

    caffeinate -i python scripts/score_sentiment.py > logs/score_sentiment.log 2>&1
    python scripts/score_sentiment.py --time-only AAPL HRL   # estimate run time first

- Run from the repository root, after download_news.py.
- Uses the Apple Silicon GPU (mps) when available, otherwise the CPU.
- Scores headlines only (fast, and consistent with the FNSPID setup in v1).
- Each ticker's result is cached, so the script can be stopped and restarted.

Timing rule: an article counts for trading day t only if it was published
before that day's 4pm ET close; later articles roll to the next trading day.
Date-only timestamps are treated as published after the close.
"""
import glob
import html as html_lib
import os
import re
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
PANEL_PATH = 'outputs/v3/data/features_df.csv'   # daily panel with Date and Ticker columns
NEWS_CACHE = 'data/news_cache_eodhd'
SENT_CACHE = 'outputs/v3/sentiment_cache'
OUTPUT_PATH = 'outputs/v3/data/sentiment_df.csv'
START, END = '2019-01-01', '2026-12-31'

MAX_SYMBOLS = 3      # drop articles tagged with more tickers (tangential mentions)
BATCH_SIZE = 64
CLOSE_HOUR = 16      # 4pm ET


def clean_news_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)    # HTML tags
    text = html_lib.unescape(text)          # &amp; -> &, &#39; -> '
    text = re.sub(r'\s+', ' ', text)        # whitespace
    return text.strip()


def assign_trading_date(published, trading_days, close_hour=CLOSE_HOUR):
    """First trading day whose close comes after the article's publish time."""
    s = published.astype(str)
    has_time = s.str.contains(r'[T ]\d{1,2}:\d{2}', regex=True)

    ts = pd.to_datetime(s, utc=True, errors='coerce', format='mixed').dt.tz_convert('America/New_York')
    day = ts.dt.tz_localize(None).dt.normalize()
    day = day.where(has_time, pd.to_datetime(s.str[:10], errors='coerce'))   # date-only: use stated date
    after_close = (ts.dt.hour >= close_hour) | ~has_time
    candidate = day + pd.to_timedelta(after_close.astype(int), unit='D')

    td = pd.DatetimeIndex(trading_days).sort_values()
    idx = np.searchsorted(td.values, candidate.values.astype('datetime64[ns]'), side='left')
    out = pd.Series(pd.NaT, index=published.index, dtype='datetime64[ns]')
    valid = candidate.notna().values & (idx < len(td))
    out[valid] = td[idx[valid]]
    return out


def load_news(ticker):
    files = sorted(glob.glob(f'{NEWS_CACHE}/{ticker}_*.csv'))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    df['n_symbols'] = pd.to_numeric(df['n_symbols'], errors='coerce')
    df['eod_polarity'] = pd.to_numeric(df['eod_polarity'], errors='coerce')
    df['headline'] = df['headline'].astype(str).map(clean_news_text)
    df = df[(df['headline'] != '') & (df['n_symbols'] <= MAX_SYMBOLS)]
    return df.drop_duplicates(subset=['published', 'headline'])


def score_ticker(ticker, trading_days, finbert):
    news = load_news(ticker)
    if news.empty:
        return pd.DataFrame()
    news['date'] = assign_trading_date(news['published'], trading_days)
    news = news.dropna(subset=['date'])
    if news.empty:
        return pd.DataFrame()

    results = finbert(news['headline'].tolist(), batch_size=BATCH_SIZE)
    labels = [r['label'].lower() for r in results]
    news['sentiment'] = labels
    news['score'] = [r['score'] if l == 'positive' else -r['score'] if l == 'negative' else 0.0
                     for r, l in zip(results, labels)]

    daily = news.groupby('date').agg(
        news_sentiment=('score', 'mean'),
        news_count=('score', 'count'),
        pct_positive=('sentiment', lambda x: (x == 'positive').mean()),
        eod_sentiment=('eod_polarity', 'mean'),
    )
    daily.index.name = 'Date'
    daily['Ticker'] = ticker
    return daily


def load_finbert():
    import torch
    from transformers import pipeline
    device = 'mps' if torch.backends.mps.is_available() else -1
    print(f'FinBERT device: {device}', flush=True)
    return pipeline('sentiment-analysis', model='ProsusAI/finbert',
                    truncation=True, max_length=128, device=device)   # headlines are short


def main():
    panel = pd.read_csv(PANEL_PATH, usecols=['Date', 'Ticker'], parse_dates=['Date'])
    panel = panel[(panel['Date'] >= START) & (panel['Date'] <= END)]
    trading_days = panel['Date'].drop_duplicates().sort_values()
    tickers = sorted(panel['Ticker'].unique())

    time_only = '--time-only' in sys.argv
    if time_only:
        tickers = [a for a in sys.argv[1:] if not a.startswith('--')] or ['AAPL']

    finbert = load_finbert()
    os.makedirs(SENT_CACHE, exist_ok=True)

    frames, failed, t0, n_articles = [], [], time.time(), 0
    for i, ticker in enumerate(tickers, 1):
        path = f'{SENT_CACHE}/{ticker}_to{END[:4]}.csv'   # rescored when END moves to a new year
        try:
            if os.path.exists(path) and not time_only:
                daily = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
            else:
                t1 = time.time()
                daily = score_ticker(ticker, trading_days, finbert)
                n = int(daily['news_count'].sum()) if not daily.empty else 0
                n_articles += n
                print(f'  {ticker}: {n} articles scored in {time.time() - t1:.0f}s', flush=True)
                if not time_only:
                    (daily if not daily.empty else pd.DataFrame(columns=['Ticker'])).to_csv(path)
            if not daily.empty:
                frames.append(daily)
        except Exception as e:
            failed.append(ticker)
            print(f'  {ticker}: FAILED ({e})', flush=True)
        if i % 25 == 0:
            print(f'{i}/{len(tickers)} tickers done ({(time.time() - t0) / 60:.0f} min)', flush=True)

    elapsed = time.time() - t0
    if time_only:
        rate = n_articles / elapsed if elapsed else 0
        print(f'{n_articles} articles in {elapsed:.0f}s ({rate:.0f}/s). '
              f'Multiply the per-ticker time by {panel["Ticker"].nunique()} tickers for an estimate.')
        return

    if not frames:
        sys.exit(f'No sentiment collected. {len(failed)} tickers failed, e.g. {failed[:5]}')
    out = pd.concat(frames)
    out.index = pd.to_datetime(out.index)
    out.index.name = 'Date'
    out = out.set_index('Ticker', append=True).sort_index()   # (Date, Ticker), matches features_df
    out.to_csv(OUTPUT_PATH)
    print(f'Saved {len(out)} rows for {len(frames)} tickers to {OUTPUT_PATH} '
          f'in {elapsed / 60:.0f} min. {len(failed)} failed: {failed}', flush=True)


if __name__ == '__main__':
    main()
