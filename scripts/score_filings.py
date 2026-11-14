"""
Download and score SEC 8-K earnings releases (Item 2.02, Exhibit 99) for every ticker.

    caffeinate -i python3 scripts/score_filings.py > logs/score_filings.log 2>&1
    python3 scripts/score_filings.py --time-only NVDA MMM     # estimate run time first

- Run from the repository root, after data_cleaning_v3.ipynb has saved sp500_tickers.csv.
- Requires EMAIL in .env (the SEC asks every client to identify itself).
- Two caches, one file per ticker, so the script can be stopped and restarted:
    outputs/v3/filing_text/    cleaned exhibit text. Downloaded once (~3 hours for ~500 tickers).
    outputs/v3/filing_scores/  topic and tone scores. Delete this folder after changing TOPICS,
                               THRESHOLD, or the sentence filter; the saved text is reused.
- Writes outputs/v3/data/filing_signals_df.csv: one row per filing, with the time the SEC
  accepted it. data_cleaning_v3.ipynb moves after-close filings to the next trading day.
"""
import os
import re
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
UNIVERSE_PATH = 'outputs/v3/data/sp500_tickers.csv'
TEXT_CACHE = 'outputs/v3/filing_text'
SCORE_CACHE = 'outputs/v3/filing_scores'
OUTPUT_PATH = 'outputs/v3/data/filing_signals_df.csv'
START, END = '2018-01-01', date.today().strftime('%Y-%m-%d')   # 2018 gives a prior filing for early 2019

NLI_MODEL = 'MoritzLaurer/deberta-v3-base-zeroshot-v2.0'
THRESHOLD = 0.8          # NLI entailment probability needed to count a sentence toward a topic
MAX_CANDIDATES = 25      # sentences per filing sent to the NLI model

# Each topic is a plain-English hypothesis the model checks against a sentence
TOPICS = {
    'ai_demand_up':   'The company reports growing customer demand for its AI products or services.',
    'ai_demand_down': 'The company reports weakening demand for its AI products or services.',
    'capex_up':       'The company is increasing its capital spending or investment.',
    'capex_down':     'The company is reducing or delaying its capital spending or investment.',
    'competition':    'The company faces competition from rival chips or alternative suppliers.',
    'supply_limited': 'The company is constrained by limited supply of chips, GPUs, or capacity.',
}
LABEL_TO_TOPIC = {v: k for k, v in TOPICS.items()}

# Domain keyword dictionaries (used only to pick candidate sentences)
NVIDIA_KEYWORDS = [
    'h100', 'h200', 'h20',
    'blackwell', 'b100', 'b200', 'gb200', 'gb300',
    'rubin', 'vera rubin', 'nvl72',
    'dgx', 'hgx',
    'cuda', 'tensorrt', 'triton', 'nim microservices',
    'hopper architecture', 'blackwell architecture',
]
AMD_KEYWORDS = [
    'instinct mi300', 'instinct mi300x', 'instinct mi325',
    'instinct mi350', 'instinct mi400',
    'instinct mi250', 'instinct mi200',
    'radeon pro', 'radeon instinct',
    'rocm', 'hip framework', 'amd cdna',
    'amd rdna', 'amd infinity fabric',
    'amd gpu demand', 'amd accelerator',
    'amd data center gpu', 'amd ai accelerator',
    'amd compute demand', 'amd inference',
    'amd training workload', 'amd cluster',
]
INTEL_KEYWORDS = [
    'gaudi2', 'gaudi 2', 'gaudi3', 'gaudi 3',
    'habana labs', 'habana gaudi',
    'ponte vecchio', 'max series gpu',
    'intel gpu max', 'flex series gpu',
    'openvino', 'intel oneapi', 'intel neural compressor',
    'intel xeon ai', 'intel core ultra ai',
    'meteor lake ai', 'lunar lake ai', 'arrow lake ai', 'intel npu',
    'intel data center gpu', 'intel ai accelerator',
    'intel compute demand', 'intel ai inference',
    'intel training workload', 'gaudi deployment', 'gaudi adoption', 'gaudi demand',
]
GENERAL_GPU_KEYWORDS = [
    'gpu demand', 'gpu supply', 'gpu allocation',
    'data center revenue', 'data center demand',
    'inference demand', 'inference workload',
    'training compute', 'compute demand',
    'cloud ai', 'ai infrastructure', 'ai accelerator',
    'tensor processing', 'custom ai chip',
]
CAPEX_UP_KEYWORDS = [
    'increasing capex', 'raised capex', 'accelerating capex',
    'accelerating investment', 'capital expenditure increase',
    'expanding capacity', 'record investment',
    'investing heavily', 'multiyear investment',
    'committed to investing', 'significant investment in ai',
    'capital investment', 'infrastructure investment', 'ai investment',
]
CAPEX_DOWN_KEYWORDS = [
    'reducing capex', 'cutting investment', 'slowing spend',
    'deferring investment', 'pausing investment',
    'capital efficiency', 'disciplined spending',
    'revisiting capex', 'moderating investment',
]
COMPETITIVE_RISK_KEYWORDS = [
    'amd instinct', 'gaudi', 'trainium', 'inferentia', 'maia',
    'google tpu', 'axion',
    'custom silicon', 'in-house chip', 'in-house ai chip',
    'merchant silicon', 'proprietary chip',
    'competitive alternative', 'alternative to nvidia',
    'amd alternative', 'nvidia alternative',
    'vendor diversification', 'multi vendor gpu',
]
ALL_KEYWORDS = (NVIDIA_KEYWORDS + AMD_KEYWORDS + INTEL_KEYWORDS + GENERAL_GPU_KEYWORDS
                + CAPEX_UP_KEYWORDS + CAPEX_DOWN_KEYWORDS + COMPETITIVE_RISK_KEYWORDS)

# Candidate sentences: AI, chips, data centers, or capital spending. Common words such as
# "demand" or "invest" are left out, because nearly every earnings sentence contains them.
CANDIDATE_RE = re.compile(
    r'\b(?:' + '|'.join(map(re.escape, ALL_KEYWORDS)) +
    r'|artificial intelligence|ai|generative|machine learning|gpus?|accelerators?|chips?'
    r'|data cent(?:er|re)s?|capex|capital expenditures?|capital spending)\b', re.I)
AI_RE = re.compile(r'\b(?:artificial intelligence|ai|generative|gpus?|accelerators?)\b', re.I)

SIGNAL_COLS = ([f'{k}_share' for k in TOPICS]
               + ['ai_demand_net', 'capex_net', 'ai_tone', 'ai_sentence_ratio'])


# ---------------------------------------------------------------- text
def clean_filing_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)          # remove HTML tags
    text = re.sub(r'&[a-z]+;', ' ', text)          # remove HTML entities
    text = re.sub(r'\s+', ' ', text)               # normalize whitespace
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)  # remove dates
    text = re.sub(r'\$[\d,\.]+', '', text)         # remove dollar amounts
    text = re.sub(r"['']", '', text)               # drop apostrophes
    text = re.sub(r'[^\w\s\.]', ' ', text)         # remove special characters
    text = re.sub(r'\s{2,}', ' ', text)            # normalize whitespace again
    return text.strip()


def split_sentences(text):
    text = re.sub(r'\s+', ' ', text)
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)     # doesn't split "U.S." or "3.5"
    return [s.strip() for s in sents if 30 <= len(s) <= 1000]


def fetch_filings(ticker):
    """Cleaned Exhibit 99 text of every Item 2.02 8-K. Failed filings are skipped."""
    from edgar import Company
    filings = Company(ticker).get_filings(form='8-K', filing_date=f'{START}:{END}')
    rows = []
    for filing in filings:
        try:
            if '2.02' not in str(filing.items):              # only earnings releases
                continue
            for attachment in filing.attachments:
                if 'EX-99' not in attachment.document_type:
                    continue
                text = attachment.text()
                if not text or len(text) < 200:
                    continue
                rows.append({'Ticker': ticker,
                             'Date': pd.to_datetime(filing.filing_date),
                             'Accepted': pd.Timestamp(filing.acceptance_datetime),   # UTC
                             'Text': clean_filing_text(text)})
                break                                        # one exhibit per filing
        except Exception as e:
            print(f'  skipped {ticker} {filing.filing_date}: {e}', flush=True)
    return pd.DataFrame(rows, columns=['Ticker', 'Date', 'Accepted', 'Text'])


def load_text(ticker):
    path = f'{TEXT_CACHE}/{ticker}.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    texts = fetch_filings(ticker)
    texts.to_parquet(path, index=False)                      # tickers with no filings are cached too
    return texts


# ---------------------------------------------------------------- scoring
def load_models():
    import torch
    from transformers import pipeline
    device = 'mps' if torch.backends.mps.is_available() else -1
    print(f'Model device: {device}', flush=True)
    nli = pipeline('zero-shot-classification', model=NLI_MODEL, device=device)
    finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert',
                       truncation=True, max_length=256, device=device)
    return nli, finbert


def score_ticker(texts, nli, finbert):
    """Topic shares and AI tone for each filing. All of a ticker's candidate sentences go to
    the models in one call each, because every model call has a fixed start-up cost."""
    filings, all_cands = [], []
    for text in texts['Text']:
        sents = split_sentences(text)
        cands = [s for s in sents if CANDIDATE_RE.search(s)][:MAX_CANDIDATES]
        filings.append((max(len(sents), 1), sum(bool(AI_RE.search(s)) for s in sents), len(all_cands), cands))
        all_cands.extend(cands)

    nli_out = []
    if all_cands:
        nli_out = nli(all_cands, candidate_labels=list(TOPICS.values()),
                      hypothesis_template='{}', multi_label=True, batch_size=16)
        nli_out = [nli_out] if isinstance(nli_out, dict) else nli_out
    is_ai = [bool(AI_RE.search(s)) for s in all_cands]
    ai_sents = [s for s, a in zip(all_cands, is_ai) if a]
    tone = []
    if ai_sents:
        tone = [r['score'] if r['label'] == 'positive' else -r['score'] if r['label'] == 'negative' else 0.0
                for r in finbert(ai_sents, batch_size=32)]
    tone_iter = iter(tone)
    tone_by_cand = [next(tone_iter) if a else None for a in is_ai]

    rows = []
    for (_, f), (n, n_ai, start, cands) in zip(texts.iterrows(), filings):
        counts = {k: 0 for k in TOPICS}
        for r in nli_out[start:start + len(cands)]:
            for label, score in zip(r['labels'], r['scores']):
                if score >= THRESHOLD:
                    counts[LABEL_TO_TOPIC[label]] += 1
        signals = {f'{k}_share': counts[k] / n for k in TOPICS}          # share of all sentences
        signals['ai_demand_net'] = signals['ai_demand_up_share'] - signals['ai_demand_down_share']
        signals['capex_net'] = signals['capex_up_share'] - signals['capex_down_share']
        ai_tones = [t for t in tone_by_cand[start:start + len(cands)] if t is not None]
        signals['ai_tone'] = float(np.mean(ai_tones)) if ai_tones else 0.0
        signals['ai_sentence_ratio'] = n_ai / n
        rows.append({'Ticker': f['Ticker'], 'Date': f['Date'], 'Accepted': f['Accepted'], **signals})
    return pd.DataFrame(rows, columns=['Ticker', 'Date', 'Accepted'] + SIGNAL_COLS), len(all_cands)


# ---------------------------------------------------------------- main
def main():
    from dotenv import load_dotenv
    from edgar import set_identity
    load_dotenv('.env')
    if not os.getenv('EMAIL'):
        sys.exit('Set EMAIL in .env first (the SEC requires an identity).')
    set_identity(os.getenv('EMAIL'))
    os.makedirs(TEXT_CACHE, exist_ok=True)
    os.makedirs(SCORE_CACHE, exist_ok=True)

    tickers = sorted(pd.read_csv(UNIVERSE_PATH)['Symbol'])
    time_only = '--time-only' in sys.argv
    if time_only:
        tickers = [a for a in sys.argv[1:] if not a.startswith('--')] or ['NVDA']
    print(f'{len(tickers)} tickers, {START} to {END}', flush=True)

    nli, finbert = load_models()
    frames, failed, t0 = [], [], time.time()
    t_fetch = t_score = n_filings = n_cands = 0
    for i, ticker in enumerate(tickers, 1):
        path = f'{SCORE_CACHE}/{ticker}.csv'
        try:
            if os.path.exists(path) and not time_only:
                scores = pd.read_csv(path)
            else:
                t1 = time.time()
                texts = load_text(ticker)
                t2 = time.time()
                scores, n = score_ticker(texts, nli, finbert)
                t3 = time.time()
                t_fetch += t2 - t1; t_score += t3 - t2; n_filings += len(texts); n_cands += n
                print(f'  {ticker}: {len(texts)} filings, {n} candidate sentences, '
                      f'text {t2 - t1:.0f}s, scoring {t3 - t2:.0f}s', flush=True)
                if not time_only:
                    scores.to_csv(path, index=False)
            frames.append(scores)
        except Exception as e:
            failed.append(ticker)                              # not cached, so a rerun retries it
            print(f'  {ticker}: FAILED ({e})', flush=True)
        if i % 25 == 0:
            print(f'{i}/{len(tickers)} tickers done ({(time.time() - t0) / 60:.0f} min)', flush=True)

    if time_only:
        if not n_filings:
            sys.exit('No filings found for those tickers.')
        total = 18000                                          # ~16,100 filings 2018-2025, plus 2026
        print(f'\n{n_filings} filings, {n_cands / n_filings:.1f} candidate sentences per filing')
        print(f'Download: {t_fetch / n_filings:.2f}s per filing (0 if the text was already cached)')
        print(f'Scoring:  {t_score / n_filings:.2f}s per filing')
        print(f'Estimate for ~{total:,} filings: download {t_fetch / n_filings * total / 3600:.1f} h, '
              f'scoring {t_score / n_filings * total / 3600:.1f} h. Tickers with more AI language '
              f'(e.g. NVDA) score slower than average.')
        return

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved {len(out)} filings for {out["Ticker"].nunique()} tickers to {OUTPUT_PATH} '
          f'in {(time.time() - t0) / 60:.0f} min. {len(failed)} failed: {failed}', flush=True)


if __name__ == '__main__':
    main()
