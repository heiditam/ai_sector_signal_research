"""
features_v2.py: improved filing features for the NEXT version of the study.

Contents
  1. strip_boilerplate        - remove safe-harbor text and numeric tables
  2. count_terms              - word-boundary keyword counting (fixes substring matches)
  3. extract_domain_signals   - same signals as the notebook, ticker passed explicitly
  4. add_surprise_features    - each filing vs. the firm's own prior filings
  5. add_similarity_to_prior  - Lazy-Prices-style change in language vs. previous filing
  6. finbert_ai_tone          - sentence-level FinBERT tone on AI/GPU/capex sentences
  7. add_days_since_filing    - how stale the text is on each trading day

Usage sketch (in data_cleaning.ipynb, replacing the per-filing signal step):

    from features_v2 import *
    dicts = {'nvidia': NVIDIA_KEYWORDS, 'amd': AMD_KEYWORDS, 'intel': INTEL_KEYWORDS,
             'gpu': GENERAL_GPU_KEYWORDS, 'capex_up': CAPEX_UP_KEYWORDS,
             'capex_down': CAPEX_DOWN_KEYWORDS, 'competitor': COMPETITIVE_RISK_KEYWORDS}
    rows = []
    for f in filings:                                  # dicts with Ticker, Date, Text
        text = strip_boilerplate(f['Text'])
        rows.append({'Ticker': f['Ticker'], 'Date': f['Date'], 'Text': text,
                     **extract_domain_signals(text, f['Ticker'], dicts)})
    ts = pd.DataFrame(rows)
    ts = add_surprise_features(ts, SIGNAL_COLS)
    ts = add_similarity_to_prior(ts)
    ts = ts.join(finbert_ai_tone(ts['Text'], finbert))      # optional, slow
    # ...merge_asof onto the daily panel, then:
    panel = add_days_since_filing(panel, ts)
    # ...and lag ALL filing-based columns one trading day per ticker.
"""

import re
import numpy as np
import pandas as pd

SIGNAL_COLS = ['gpu_mentions', 'capex_up_score', 'capex_down_score', 'capex_net',
               'competitor_mentions', 'ai_sentence_ratio']

AI_PHRASES = ['artificial intelligence', 'ai demand', 'ai revenue', 'ai growth',
              'ai contribution', 'generative ai', 'ai infrastructure', 'ai workloads']

# ---------------------------------------------------------------------------
# 1. Boilerplate removal
# ---------------------------------------------------------------------------
_SAFE_HARBOR = re.compile(
    r'(forward[- ]looking statements|safe harbor|cautionary (note|statement))', re.I)

def strip_boilerplate(text, min_alpha_ratio=0.5):
    """Cut everything from the first safe-harbor heading onward, then drop lines that
    are mostly numbers (financial tables). Returns cleaned text."""
    if not isinstance(text, str):
        return ''
    m = _SAFE_HARBOR.search(text)
    # Only cut if the heading appears after the opening of the release; some filings
    # mention 'forward-looking' in the first paragraph before the real content.
    if m and m.start() > 0.3 * len(text):
        text = text[:m.start()]
    keep = []
    for line in re.split(r'(?<=[.!?])\s+|\n', text):
        letters = sum(c.isalpha() for c in line)
        if letters >= min_alpha_ratio * max(len(line.replace(' ', '')), 1):
            keep.append(line)
    return ' '.join(keep)

# ---------------------------------------------------------------------------
# 2. Word-boundary counting
# ---------------------------------------------------------------------------
def _compile_terms(terms):
    """One regex per dictionary; \\b stops 'h20' matching inside 'h200' etc."""
    terms = sorted(set(t.lower() for t in terms), key=len, reverse=True)
    return re.compile(r'\b(?:' + '|'.join(re.escape(t) for t in terms) + r')\b')

def count_terms(text_lower, pattern):
    return len(pattern.findall(text_lower))

# ---------------------------------------------------------------------------
# 3. Signal extraction with ticker passed explicitly
# ---------------------------------------------------------------------------
_PATTERN_CACHE = {}

def extract_domain_signals(text, ticker, dicts, word_boundary=True):
    """Same outputs as the notebook's extract_domain_signals, per 1,000 words.

    dicts: {'nvidia', 'amd', 'intel', 'gpu', 'capex_up', 'capex_down', 'competitor'}
           mapping to keyword lists.
    word_boundary=False reproduces the original substring counting exactly
    (use that for the current paper; True for the next version).
    """
    text_lower = text.lower()
    words = max(len(text_lower.split()), 1)

    def n(key, terms):
        if not word_boundary:
            return sum(text_lower.count(k) for k in terms)
        cache_key = (key, tuple(terms))
        if cache_key not in _PATTERN_CACHE:
            _PATTERN_CACHE[cache_key] = _compile_terms(terms)
        return count_terms(text_lower, _PATTERN_CACHE[cache_key]) if terms else 0

    competitors = [k for k in dicts['competitor'] if k.lower() != ticker.lower()]
    s = {
        'nvidia_mentions':     n('nvidia', dicts['nvidia']) / words * 1000,
        'amd_mentions':        n('amd', dicts['amd']) / words * 1000,
        'intel_mentions':      n('intel', dicts['intel']) / words * 1000,
        'gpu_mentions':        n('gpu', dicts['gpu']) / words * 1000,
        'capex_up_score':      n('capex_up', dicts['capex_up']) / words * 1000,
        'capex_down_score':    n('capex_down', dicts['capex_down']) / words * 1000,
        'competitor_mentions': n(f'competitor_{ticker}', competitors) / words * 1000,
    }
    s['capex_net'] = s['capex_up_score'] - s['capex_down_score']
    sentences = [x for x in re.split(r'(?<=[.!?])\s+', text) if x.strip()]
    ai = [x for x in sentences if any(p in x.lower() for p in AI_PHRASES)]
    s['ai_sentence_ratio'] = len(ai) / max(len(sentences), 1)
    return s

# ---------------------------------------------------------------------------
# 4. Surprise features: each filing vs. the firm's own history
# ---------------------------------------------------------------------------
def add_surprise_features(ts, cols=SIGNAL_COLS, min_prior=2):
    """s_<col> = value minus mean of the firm's PRIOR filings (no lookahead).
    Filings with fewer than `min_prior` previous filings get 0."""
    ts = ts.sort_values(['Ticker', 'Date']).copy()
    g = ts.groupby('Ticker')
    n_prior = g.cumcount()
    for c in cols:
        prior_mean = g[c].transform(lambda x: x.shift(1).expanding().mean())
        ts[f's_{c}'] = np.where(n_prior >= min_prior, ts[c] - prior_mean, 0.0)
    return ts

# ---------------------------------------------------------------------------
# 5. Lazy-Prices-style similarity to the previous filing
# ---------------------------------------------------------------------------
def add_similarity_to_prior(ts, text_col='Text'):
    """Cosine similarity between each filing and the same firm's previous filing.
    The vectorizer is fit on just that pair, so no later filing influences the value
    (no lookahead). Low similarity = the firm changed how it talks."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ts = ts.sort_values(['Ticker', 'Date']).copy()
    ts['sim_to_prior'] = np.nan
    for tk, grp in ts.groupby('Ticker'):
        texts = grp[text_col].fillna('').tolist()
        sims = [np.nan]
        for i in range(1, len(texts)):
            try:
                X = CountVectorizer(stop_words='english').fit_transform([texts[i - 1], texts[i]])
                sims.append(float(cosine_similarity(X[0], X[1])[0, 0]))
            except ValueError:            # empty vocabulary (blank text)
                sims.append(np.nan)
        ts.loc[grp.index, 'sim_to_prior'] = sims
    ts['change_vs_prior'] = 1 - ts['sim_to_prior']   # higher = bigger language change
    return ts

# ---------------------------------------------------------------------------
# 6. Sentence-level FinBERT tone on AI-related sentences
# ---------------------------------------------------------------------------
TONE_TERMS = AI_PHRASES + ['gpu', 'accelerator', 'data center', 'capex',
                           'capital expenditure', 'inference', 'training']

def finbert_ai_tone(texts, finbert, batch_size=32, max_sentences=40):
    """For each filing, score sentences mentioning AI/GPU/capex with FinBERT.
    Returns a DataFrame: ai_tone (mean pos-neg score), ai_pos_share, ai_neg_share,
    n_ai_sentences. `finbert` is the transformers pipeline from the notebook."""
    out = []
    for text in texts:
        sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+', text or '')
                 if any(t in x.lower() for t in TONE_TERMS) and len(x.split()) >= 5]
        sents = sents[:max_sentences]
        if not sents:
            out.append({'ai_tone': 0.0, 'ai_pos_share': 0.0, 'ai_neg_share': 0.0, 'n_ai_sentences': 0})
            continue
        res = finbert(sents, batch_size=batch_size, truncation=True)
        sign = [1 if r['label'] == 'positive' else -1 if r['label'] == 'negative' else 0 for r in res]
        out.append({
            'ai_tone': float(np.mean([s * r['score'] for s, r in zip(sign, res)])),
            'ai_pos_share': float(np.mean([s == 1 for s in sign])),
            'ai_neg_share': float(np.mean([s == -1 for s in sign])),
            'n_ai_sentences': len(sents),
        })
    return pd.DataFrame(out, index=getattr(texts, 'index', None))

# ---------------------------------------------------------------------------
# 7. Staleness of the text on each trading day
# ---------------------------------------------------------------------------
def add_days_since_filing(panel, ts):
    """panel: daily DataFrame with MultiIndex (Date, Ticker).
    Adds trading days since the most recent filing (NaN before the first filing).
    Apply the same one-day lag as the other filing features afterwards."""
    p = panel.reset_index()[['Date', 'Ticker']].sort_values('Date')
    f = ts[['Date', 'Ticker']].rename(columns={'Date': 'filing_date'}).sort_values('filing_date')
    m = pd.merge_asof(p, f, left_on='Date', right_on='filing_date', by='Ticker', direction='backward')
    m = m.sort_values(['Ticker', 'Date'])
    m['tday'] = m.groupby('Ticker').cumcount()
    first = m.groupby(['Ticker', 'filing_date'])['tday'].transform('min')
    m['days_since_filing'] = np.where(m['filing_date'].notna(), m['tday'] - first, np.nan)
    out = panel.copy()
    out['days_since_filing'] = m.set_index(['Date', 'Ticker'])['days_since_filing'].reindex(out.index)
    return out
