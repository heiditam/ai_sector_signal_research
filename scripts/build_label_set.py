"""
Build, then evaluate, the hand-labeled sentence set for the AI-disclosure dataset.

Step 1, sample sentences to label (minutes, no models):
    python3 scripts/build_label_set.py sample
  writes  outputs/v3/labels/to_label.csv          you fill in the six topic columns (0/1)
          outputs/v3/labels/agreement_subset.csv  same 200 sentences, for a second labeler
          outputs/v3/labels/sample_key.csv        how each sentence was chosen (do not open
                                                  while labeling)

Step 2, after labeling, score the labels against the NLI model (~10 minutes):
    python3 scripts/build_label_set.py evaluate
  reads   outputs/v3/labels/to_label.csv (filled in) and, if present,
          outputs/v3/labels/agreement_subset_labeler2.csv
  writes  outputs/v3/labels/nli_scores.csv and prints agreement, precision, recall and F1

Run from the repository root. Uses the raw-text cache that score_filings.py already built,
and the same cleaning, sentence splitting and candidate filter, so the labels match exactly
what the model sees.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from score_filings import (TEXT_CACHE, TOPICS, THRESHOLD, NLI_MODEL, CANDIDATE_RE,      # noqa: E402
                           CAPEX_DOWN_KEYWORDS, COMPETITIVE_RISK_KEYWORDS,
                           clean_filing_text, split_sentences)

LABEL_DIR = 'outputs/v3/labels'
UNIVERSE_PATH = 'outputs/v3/data/sp500_tickers.csv'
DEV_END = '2026-01-01'        # label development-period filings only
SEED = 2026

N_RANDOM = 700                # candidate sentences, spread across sectors and years
N_RARE = 200                  # candidates that mention rarer topics, so each topic has positives
N_NONCAND = 100               # sentences the candidate filter skips, to measure what it misses
N_AGREE = 200                 # subset for the second labeler

RARE_RE = '|'.join(CAPEX_DOWN_KEYWORDS + COMPETITIVE_RISK_KEYWORDS
                   + ['supply constrain', 'capacity constrain', 'shortage', 'allocation',
                      'slowing demand', 'weaker demand', 'softer demand', 'reduce capital'])


def all_sentences():
    """Every sentence of every development-period filing, with its metadata."""
    sectors = pd.read_csv(UNIVERSE_PATH).set_index('Symbol')['GICS Sector']
    rows = []
    for path in sorted(glob.glob(f'{TEXT_CACHE}/*.parquet')):
        texts = pd.read_parquet(path)
        texts = texts[pd.to_datetime(texts['Date']) < DEV_END]
        for _, f in texts.iterrows():
            sents = split_sentences(clean_filing_text(f['Text']))
            for i, s in enumerate(sents):
                rows.append({'Ticker': f['Ticker'], 'Date': pd.to_datetime(f['Date']).date(),
                             'sentence': s,
                             'before': sents[i - 1] if i > 0 else '',
                             'after': sents[i + 1] if i + 1 < len(sents) else '',
                             'candidate': bool(CANDIDATE_RE.search(s))})
    df = pd.DataFrame(rows).drop_duplicates(subset='sentence')        # boilerplate repeats a lot
    df['sector'] = df['Ticker'].map(sectors).fillna('Unknown')
    df['year'] = pd.to_datetime(df['Date']).dt.year
    return df


def stratified(df, n, rng):
    """About n rows, spread evenly over sector x year cells (small cells give what they have)."""
    cells = df.groupby(['sector', 'year'])
    per_cell = max(1, n // max(cells.ngroups, 1))
    picked = cells.apply(lambda g: g.sample(min(len(g), per_cell), random_state=rng.integers(1e9)))
    picked = picked.reset_index(drop=True)
    if len(picked) < n:                                               # top up from what is left
        rest = df[~df['sentence'].isin(picked['sentence'])]
        picked = pd.concat([picked, rest.sample(min(len(rest), n - len(picked)),
                                                random_state=rng.integers(1e9))])
    return picked.head(n)


def sample():
    rng = np.random.default_rng(SEED)
    os.makedirs(LABEL_DIR, exist_ok=True)
    df = all_sentences()
    cand, noncand = df[df['candidate']], df[~df['candidate']]
    print(f'{len(df):,} unique sentences, {len(cand):,} candidates')

    rare_pool = cand[cand['sentence'].str.contains(RARE_RE, case=False, regex=True)]
    rare = rare_pool.sample(min(len(rare_pool), N_RARE), random_state=SEED)
    rest = cand[~cand['sentence'].isin(rare['sentence'])]
    parts = [stratified(rest, N_RANDOM, rng).assign(stratum='random_candidate'),
             rare.assign(stratum='rare_topic_candidate'),
             noncand.sample(min(len(noncand), N_NONCAND), random_state=SEED).assign(stratum='non_candidate')]
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle
    out.insert(0, 'id', [f's{i:04d}' for i in range(len(out))])
    out['agreement'] = False
    out.loc[out.sample(min(N_AGREE, len(out)), random_state=SEED).index, 'agreement'] = True

    blank = {k: '' for k in TOPICS}
    sheet = out[['id', 'Ticker', 'Date', 'before', 'sentence', 'after']].assign(**blank, notes='')
    sheet.to_csv(f'{LABEL_DIR}/to_label.csv', index=False)
    sheet[out['agreement']].to_csv(f'{LABEL_DIR}/agreement_subset.csv', index=False)
    out[['id', 'stratum', 'sector', 'year', 'candidate', 'agreement']].to_csv(
        f'{LABEL_DIR}/sample_key.csv', index=False)
    print(out['stratum'].value_counts().to_string())
    print(f'Wrote {len(sheet)} sentences to {LABEL_DIR}/to_label.csv '
          f'({out["agreement"].sum()} also in agreement_subset.csv)')
    
    # practice set for the warm-up: separate from the labeled sentences, never scored
    unused = cand[~cand['sentence'].isin(out['sentence'])]
    practice = unused.sample(min(15, len(unused)), random_state=SEED + 1)
    practice.insert(0, 'id', [f'p{i:02d}' for i in range(len(practice))])
    practice[['id', 'Ticker', 'Date', 'before', 'sentence', 'after']].assign(**blank, notes='') \
        .to_csv(f'{LABEL_DIR}/practice.csv', index=False)
    print(f'Wrote {len(practice)} practice sentences to {LABEL_DIR}/practice.csv')


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return np.nan if pe == 1 else (po - pe) / (1 - pe)


def prf(y, pred):
    tp = ((pred == 1) & (y == 1)).sum()
    p = tp / max(pred.sum(), 1)
    r = tp / max(y.sum(), 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate():
    labels = pd.read_csv(f'{LABEL_DIR}/to_label.csv')
    missing = labels[list(TOPICS)].isna().any(axis=1).sum()
    if missing:
        sys.exit(f'{missing} sentences still have blank topic columns; finish labeling first.')
    key = pd.read_csv(f'{LABEL_DIR}/sample_key.csv')
    labels = labels.merge(key, on='id')

    second = f'{LABEL_DIR}/agreement_subset_labeler2.csv'
    if os.path.exists(second):
        l2 = pd.read_csv(second).set_index('id')
        both = labels.set_index('id').loc[l2.index]
        print('\nInter-labeler agreement (Cohen kappa) on', len(l2), 'sentences')
        for k in TOPICS:
            print(f'  {k:16s} kappa = {kappa(both[k], l2[k]):.2f}')

    scores_path = f'{LABEL_DIR}/nli_scores.csv'
    if os.path.exists(scores_path):
        scores = pd.read_csv(scores_path).set_index('id')
    else:
        import torch
        from transformers import pipeline
        device = 'mps' if torch.backends.mps.is_available() else -1
        nli = pipeline('zero-shot-classification', model=NLI_MODEL, device=device)
        out = nli(labels['sentence'].tolist(), candidate_labels=list(TOPICS.values()),
                  hypothesis_template='{}', multi_label=True, batch_size=16)
        label_to_topic = {v: k for k, v in TOPICS.items()}
        scores = pd.DataFrame([{label_to_topic[l]: s for l, s in zip(r['labels'], r['scores'])}
                               for r in out], index=labels['id'])
        scores.index.name = 'id'
        scores.to_csv(scores_path)

    # Tune the threshold on one half of the labels, report on the other half
    rng = np.random.default_rng(SEED)
    tune = rng.random(len(labels)) < 0.5
    grid = np.round(np.arange(0.5, 0.96, 0.05), 2)
    print(f'\nNLI vs. your labels (current THRESHOLD = {THRESHOLD}); '
          f'best threshold chosen on {tune.sum()} sentences, reported on {(~tune).sum()}')
    print(f'  {"topic":16s} {"positives":>9s} {"P@cur":>6s} {"R@cur":>6s} {"F1@cur":>6s} '
          f'{"best_t":>6s} {"F1@best":>7s}')
    s = scores.loc[labels['id']].to_numpy()
    for j, k in enumerate(TOPICS):
        y = labels[k].astype(int).to_numpy()
        p, r, f1 = prf(y[~tune], (s[~tune, j] >= THRESHOLD).astype(int))
        best = max(grid, key=lambda t: prf(y[tune], (s[tune, j] >= t).astype(int))[2])
        f1b = prf(y[~tune], (s[~tune, j] >= best).astype(int))[2]
        print(f'  {k:16s} {y.sum():9d} {p:6.2f} {r:6.2f} {f1:6.2f} {best:6.2f} {f1b:7.2f}')

    nc = labels[labels['stratum'] == 'non_candidate']
    if len(nc):
        hit = nc[list(TOPICS)].astype(int).any(axis=1).mean()
        print(f'\nCandidate filter: {hit:.0%} of {len(nc)} skipped sentences had a topic '
              f'(what the filter misses)')


if __name__ == '__main__':
    {'sample': sample, 'evaluate': evaluate}.get(
        sys.argv[1] if len(sys.argv) > 1 else '', lambda: sys.exit(__doc__))()
