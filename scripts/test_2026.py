"""
Phase 2: ONE-TIME out-of-sample test on 2026.

Frozen on 2026-10-14 from tune_2019_2025.py (best of 12 configs on 2019-2025 folds):
    features = raw vol_5d, vol_20d, vol_spike + 6 EDGAR features (1-day lag)
    model    = regularized LightGBM (300 trees, lr 0.01, 7 leaves, min_child_samples 200)
Pre-specified comparisons (decided before seeing 2026):
    1. frozen model vs. raw vol_5d baseline
    2. frozen model vs. the same model with volatility features only (text contribution)
Run ONCE after rebuilding data through 2026. Do not change anything and rerun.

Usage (from repo root):  python scripts/test_2026.py data/ai_sector_features_2026.csv data/transcript_signals_df.csv
"""
import sys, os, numpy as np, pandas as pd, statsmodels.api as sm, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tune_2019_2025 import build_features, fit_predict, daily_ic

feat_path = sys.argv[1] if len(sys.argv) > 1 else 'data/ai_sector_features_2026.csv'
ts_path   = sys.argv[2] if len(sys.argv) > 2 else 'data/transcript_signals_df.csv'
TRAIN_END, TEST_START = os.getenv('TRAIN_END', '2025-12-31'), os.getenv('TEST_START', '2026-02-02')   # >= 21 trading days of gap

df = pd.read_csv(feat_path, parse_dates=['Date']).set_index(['Date', 'Ticker']).sort_index()
df = df.dropna(subset=['fwd_5d', 'Target'])
ts = pd.read_csv(ts_path, parse_dates=['Date'])
d, sets = build_features(df, ts)

dates = d.index.get_level_values('Date')
train, test = d[dates <= TRAIN_END], d[dates >= TEST_START]
assert test.index.get_level_values('Date').min() >= pd.Timestamp(TEST_START)
print(f"train: {train.index.get_level_values('Date').min().date()} -> {train.index.get_level_values('Date').max().date()} ({len(train)} rows)")
print(f"test : {test.index.get_level_values('Date').min().date()} -> {test.index.get_level_values('Date').max().date()} ({len(test)} rows)\n")

V = ['vol_5d', 'vol_20d', 'vol_spike']
full = sets['raw_vol+edgar']
p_full = fit_predict('lgbm_shallow', train, None, test, full)
p_vol  = fit_predict('lgbm_shallow', train, None, test, V)

def hac(s):
    r = sm.OLS(s.values, np.ones(len(s))).fit(cov_type='HAC', cov_kwds={'maxlags': 20})
    return r.params[0], r.tvalues[0], r.pvalues[0]

ic_full, ic_vol = daily_ic(test, p_full), daily_ic(test, p_vol)
ic_base = daily_ic(test, test['vol_5d'].fillna(0).values)
for name, s in [('frozen model (vol + EDGAR)', ic_full), ('volatility only, same model', ic_vol),
                ('vol_5d baseline', ic_base), ('model - baseline', (ic_full - ic_base).dropna()),
                ('text contribution', (ic_full - ic_vol).dropna())]:
    m, t, p = hac(s)
    print(f"{name:28s} mean daily IC={m:+.4f}  Newey-West t={t:+.2f}  p={p:.3f}  (n={len(s)} days)")
