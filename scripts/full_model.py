"""
Full model: the main model of the study.

    features = vol_5d, vol_20d, vol_spike + 6 EDGAR keyword features + nvidia/amd/intel mentions
    target   = fwd_5d = sum of the NEXT 5 daily returns (t+1 ... t+5); Target = 1 if above that date's median
    model    = LightGBM, early stopping on the last 15% of training dates (5-day purge),
               5 expanding walk-forward folds on unique dates with a 21-trading-day gap

Data assumption: EDGAR features in data/ai_sector_features_df.csv are already lagged one
trading day (after-close filings are not usable on their filing date).

Walk-forward results (5 folds, 2019-2025, 1,455 test days):
    mean fold IC   = 0.0185, fold ICIR = 0.67 (5 folds; descriptive only)
    mean daily IC  = 0.0149, daily ICIR = 0.06, not significantly different from zero
                     (Newey-West t = 1.34, p = 0.18)

    vs. raw vol_5d baseline (corr of vol_5d with Target in each test fold):
        mean baseline IC = 0.0016; model IC is higher in 4 of 5 folds (mean delta IC = 0.0169)
    vs. volatility-only LightGBM (vol_5d, vol_20d, vol_spike; same folds):
        text features add delta IC = 0.0082 (Newey-West p = 0.57) -> no significant incremental value

    Early stopping keeps 1 tree in every fold, consistent with little learnable structure.

WARNING: do not change the target construction. An earlier version used
x.shift(-1).rolling(5).sum(), which spans t-3 ... t+1 and leaks past returns into the label;
it produced a spurious IC of 0.061. See the leakage analysis in notebooks/cleaned/training.ipynb.

Usage (from the repository root):
    python scripts/full_model.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

FULL_MODEL_FEATS = [
    'vol_5d', 'vol_20d', 'vol_spike',
    'gpu_mentions', 'capex_up_score', 'capex_down_score', 'capex_net',
    'competitor_mentions', 'ai_sentence_ratio',
    'nvidia_mentions', 'amd_mentions', 'intel_mentions'
]
VOL_FEATS = ['vol_5d', 'vol_20d', 'vol_spike']

PURGE_DAYS = 5


def load_full_model_data(path):
    """Load the daily panel and rebuild fwd_5d / Target so fwd_5d is exactly ret(t+1)+...+ret(t+5)."""
    df = pd.read_csv(path, parse_dates=['Date']).set_index(['Date', 'Ticker']).sort_index()
    g = df.groupby(level='Ticker')['ret_1d']
    df['fwd_5d'] = sum(g.shift(-k) for k in range(1, 6))
    df = df.dropna(subset=['fwd_5d']).copy()
    df['Target'] = (df['fwd_5d'] > df.groupby(level='Date')['fwd_5d'].transform('median')).astype(int)
    return df


def make_splits(df, n_splits=5, gap_days=21):
    """Walk-forward splits on unique dates (same as the training notebook)."""
    dates = df.index.get_level_values('Date').unique().sort_values()
    return [(df[df.index.get_level_values('Date').isin(dates[tr])],
             df[df.index.get_level_values('Date').isin(dates[te])])
            for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=gap_days).split(dates)]


def fit_full_model(train_df, feat_cols=FULL_MODEL_FEATS):
    """Fit one fold exactly as train_evaluate does: last 15% of train dates for validation, purge gap."""
    train_dates = train_df.index.get_level_values('Date').unique().sort_values()
    cut_idx = int(len(train_dates) * 0.85)
    fit_df = train_df[train_df.index.get_level_values('Date') <= train_dates[cut_idx - PURGE_DAYS]]
    val_df = train_df[train_df.index.get_level_values('Date') > train_dates[cut_idx]]
    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
        reg_alpha=0.1, reg_lambda=0.1, verbose=-1
    )
    model.fit(fit_df[feat_cols], fit_df['Target'],
              eval_set=[(val_df[feat_cols], val_df['Target'])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def daily_ic(test_df):
    """Cross-sectional IC for each test date (correlation of predicted prob with Target)."""
    return test_df.groupby(level='Date').apply(lambda g: g['prob'].corr(g['Target']))


if __name__ == "__main__":
    import warnings
    import statsmodels.api as sm
    warnings.filterwarnings("ignore")   # LightGBM deprecation notices; NaN IC on days with constant predictions

    def newey_west(s, lags=20):
        """Newey-West (HAC) t-test that the mean of a daily series is zero."""
        s = pd.Series(s).dropna()
        return sm.OLS(s.values, np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    df = load_full_model_data("data/ai_sector_features_df.csv")
    splits = make_splits(df)

    # ---- Full model vs. raw vol_5d baseline, fold by fold ----
    rows, daily_full, daily_vol = [], [], []
    for i, (train_df, test_df) in enumerate(splits, 1):
        model = fit_full_model(train_df)
        t = test_df.assign(prob=model.predict_proba(test_df[FULL_MODEL_FEATS])[:, 1])
        ic = np.corrcoef(t['prob'], t['Target'])[0, 1]
        base_ic = np.corrcoef(t['vol_5d'].fillna(0), t['Target'])[0, 1]   # same baseline as the notebook
        daily_full.append(daily_ic(t))

        vol_model = fit_full_model(train_df, VOL_FEATS)                    # for the text ablation
        tv = test_df.assign(prob=vol_model.predict_proba(test_df[VOL_FEATS])[:, 1])
        daily_vol.append(daily_ic(tv))

        dates = t.index.get_level_values('Date')
        rows.append({'fold': i, 'test_start': dates.min().date(), 'test_end': dates.max().date(),
                     'model_IC': ic, 'baseline_IC': base_ic, 'delta_IC': ic - base_ic,
                     'trees': model.best_iteration_})

    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))

    n_beat = int((res['model_IC'] > res['baseline_IC']).sum())
    print(f"\nMean fold IC:            {res['model_IC'].mean():.4f}")
    print(f"Fold ICIR:               {res['model_IC'].mean() / res['model_IC'].std():.4f}")
    print(f"Mean baseline IC:        {res['baseline_IC'].mean():.4f}  (raw vol_5d)")
    print(f"Mean delta IC vs base:   {res['delta_IC'].mean():.4f}")
    print(f"Folds beating baseline:  {n_beat}/{len(res)}")

    # ---- Significance (daily IC) ----
    daily_full = pd.concat(daily_full).dropna()
    daily_vol = pd.concat(daily_vol).dropna()
    nw_full = newey_west(daily_full)
    print(f"\nMean daily IC:           {daily_full.mean():.4f}  ({len(daily_full)} days)")
    print(f"Daily ICIR:              {daily_full.mean() / daily_full.std():.4f}")
    print(f"Newey-West (IC > 0):     t = {nw_full.tvalues[0]:.2f}, p = {nw_full.pvalues[0]:.3f}")

    # ---- RQ1: does text add value over a volatility-only model? ----
    delta = (daily_full - daily_vol).dropna()
    nw_delta = newey_west(delta)
    print(f"\nVol-only model daily IC: {daily_vol.mean():.4f}")
    print(f"Text delta IC vs vol-only: {delta.mean():.4f}  "
          f"(Newey-West t = {nw_delta.tvalues[0]:.2f}, p = {nw_delta.pvalues[0]:.3f}, {len(delta)} days)")