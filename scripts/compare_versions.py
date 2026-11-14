"""
compare_versions.py: v1 vs v2 feature data under the identical full-model pipeline.

Each dataset gets the same target rebuild, walk-forward splits, LightGBM settings,
and volatility-only ablation as scripts/full_model.py, so any difference in IC
comes from the feature data alone.

Usage (from the repository root):
    python scripts/compare_versions.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_model import (FULL_MODEL_FEATS, VOL_FEATS, load_full_model_data,
                        make_splits, fit_full_model, daily_ic)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONS = {
    'v1': os.path.join(ROOT, 'data', 'ai_sector_features_df.csv'),
    'v2': os.path.join(ROOT, 'data', 'ai_sector_features_news_v2.csv'),
}


def newey_west(s, lags=20):
    """Newey-West (HAC) t-test that the mean of a daily series is zero."""
    s = pd.Series(s).dropna()
    return sm.OLS(s.values, np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def evaluate(path, feat_cols=FULL_MODEL_FEATS):
    """Fold ICs plus daily IC series for the full model and the vol-only model."""
    splits = make_splits(load_full_model_data(path))
    fold_ic, daily_full, daily_vol = [], [], []
    for train_df, test_df in splits:
        model = fit_full_model(train_df, feat_cols)
        t = test_df.assign(prob=model.predict_proba(test_df[feat_cols])[:, 1])
        fold_ic.append(np.corrcoef(t['prob'], t['Target'])[0, 1])
        daily_full.append(daily_ic(t))

        vol_model = fit_full_model(train_df, VOL_FEATS)
        tv = test_df.assign(prob=vol_model.predict_proba(test_df[VOL_FEATS])[:, 1])
        daily_vol.append(daily_ic(tv))
    return np.array(fold_ic), pd.concat(daily_full).dropna(), pd.concat(daily_vol).dropna()


def compare(versions=VERSIONS, feat_cols=FULL_MODEL_FEATS):
    """Summary table (one row per version) and the v2 - v1 daily IC difference."""
    rows, daily = [], {}
    for name, path in versions.items():
        folds, full, vol = evaluate(path, feat_cols)
        delta = (full - vol).dropna()
        nw_full, nw_delta = newey_west(full), newey_west(delta)
        daily[name] = full
        rows.append({
            'version': name,
            **{f'fold{i}_IC': ic for i, ic in enumerate(folds, 1)},
            'mean_fold_IC': folds.mean(),
            'fold_ICIR': folds.mean() / folds.std(ddof=1),
            'mean_daily_IC': full.mean(),
            'p_IC_gt_0': nw_full.pvalues[0],
            'vol_only_daily_IC': vol.mean(),
            'text_delta_IC': delta.mean(),
            'p_text_delta': nw_delta.pvalues[0],
        })
    summary = pd.DataFrame(rows).set_index('version')
    names = list(versions)
    diff = (daily[names[1]] - daily[names[0]]).dropna() if len(names) == 2 else None
    return summary, diff


if __name__ == "__main__":
    warnings.filterwarnings("ignore")   # LightGBM deprecation notices; NaN IC on days with constant predictions
    summary, diff = compare()
    print(summary.T.round(4).to_string())
    if diff is not None:
        nw = newey_west(diff)
        print(f"\nv2 - v1 mean daily IC: {diff.mean():.4f}  "
              f"(Newey-West t = {nw.tvalues[0]:.2f}, p = {nw.pvalues[0]:.3f}, {len(diff)} days)")
