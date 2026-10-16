"""
Full model: the main model of the study.

    features = raw vol_5d, vol_20d, vol_spike + 6 EDGAR features + nvidia/amd/intel mentions
    target   = fwd_5d rebuilt as the sum of the NEXT 5 daily returns, Target = above that date's median
    model    = LightGBM from train_evaluate (training notebook), early stopping on a purged validation split

Walk-forward results (5 folds, 2019-2025):
    mean fold IC = 0.0185, fold ICIR = 0.6671, mean daily IC = 0.0149
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

FULL_MODEL_FEATS = [
    'vol_5d', 'vol_20d', 'vol_spike',
    'gpu_mentions', 'capex_up_score', 'capex_down_score', 'capex_net',
    'competitor_mentions', 'ai_sentence_ratio',
    'nvidia_mentions', 'amd_mentions', 'intel_mentions'
]

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
