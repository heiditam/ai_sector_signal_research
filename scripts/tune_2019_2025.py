"""Phase 1: model selection on 2019-2025 only. 2026 is never loaded here."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')
root = sys.argv[1] if len(sys.argv) > 1 else "."

def build_features(df, ts):
    E = ['gpu_mentions','capex_up_score','capex_down_score','capex_net','competitor_mentions','ai_sentence_ratio']
    V = ['vol_5d','vol_20d','vol_spike']
    ts = ts.sort_values(['Ticker','Date']).copy()
    for c in E:
        ts[f's_{c}'] = (ts[c] - ts.groupby('Ticker')[c].transform(lambda x: x.shift(1).expanding().mean())).fillna(0)
    S = [f's_{c}' for c in E]
    d = df.reset_index().sort_values('Date')
    d = pd.merge_asof(d, ts[['Date','Ticker']+S].sort_values('Date'), on='Date', by='Ticker', direction='backward')
    d = d.set_index(['Date','Ticker']).sort_index(); d[S] = d[S].fillna(0)
    d[E+S] = d.groupby(level='Ticker')[E+S].shift(1).fillna(0)              # 1-day filing lag
    for c in V+E+S: d[f'r_{c}'] = d.groupby(level='Date')[c].rank(pct=True)  # cross-sectional ranks
    sets = {'raw_vol+edgar': V+E,
            'rank_vol+edgar': [f'r_{c}' for c in V+E],
            'rank_vol+surprise': [f'r_{c}' for c in V+S],
            'rank_vol+edgar+surprise': [f'r_{c}' for c in V+E+S]}
    return d, sets

def fit_predict(kind, fit, val, test, feats):
    X, y = fit[feats].fillna(0), fit['Target']
    if kind == 'lgbm_default':
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                               colsample_bytree=0.8, class_weight='balanced', reg_alpha=0.1, reg_lambda=0.1, verbose=-1)
        m.fit(X, y, eval_set=[(val[feats].fillna(0), val['Target'])], callbacks=[lgb.early_stopping(50, verbose=False)])
    elif kind == 'lgbm_shallow':   # heavily regularized, fixed size: no early-stopping collapse
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.01, num_leaves=7, min_child_samples=200, subsample=0.7,
                               subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
        m.fit(X, y)
    else:                          # logistic regression
        m = LogisticRegression(C=0.1, max_iter=2000); m.fit(X, y)
    return m.predict_proba(test[feats].fillna(0))[:, 1]

def daily_ic(test, p):
    t = pd.DataFrame({'p': p, 'y': test['Target'].values}, index=test.index)
    return t.groupby(level='Date').apply(lambda g: np.corrcoef(g.p, g.y)[0,1] if g.y.nunique()>1 and g.p.nunique()>1 else np.nan).dropna()

def split_fit_val(train_df):
    td = train_df.index.get_level_values('Date').unique().sort_values()
    cut = int(len(td)*0.85)
    return (train_df[train_df.index.get_level_values('Date') <= td[cut-5]],
            train_df[train_df.index.get_level_values('Date') > td[cut]])

if __name__ == '__main__':
    df = pd.read_csv(f'{root}/data/ai_sector_features_df.csv', parse_dates=['Date']).set_index(['Date','Ticker']).sort_index()
    df = df[df.index.get_level_values('Date') <= '2025-12-31']
    ts = pd.read_csv(f'{root}/data/transcript_signals_df.csv', parse_dates=['Date'])
    ts = ts[ts.Date <= '2025-12-31']
    d, sets = build_features(df, ts)
    dates = d.index.get_level_values('Date').unique().sort_values()
    splits = [(d[d.index.get_level_values('Date').isin(dates[a])], d[d.index.get_level_values('Date').isin(dates[b])])
              for a, b in TimeSeriesSplit(n_splits=5, gap=21).split(dates)]
    rows = []
    for fs, feats in sets.items():
        for kind in ['lgbm_default', 'lgbm_shallow', 'logit']:
            fold_ic, dl = [], []
            for tr, te in splits:
                fit, val = split_fit_val(tr); p = fit_predict(kind, fit, val, te, feats)
                fold_ic.append(np.corrcoef(p, te['Target'])[0,1]); dl.append(daily_ic(te, p))
            dl = pd.concat(dl); base = [np.corrcoef(te['vol_5d'].fillna(0), te['Target'])[0,1] for _, te in splits]
            rows.append({'features': fs, 'model': kind, 'mean_daily_IC': dl.mean(), 'naive_t': dl.mean()/dl.std()*np.sqrt(len(dl)),
                         'mean_fold_IC': np.mean(fold_ic), 'folds_>_base': int(sum(f > b for f, b in zip(fold_ic, base))),
                         'fold_ICs': [round(x, 3) for x in fold_ic]})
    res = pd.DataFrame(rows).sort_values('mean_daily_IC', ascending=False)
    pd.set_option('display.width', 250); print(res.round(4).to_string(index=False))
    best = res.iloc[0]; json.dump({"features": best.features, "model": best.model}, open(sys.argv[2] if len(sys.argv) > 2 else "outputs/frozen_config.json", "w"))
    print(f"\nFROZEN CHOICE: {best.features} + {best.model}")
