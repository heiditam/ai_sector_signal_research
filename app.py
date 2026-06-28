# imports
import streamlit as st
import lightgbm as lgb
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import date, timedelta

st.cache_data.clear()

# Next 5 Days Graph
def make_splits(df, n_splits=5, target_col='target', gap_days=21):
    """
    Create walk-forward splits from the merged DataFrame.
    gap_days: trading days between train end and test start (avoids lookahead).
    """
    # Work on unique dates to split time, not rows
    dates = df.index.get_level_values('Date').unique().sort_values()
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap_days)
    
    splits = []
    for train_idx, test_idx in tss.split(dates):
        train_dates = dates[train_idx]
        test_dates  = dates[test_idx]
        
        train_df = df[df.index.get_level_values('Date').isin(train_dates)]
        test_df  = df[df.index.get_level_values('Date').isin(test_dates)]
        
        splits.append((train_df, test_df))
    
    return splits

ai_sector_features_signals_df = pd.read_csv('data/ai_sector_features_df.csv', index_col=['Date', 'Ticker'], parse_dates=['Date'])

splits = make_splits(ai_sector_features_signals_df)

def train_evaluate(splits):
    all_results = []
    for fold_num, (train_df, test_df) in enumerate(splits):
        feat_cols = [
            'vol_5d', 'vol_20d', 'vol_spike', 'news_sentiment', 'pct_positive', 'news_count','gpu_mentions', 'capex_up_score', 
            'capex_down_score','capex_net', 'competitor_mentions', 'ai_sentence_ratio'
        ]
        X_tr, y_tr = train_df[feat_cols], train_df['Target']
        X_te, y_te = test_df[feat_cols],  test_df['Target']
        asset_returns = test_df['ret_5d'].values

        model = lgb.LGBMClassifier(
            n_estimators=600, 
            learning_rate=0.05,
            num_leaves=63,    
            min_child_samples=20,
            subsample=0.8,    
            colsample_bytree=0.8,
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        probs = model.predict_proba(X_te)[:,1]
        ic  = np.corrcoef(probs, y_te)[0,1]
        acc = accuracy_score(y_te, probs > 0.5)

        # volatile threshold
        signals = np.where(probs > 0.52, 1, np.where(probs < 0.48, -1, 0))
        strat_returns = signals * asset_returns
        excess_returns = strat_returns - (0.04 / 52) # assume 4% annual risk-free rate; 52 trading periods

        if excess_returns.std() == 0:
            sharpe = 0
        else:
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(52)

        # correct big moves
        correct_big_moves = np.mean(signals[(np.abs(asset_returns) > 0.02)] == 
                                    np.sign(asset_returns[(np.abs(asset_returns) > 0.02)]))

        all_results.append({'fold': fold_num+1, 'IC': round(ic,4), 'accuracy': round(acc,4), 'sharpe': round(sharpe, 4),\
                            'Accuracy on >2% moves': round(correct_big_moves, 3)})
        X_te = X_te.fillna(0)
        
    return pd.DataFrame(all_results), model, X_te, y_te

results_df, final_model, X_te, y_te = train_evaluate(splits)

tickers = pd.read_csv('data/tickers.csv')
tickers = tickers.drop(columns=['Unnamed: 0'])

ai_tickers = {
    'semis':   ['NVDA','AMD','INTC','ASML','TSM','AVGO','MRVL','ARM','SMCI'], # semiconductors
    'hypers':  ['MSFT','GOOGL','AMZN','META'], # hyperscalar
    'pure':    ['PLTR','AI','SOUN'], # AI pure plays
    'infra':   ['DELL','ANET','HPE'] # infrastructure
}
all_tickers = [t for group in ai_tickers.values() for t in group]

# SOXX: holds 30 largest semiconductor companies
# QQQ: holds 100 largest non-financial companies on the NASDAQ, heavily weighted towards big tech
def make_features(close: pd.Series,
                     volume: pd.Series,
                     soxx: pd.Series,
                     qqq: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)

    # Lag returns: percent change over x days
    for lag in [1, 3, 5, 10, 20]:
        df[f'ret_{lag}d'] = close.pct_change(lag)

    # Rolling volatility
    df['vol_5d']  = close.pct_change().rolling(5).std()
    df['vol_20d'] = close.pct_change().rolling(20).std()

    # RSI (14-day): Relative Strength Index; measures magnitude and recent price changes in stocks
    delta = close.pct_change()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / loss))

    # How does the stock compare to SOXX and QQQ?
    df['rel_soxx'] = close.pct_change(20) - soxx.pct_change(20)
    df['rel_qqq']  = close.pct_change(20) - qqq.pct_change(20)

    # Volume spike: number of shares traded on a given day
    df['vol_spike'] = volume / volume.rolling(20).mean()

    # 52-week high proximity: the highest price traded at over the last 252 trading days (1 year)
    df['pct_from_52wk_high'] = close / close.rolling(252).max() - 1

    return df.dropna()

end = date.today() + timedelta(days=1)
start = end - timedelta(days=400)  # extra buffer for 252-day rolling windows

prices_live = yf.download(
    tickers=all_tickers + ['SOXX', 'QQQ'],
    start=start,
    end=end,
    auto_adjust=True,
    progress=False
)

close_live = prices_live['Close']
volume_live = prices_live['Volume']

# Build features for each ticker
live_features = []
for ticker in all_tickers:
    feats = make_features(
        close=close_live[ticker],
        volume=volume_live[ticker],
        soxx=close_live['SOXX'],
        qqq=close_live['QQQ']
    )
    feats['Ticker'] = ticker
    live_features.append(feats)

live_df = pd.concat(live_features).reset_index()
live_df = live_df.set_index(['Date', 'Ticker'])

# Get the most recent row per ticker (should be June 18)
latest = live_df.groupby('Ticker').tail(1)

feat_cols = [
    'vol_5d', 'vol_20d', 'vol_spike',
    'news_sentiment', 'pct_positive', 'news_count',
    'gpu_mentions', 'capex_up_score', 'capex_down_score',
    'capex_net', 'competitor_mentions', 'ai_sentence_ratio'
]

for col in feat_cols:
    if col not in latest.columns:
        latest[col] = 0
latest[feat_cols] = latest[feat_cols].fillna(0)

probs_today = final_model.predict_proba(latest[feat_cols])[:, 1]

ranking = pd.Series(probs_today, index=latest.index.get_level_values('Ticker'))
ranking = ranking.sort_values(ascending=False)

colors = ['#2ecc71' if p > 0.5 else '#e74c3c' for p in ranking.values[::-1]]

fig = go.Figure(go.Bar(
    x=ranking.values[::-1],
    y=ranking.index[::-1],
    orientation='h',
    marker_color=colors,
    text=[f'{p:.3f}' for p in ranking.values[::-1]],
    textposition='outside',
    textfont=dict(color='black', size=12)
))

fig.add_vline(
    x=0.5,
    line_dash='dash',
    line_color='dark blue',
    annotation_position='bottom right'
)

last_date = latest.index.get_level_values('Date').max().date()

fig.update_layout(
    title=f'AI Sector Stock Rankings — Next 5 Trading Days<br>{last_date}',
    xaxis_title='Predicted Outperformance Probability',
    xaxis=dict(range=[0, 0.75]),
    height=600,
    width=900,
    margin=dict(t=120, r=100, b=40, l=40),
    plot_bgcolor='white',
    showlegend=False,
    annotations=[dict(text='Decision Threshold (0.5)', font=dict(color='black'))]
)

st.set_page_config(page_title="AI Stock Forecasting", layout="wide")
st.title("AI Sector Stock Forecasting Dashboard")
st.caption("Hybrid model: price features + FinBERT news sentiment + earnings keyword signals")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
median_ic = results_df['IC'].median()
icir = results_df['IC'].mean() / results_df['IC'].std()
top_feature = pd.Series(
    final_model.feature_importances_,
    index=feat_cols
).idxmax()
n_tickers = len(all_tickers)

# Baseline median IC
baseline_ics = results_df['baseline_IC'].median() if 'baseline_IC' in results_df.columns else 0.056
ic_delta = median_ic - baseline_ics

col1.metric("Median Walk-forward IC", f"{median_ic:.3f}", f"{ic_delta:+.3f} vs vol-only baseline")
col2.metric("ICIR", f"{icir:.2f}")
col3.metric("Top Feature", top_feature)
col4.metric("Universe", f"{n_tickers} AI stocks")

st.plotly_chart(fig, use_container_width=True)

# Last 30 Days Graph
st.subheader("AI sector performance Last 30 Days")
data = yf.download(all_tickers, period='30d', auto_adjust=True)['Close']
returns_30d = data.pct_change(len(data)-1).iloc[-1].sort_values(ascending=False)
fig = go.Figure(go.Bar(x=returns_30d.index, y=returns_30d.values*100,
    marker_color=['#185FA5' if v>0 else '#993C1D' for v in returns_30d.values]))
fig.update_layout(yaxis_title='30-day return (%)', height=300)
st.plotly_chart(fig, use_container_width=True)

# SHAP chart
st.subheader("What drives predictions: SHAP feature importance")
st.image('outputs/shap_summary.png')