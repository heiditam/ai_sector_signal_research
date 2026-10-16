# imports
import streamlit as st
import lightgbm as lgb
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from datetime import date, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Stock Forecasting", layout="wide")

# Full model (scripts/full_model.py): mean fold IC = 0.0185, fold ICIR = 0.6671, mean daily IC = 0.0149
import sys
sys.path.insert(0, 'scripts')
from full_model import FULL_MODEL_FEATS, load_full_model_data, make_splits, fit_full_model

@st.cache_resource
def train_evaluate():
    df = load_full_model_data('data/ai_sector_features_df.csv')
    all_results = []
    for fold_num, (train_df, test_df) in enumerate(make_splits(df)):
        model = fit_full_model(train_df)
        X_te, y_te = test_df[FULL_MODEL_FEATS], test_df['Target']
        probs = model.predict_proba(X_te)[:, 1]
        all_results.append({
            'fold': fold_num + 1,
            'IC': round(np.corrcoef(probs, y_te)[0, 1], 4),
            'accuracy': round(accuracy_score(y_te, probs > 0.5), 4),
            'baseline_IC': round(np.corrcoef(X_te['vol_5d'].fillna(0), y_te)[0, 1], 4),
        })
    return pd.DataFrame(all_results), model   # final model = fold-5 model

results_df, final_model = train_evaluate()

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

@st.cache_data(ttl=3600) #cache for 1 hour
def load_price_data(all_tickers, start, end):
    return yf.download(
        tickers=list(all_tickers) + ['SOXX', 'QQQ'],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

prices_live = load_price_data(tuple(all_tickers), start, end)

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

feat_cols = FULL_MODEL_FEATS   # EDGAR features are not available live and are set to 0 below

# feature importance
importance = pd.Series(
    final_model.feature_importances_, 
    index=feat_cols
).sort_values(ascending=False)
importance.to_csv('outputs/feature_importance.csv')

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
    title=f'AI Sector Stock Overall Rankings — Next 5 Trading Days<br>{last_date}',
    xaxis_title='Predicted Outperformance Probability',
    xaxis=dict(range=[0, 0.75]),
    height=600,
    width=900,
    margin=dict(t=120, r=100, b=40, l=40),
    plot_bgcolor='white',
    showlegend=False,
    annotations=[dict(text='Decision Threshold (0.5)', font=dict(color='black'))]
)

st.title("AI Sector Stock Forecasting Dashboard")
st.caption("Full model: volatility features + EDGAR filing keyword signals (LightGBM)")

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
baseline_ics = results_df['baseline_IC'].median()
ic_delta = median_ic - baseline_ics

col1.metric("Median Walk-forward IC", f"{median_ic:.3f}", f"{ic_delta:+.3f} vs vol-only baseline")
col2.metric("ICIR", f"{icir:.2f}")
col3.metric("Top Feature", top_feature)
col4.metric("Universe", f"{n_tickers} AI stocks")

st.plotly_chart(fig, use_container_width=True)

# Separated by bucket (type of company)
st.subheader("Rankings by Sector Bucket")

bucket_map = {}
for bucket, tickers_list in ai_tickers.items():
    for ticker in tickers_list:
        bucket_map[ticker] = bucket

bucket_labels = {
    'semis': 'Semiconductors',
    'hypers': 'Hyperscalers',
    'pure': 'AI Pure Plays',
    'infra': 'Infrastructure'
}

ranking_df = ranking.reset_index()
ranking_df.columns = ['Ticker', 'Probability']
ranking_df['Bucket'] = ranking_df['Ticker'].map(bucket_map)

cols = st.columns(4)
for i, (bucket, label) in enumerate(bucket_labels.items()):
    bucket_df = ranking_df[ranking_df['Bucket'] == bucket].sort_values('Probability')
    
    with cols[i]:
        st.markdown(f"**{label}**")
        fig_bucket = go.Figure(go.Bar(
            x=bucket_df['Probability'],
            y=bucket_df['Ticker'],
            orientation='h',
            marker_color=['#2ecc71' if p > 0.5 else '#e74c3c' for p in bucket_df['Probability']],
            text=[f'{p:.3f}' for p in bucket_df['Probability']],
            textposition='outside',
            textfont=dict(color='black', size=10)
        ))
        fig_bucket.add_vline(x=0.5, line_dash='dash', line_color='gray')
        fig_bucket.update_layout(
            height=350,
            xaxis=dict(range=[0, 0.75]),
            margin=dict(t=20, r=60, b=20, l=20),
            plot_bgcolor='white',
            showlegend=False
        )
        st.plotly_chart(fig_bucket, use_container_width=True)

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

# Other visualizations
# IC by fold bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df['IC'], width, label='Full model IC', color='#2ecc71')
ax.bar(x + width/2, results_df['baseline_IC'], width, label='Baseline IC', color='#95a5a6')
ax.set_xlabel('Fold')
ax.set_ylabel('Information Coefficient')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Full Model IC vs Volatility-Only Baseline by Fold')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax.legend()
plt.tight_layout()
plt.savefig('outputs/ic_by_fold.png', dpi=150, bbox_inches='tight')
st.image('outputs/ic_by_fold.png', width=700)

# Cumulative IC Over Time
fig, ax = plt.subplots(figsize=(10, 5))
folds = [f'Fold {i+1}' for i in range(5)]
model_ics = results_df['IC'].values
baseline_ics = results_df['baseline_IC'].values

ax.plot(folds, model_ics, marker='o', color='#2ecc71', linewidth=2, label='Full model IC')
ax.plot(folds, baseline_ics, marker='s', color='#95a5a6', linewidth=2, linestyle='--', label='Baseline IC')
ax.fill_between(folds, model_ics, baseline_ics, alpha=0.2, color='#2ecc71')
ax.set_xlabel('Fold')
ax.set_ylabel('Information Coefficient')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Full Model vs Baseline IC Across Walk-Forward Folds')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/ic_over_folds.png', dpi=150, bbox_inches='tight')
st.image('outputs/ic_over_folds.png', width=700)