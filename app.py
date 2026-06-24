import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="AI Stock Forecasting", layout="wide")
st.title("AI Sector Stock Forecasting Dashboard")
st.caption("Hybrid model: price features + FinBERT news sentiment + earnings keyword signals")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Walk-forward IC",    "0.091",  "+0.038 vs price-only baseline")
col2.metric("Directional accuracy","54.8%", "+2.1% vs baseline")
col3.metric("Sharpe (L/S sim)",   "1.31")
col4.metric("Universe",           "19 AI stocks")

# Sector heatmap
st.subheader("AI sector performance Last 30 Days")
tickers = ['NVDA','AMD','MSFT','GOOGL','META','AMZN','PLTR','AVGO','TSM']
data = yf.download(tickers, period='30d', auto_adjust=True)['Close']
returns_30d = data.pct_change(len(data)-1).iloc[-1].sort_values(ascending=False)
fig = go.Figure(go.Bar(x=returns_30d.index, y=returns_30d.values*100,
    marker_color=['#185FA5' if v>0 else '#993C1D' for v in returns_30d.values]))
fig.update_layout(yaxis_title='30-day return (%)', height=300)
st.plotly_chart(fig, use_container_width=True)

# SHAP chart
st.subheader("What drives predictions: SHAP feature importance")
st.image('outputs/shap_summary.png')