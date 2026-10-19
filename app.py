"""
Streamlit dashboard for the AI-sector 8-K text-signal study.

The model, target, and walk-forward evaluation are imported from scripts/full_model.py,
so every number on this page matches the paper. Live rankings combine today's price-based
volatility features (yfinance) with each firm's most recent 8-K text features from the dataset.

Run from the repository root:
    streamlit run app.py
"""
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from full_model import (  # noqa: E402
    FULL_MODEL_FEATS, VOL_FEATS, load_full_model_data, make_splits, fit_full_model, daily_ic,
)

warnings.filterwarnings("ignore")   # LightGBM deprecation notices; NaN IC on days with constant predictions
st.set_page_config(page_title="AI Sector Text Signals", layout="wide")

DATA_PATH = ROOT / "data" / "ai_sector_features_df.csv"
OUTPUTS = ROOT / "outputs"
TEXT_FEATS = [c for c in FULL_MODEL_FEATS if c not in VOL_FEATS]

AI_TICKERS = {
    "Semiconductors": ["NVDA", "AMD", "INTC", "ASML", "TSM", "AVGO", "MRVL", "ARM", "SMCI"],
    "Hyperscalers":   ["MSFT", "GOOGL", "AMZN", "META"],
    "AI Pure Plays":  ["PLTR", "AI", "SOUN"],
    "Infrastructure": ["DELL", "ANET", "HPE"],
}
ALL_TICKERS = [t for group in AI_TICKERS.values() for t in group]
BUCKET_OF = {t: b for b, ts in AI_TICKERS.items() for t in ts}

GREEN, RED, GRAY, BLUE = "#2e9e5b", "#c0392b", "#95a5a6", "#2c6fbb"


def newey_west(s, lags=20):
    """Newey-West (HAC) t-test that the mean of a daily series is zero."""
    s = pd.Series(s).dropna()
    return sm.OLS(s.values, np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


# --------------------------------------------------------------------------------------
# Study results (same code path as scripts/full_model.py), computed once and cached
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Running the walk-forward evaluation...")
def run_study():
    df = load_full_model_data(DATA_PATH)
    rows, daily_full, daily_vol = [], [], []
    for i, (train_df, test_df) in enumerate(make_splits(df), 1):
        model = fit_full_model(train_df)
        t = test_df.assign(prob=model.predict_proba(test_df[FULL_MODEL_FEATS])[:, 1])
        vol_model = fit_full_model(train_df, VOL_FEATS)
        tv = test_df.assign(prob=vol_model.predict_proba(test_df[VOL_FEATS])[:, 1])
        daily_full.append(daily_ic(t))
        daily_vol.append(daily_ic(tv))
        dates = t.index.get_level_values("Date")
        rows.append({
            "Fold": i,
            "Test period": f"{dates.min():%b %Y} – {dates.max():%b %Y}",
            "Model IC": np.corrcoef(t["prob"], t["Target"])[0, 1],
            "Baseline IC": np.corrcoef(t["vol_5d"].fillna(0), t["Target"])[0, 1],
            "Trees": model.best_iteration_,
        })
    folds = pd.DataFrame(rows)
    folds["Delta IC"] = folds["Model IC"] - folds["Baseline IC"]

    daily_full = pd.concat(daily_full).dropna()
    daily_vol = pd.concat(daily_vol).dropna()
    delta = (daily_full - daily_vol).dropna()

    final_model = fit_full_model(df)   # trained on all 2019-2025 data, used for live rankings
    return {
        "folds": folds,
        "daily_ic": daily_full,
        "nw_daily": newey_west(daily_full),
        "delta": delta,
        "nw_delta": newey_west(delta),
        "final_model": final_model,
    }


@st.cache_data
def latest_text_features():
    """Most recent 8-K keyword features per ticker (already lagged one trading day in the dataset)."""
    raw = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date")
    last = raw.groupby("Ticker")[TEXT_FEATS + ["Date"]].last()
    return last.rename(columns={"Date": "text_as_of"})


# --------------------------------------------------------------------------------------
# Live price features
# --------------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading live prices...")
def load_prices(tickers, start, end):
    return yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)


def live_volatility_features(px):
    """Latest vol_5d, vol_20d, vol_spike per ticker, defined exactly as in data_cleaning.ipynb."""
    if px is None or px.empty:
        return pd.DataFrame()
    close, volume = px["Close"], px["Volume"]
    rows = []
    for t in ALL_TICKERS:
        if t not in close.columns:
            continue
        c = close[t].dropna()
        v = volume[t].reindex(c.index)
        r = c.pct_change()
        f = pd.DataFrame({
            "vol_5d": r.rolling(5).std(),
            "vol_20d": r.rolling(20).std(),
            "vol_spike": v / v.rolling(20).mean(),
        }).dropna()
        if not f.empty:
            rows.append({"Ticker": t, "price_date": f.index[-1], **f.iloc[-1].to_dict()})
    return pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
st.title("AI Sector Text Signals Dashboard")
st.caption("LightGBM on price volatility + SEC 8-K (Item 2.02) keyword features · 19 AI-sector stocks · "
           "five-day cross-sectional ranking")

st.info(
    "**Research demo, not investment advice.** In walk-forward testing (2019–2025), the model's "
    "daily IC was not statistically different from zero, and the 8-K text features added no "
    "significant value over a volatility-only model. Rankings below show what the model predicts, "
    "not a validated trading signal."
)

study = run_study()
folds = study["folds"]
n_beat = int((folds["Model IC"] > folds["Baseline IC"]).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean fold IC", f"{folds['Model IC'].mean():.4f}")
c1.caption(f"vol_5d baseline: {folds['Baseline IC'].mean():.4f}")
c2.metric("Mean daily IC", f"{study['daily_ic'].mean():.4f}")
c2.caption(f"Newey-West p = {study['nw_daily'].pvalues[0]:.2f} (not significant)")
c3.metric("Folds beating baseline", f"{n_beat} / {len(folds)}")
c3.caption(f"Mean delta IC = {folds['Delta IC'].mean():+.4f}")
c4.metric("Text added vs vol-only model", f"{study['delta'].mean():+.4f}")
c4.caption(f"Newey-West p = {study['nw_delta'].pvalues[0]:.2f} (not significant)")


def ranking_bars(frame, x_lim, height, text_size=12):
    """Diverging bars: each stock's predicted probability minus today's cross-sectional median."""
    colors = [GRAY if abs(v) < 1e-9 else (GREEN if v > 0 else RED) for v in frame["rank_vs_median"]]
    fig = go.Figure(go.Bar(
        x=frame["rank_vs_median"], y=frame.index, orientation="h", marker_color=colors,
        text=[f"{p:.3f}" for p in frame["prob"]], textposition="outside",
        textfont=dict(size=text_size), cliponaxis=False,
        customdata=frame["prob"], hovertemplate="%{y}: p = %{customdata:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(height=height, plot_bgcolor="white", showlegend=False,
                      xaxis=dict(range=[-x_lim, x_lim], zeroline=False, tickformat="+.3f"),
                      margin=dict(t=10, r=60, b=40, l=10))
    return fig

# ---- Live rankings ----------------------------------------------------------------------
st.header("Live rankings: next 5 trading days")

end = date.today() + timedelta(days=1)
try:
    prices = load_prices(tuple(ALL_TICKERS), end - timedelta(days=90), end)
    live = live_volatility_features(prices)
except Exception as e:   # network or Yahoo rate limits
    prices, live = None, pd.DataFrame()
    st.warning(f"Live price download failed ({e}).")

if live.empty:
    st.warning("Live prices are unavailable right now (Yahoo Finance may be rate-limiting). "
               "The research results below are unaffected. Try refreshing in a few minutes.")
else:
    text = latest_text_features()
    live = live.join(text, how="left")
    live[TEXT_FEATS] = live[TEXT_FEATS].fillna(0)
    live["prob"] = study["final_model"].predict_proba(live[FULL_MODEL_FEATS])[:, 1]
    live["rank_vs_median"] = live["prob"] - live["prob"].median()
    live["Bucket"] = live.index.map(BUCKET_OF)

    price_date = live["price_date"].max().date()
    text_date = pd.to_datetime(live["text_as_of"]).max().date()
    st.caption(f"Prices as of {price_date}. Text features use each firm's latest 8-K in the dataset "
               f"(through {text_date}). ASML, TSM, and ARM file 6-Ks rather than 8-Ks, so their text "
               "features are zero.")

    ranked = live.sort_values(["rank_vs_median", "vol_20d"])
    x_lim = max(ranked["rank_vs_median"].abs().max() * 1.35, 0.002)

    fig = ranking_bars(ranked, x_lim, height=620)
    fig.update_layout(xaxis_title="Predicted probability minus today's median (labels show the probability)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Green = predicted to beat the median stock; red = predicted to lag it; gray = tied at the "
               "median. Probabilities sit close to 0.5 and the model (one shallow tree) assigns many stocks "
               "the same value, reflecting its weak signal.")

    st.subheader("Rankings by sector bucket")
    cols = st.columns(len(AI_TICKERS))
    for col, bucket in zip(cols, AI_TICKERS):
        with col:
            st.markdown(f"**{bucket}**")
            st.plotly_chart(ranking_bars(ranked[ranked["Bucket"] == bucket], x_lim, height=320, text_size=10),
                            width="stretch")

    # 30-day performance from the same download (no second request)
    close = prices["Close"][[t for t in ALL_TICKERS if t in prices["Close"].columns]].dropna(how="all")
    last_21 = close.tail(22)
    ret_30d = (last_21.iloc[-1] / last_21.iloc[0] - 1).dropna().sort_values(ascending=False) * 100
    st.subheader("Trailing one-month return (21 trading days)")
    fr = go.Figure(go.Bar(x=ret_30d.index, y=ret_30d.values,
                          marker_color=[BLUE if v > 0 else RED for v in ret_30d.values],
                          hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    fr.update_layout(height=320, yaxis_title="Return (%)", plot_bgcolor="white",
                     margin=dict(t=10, r=20, b=40, l=40))
    st.plotly_chart(fr, width="stretch")

# ---- Research results ---------------------------------------------------------------------
st.header("Walk-forward results (2019–2025)")

left, right = st.columns([3, 2])
with left:
    ff = go.Figure()
    ff.add_bar(x=[f"Fold {i}" for i in folds["Fold"]], y=folds["Model IC"], name="Model IC",
               marker_color=GREEN)
    ff.add_bar(x=[f"Fold {i}" for i in folds["Fold"]], y=folds["Baseline IC"],
               name="Baseline IC (vol_5d)", marker_color=GRAY)
    ff.add_hline(y=0, line_color="black", line_width=1)
    ff.update_layout(barmode="group", height=380, yaxis_title="Information Coefficient",
                     plot_bgcolor="white", margin=dict(t=30, r=20, b=40, l=40),
                     legend=dict(orientation="h", y=1.1))
    st.plotly_chart(ff, width="stretch")
with right:
    show = folds[["Fold", "Test period", "Model IC", "Baseline IC", "Delta IC", "Trees"]]
    st.dataframe(show.style.format({"Model IC": "{:.4f}", "Baseline IC": "{:.4f}", "Delta IC": "{:+.4f}"}),
                 hide_index=True, width="stretch")
    st.caption("Each fold's IC carries a standard error of roughly 0.015, so differences across folds "
               "are consistent with noise. Early stopping kept a single tree in every fold.")

st.subheader("What drives predictions")
imp = pd.Series(study["final_model"].booster_.feature_importance("split"), index=FULL_MODEL_FEATS)
imp = imp[imp > 0].sort_values()
fi = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE))
fi.update_layout(height=max(250, 32 * len(imp)), xaxis_title="Number of splits (final model)",
                 plot_bgcolor="white", margin=dict(t=10, r=20, b=40, l=40))
st.plotly_chart(fi, width="stretch")
st.caption("Split counts for the live model, trained on all 2019–2025 data. Split counts show how often the "
           "model uses a feature, not whether it improves prediction.")
if (OUTPUTS / "shap_summary.png").exists():
    with st.expander("SHAP summary (from the training notebook)"):
        st.image(str(OUTPUTS / "shap_summary.png"))

# ---- Leakage analysis ------------------------------------------------------------------------
leak_path = OUTPUTS / "leakage_tests.csv"
if leak_path.exists():
    st.header("Lookahead leakage inflates IC")
    leak = pd.read_csv(leak_path)
    se = leak["IC"] / leak["t"]
    fl = go.Figure()
    fl.add_scatter(x=leak["k"], y=leak["IC"], mode="lines+markers", name="Model daily IC",
                   line=dict(color=GREEN, width=3),
                   error_y=dict(type="data", array=1.96 * se, visible=True))
    fl.add_scatter(x=leak["k"], y=leak["vol_IC"], mode="lines+markers", name="vol_5d daily IC",
                   line=dict(color=GRAY, dash="dot", width=2))
    fl.add_hline(y=0, line_color="black", line_width=1)
    fl.update_layout(height=400, plot_bgcolor="white", margin=dict(t=30, r=20, b=50, l=40),
                     xaxis=dict(title="Days of overlap between label window and feature date (k)",
                                tickvals=list(leak["k"]),
                                ticktext=["0 (correct)", "1", "2", "3", "4 (original bug)"]),
                     yaxis_title="Mean daily IC", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fl, width="stretch")
    p0 = leak.loc[leak["k"] == 0, "p"].iloc[0]
    st.caption(f"With the correct target (k = 0), IC is not significant (p = {p0:.2f}, computed on the days "
               "shared by all five versions); every k ≥ 1 is "
               "significant at p < 0.001. The volatility baseline rises in parallel, so beating it does "
               "not reveal the leak. Error bars are 95% Newey-West confidence intervals.")

st.divider()
st.caption("Code and paper: github.com/heiditam/ai_sector_signal_research")
