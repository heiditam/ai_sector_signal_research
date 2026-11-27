# Pre-registration: v3 confirmatory analysis

**Registered:** 2026-11-27, before rescoring the 8-K filings with the new sentence splitter.
**Rule:** nothing below changes after new results are seen. Any change goes in the deviation log at the end, with its date and reason.

## 1. What had been seen before this date

- v3 daily-model results from runs with mixed settings (Nov 17, old-splitter scores, 2019–2025).
- A preliminary event study, primary horizon only, on old-splitter scores (2026-11-25). No signal was significant (smallest Holm-adjusted p = 0.22).
- An EODHD news coverage check for 2015–2025 (2026-11-27), described in section 3.
- One accidental 2026 holdout run (`outputs/v3/sealed/holdout_2026_accidental_run.csv`). Its results were not viewed.

Not seen: the 5- and 60-day event-study horizons, any 2015–2018 result, any result on the new-splitter scores, any 2026 result, and any S&P 400 data.

## 2. Sample

| | Range |
|---|---|
| Development | 2015-01-01 to 2025-12-31 |
| Price history loaded | from 2014-01-01 (the 52-week-high feature needs 252 prior days) |
| Filings loaded | from 2014-01-01 (so 2015 releases have a previous release to diff against) |
| Time holdout | 2026, sealed in `outputs/v3/sealed/` and used once, after this design is final |
| Stock holdout | S&P 400 constituents (snapshot 2026-11-27) that never appear in the S&P 500 history. Downloaded unopened and unscored, for the agent paper |

- Rows whose target window reaches into 2026 are dropped from development: the last 5 trading days of 2025 for the daily model, and any release whose event window ends in 2026 for the event study.
- **Universe:** point-in-time S&P 500 membership (Wikipedia change history, downloaded 2026-11-21). Stocks are included only on dates they were in the index. Removed stocks that yfinance cannot provide are missing, and the number missing will be reported.

## 3. Features

- **Price features:** returns over 1, 3, 5, 10 and 20 days; volatility over 5 and 20 days; 14-day RSI; return relative to SOXX and QQQ; volume spike; distance from the 52-week high.
- **Filing text:** Exhibit 99 of 8-K Item 2.02 releases, scored with zero-shot NLI (DeBERTa) and FinBERT using the new sentence splitter. Assigned to the first trading day whose 4 pm ET close comes after the SEC acceptance time.
- **News (EODHD):** used **from 2021-01-01 only**, and set to missing before that. The coverage check found news on 0–3% of weekdays before 2021 for typical stocks, compared with 31–63% from 2021 onward.
- All features are ranked as percentiles within GICS sector and date.

## 4. Daily model

- **Target:** 1 if the stock's return over the next 5 trading days beats its GICS sector median that day.
- **Model:** LightGBM with fixed seeds and pinned package versions (`requirements-lock.txt`).
- **Validation:** walk-forward, 5 folds, 21-day gap, 5-day purge, early stopping on the last 15% of each training window.
- **Primary test:** ΔIC = daily rank IC of the full model minus the price-only model, averaged over all development test days. Newey-West standard errors, two-sided, α = 0.05.
- **Secondary tests** (Holm correction within this family):
  - feature-group ablations;
  - what news adds, measured on test days from 2021 to 2025 only.

## 5. Event study (one row per earnings release)

- **Day D:** the first trading day whose 4 pm ET close comes after the filing's acceptance time.
- **Signals:** change since the company's previous release in `ai_demand_net`, `capex_net`, `ai_tone` and `ai_sentence_ratio`.
- **Controls:** announcement return (D to D+1), `ret_20d`, `vol_20d`.
- **Method:** Fama-MacBeth by calendar quarter on percentile ranks, Newey-West with 2 lags.
- **Primary:** sector-relative drift from D+2 to D+21. 4 tests, Holm, α = 0.05.
- **Secondary:** drift from D+2 to D+6 and from D+2 to D+61. 8 tests, Holm within this family.

## 6. Holdouts

- The 2026 holdout is run once, with the frozen model and code, after all development analyses above are complete. Results are reported whatever they show.
- The S&P 400 holdout is not opened until the agent paper's design is registered separately.

## Deviation log 

| Date | Change | Reason |
|---|---|---|
| 2026-11-27 | Universe: 600 tickers = 503 current members + 97 removed members whose yfinance prices cover ≥95% of their membership days (`scripts/build_universe.py`). 162 of 259 removed members are missing. Robustness check added: rerun the primary test on the 503 current members only. Sectors for the 97 entered by hand (current GICS structure). | yfinance has no usable prices for most delisted stocks. Measures how much survivorship bias affects results. |
| 2026-11-27 | 2026 holdout first sealed before the rescore, with old-splitter scores. It will be resealed after the rescore and the 2015 rebuild. | The split cell was run early. The 2026 data was not viewed. |
