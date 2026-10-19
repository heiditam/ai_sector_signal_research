# NLP-Driven Equity Signals: Predicting AI Sector Stock Performance Using SEC EDGAR Filings

**Contributors:** Heidi Tam <br>
**Creation Date:** June 8, 2026 <br>
**Last Updated:** October 19, 2026

## Overview
## Background

Since the public launch of OpenAI's large language model ChatGPT in November 2022, generative artificial intelligence (AI) has rapidly permeated nearly every industry, including healthcare, financial services, and software engineering. Within three years, generative AI has become the catalyst for agentic AI, transforming artificial intelligence from passively answering prompts to autonomously operating software, tools, and APIs. This rapid diffusion has triggered a surge of investment in AI infrastructure: in 2025, over 60% of the total capex spent by the four largest hyperscalers (Amazon, Microsoft, Google, and Meta) went toward AI, including GPUs and data centers (\$156B of \$260B collectively).

This raises a question for investors:

> *As firms increasingly describe AI demand, GPU capacity, and capital spending in their earnings disclosures, does that language contain additional information about future relative returns that prices and volatility do not already capture?*

### Why SEC 8-K filings?

We use SEC Form 8-K filings, which report material events such as revenue guidance and unaudited quarterly results. Specifically, we focus on **Item 2.02 (Results of Operations)**, whose earnings releases (Exhibit 99.1) are *furnished* rather than *filed*. They face less legal scrutiny than audited reports and therefore more directly reflect how management chooses to frame its results, including how much emphasis it places on AI demand, GPU capacity, and capital spending. These filings are also mandated, timestamped, and consistently structured across firms, making them well suited to systematic analysis. Rather than relying solely on backward-looking price and volume signals, we test whether filing text provides forward-looking information.

### A methodological concern

Financial machine learning results are especially prone to **lookahead bias**, where information unavailable at prediction time leaks into the features or label and produces large, spuriously significant results. In our own pipeline, an off-by-one error in constructing the five-day forward-return target initially produced an IC of **0.061**, which disappeared once the target was corrected. We therefore treat leakage as a research question in its own right and measure how much it inflates results.

## Research Questions

| | Question |
|---|---|
| **RQ1** | Do NLP signals extracted from SEC EDGAR 8-K filings (AI demand language, GPU mentions, and capital expenditure signals) add measurable predictive value beyond a volatility-only baseline for cross-sectional stock ranking in the AI sector? |
| **RQ2** | How sensitive are cross-sectional IC estimates to lookahead leakage in target construction, and does the inflation scale with the degree of overlap between the label window and the feature date? |

## Contributions

1. **AI-specific text dictionaries.** We construct keyword dictionaries covering AI demand, GPU, capital expenditure, and competitive language commonly used by semiconductor and AI-infrastructure firms.
2. **A leakage-controlled evaluation.** We test these signals with purged walk-forward cross-validation against a volatility-only model and find no statistically significant incremental value (Newey-West p = 0.57), a result robust to adding news sentiment and momentum features.
3. **A leakage dose-response analysis.** We show that even one day of overlap between the label window and the feature date turns this null result into a highly significant one, and that a baseline comparison fails to detect the leak.

## Running the Project
1) Navigate into your respective folder and run the following command in your command line or terminal: <br>
```git clone https://github.com/heiditam/ai_sector_signal_research.git```

2) Set up and activate the environment using conda: <br>
```conda env create -f environment.yml```<br>
```conda activate ai-sector-signals```

3) Dependencies (see environment.yml):
```dotenv==0.9.9
edgartools==5.36.0
finnhub-python==2.4.28
geopandas==0.8.1
lightgbm==4.6.0
matplotlib==3.9.2
matplotlib-inline==0.1.6
matplotlylib==0.1.0
numpy==2.1.3
pandas==2.2.2
plotly==5.23.0
plotly-geo==1.0.0
python-dotenv==1.1.1
scikit-learn==1.6.1
scikit-surprise==1.1.4
sentence-transformers==5.0.0
shap==0.48.0
streamlit==1.58.0
transformers==4.51.3
yfinance==1.2.2 
```
4) **Build the dataset.** Data comes from three sources:
   - **Prices:** daily prices from [yfinance](https://github.com/ranaroussi/yfinance)
   - **Filings:** SEC EDGAR Form 8-K filings (Item 2.02, Exhibit 99.1 earnings releases)
   - **News (robustness check only):** [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)

   Run `notebooks/cleaned/version1/data_cleaning.ipynb` to download and process the data. All CSVs are saved to the `data/` folder.

5) **Reproduce the main results.** From the repository root, run:
```python scripts/full_model.py```
  This trains the walk-forward model and prints the per-fold IC, mean fold IC, fold ICIR, and mean daily IC reported in the paper. The text ablation, leakage analysis, and figures are in `notebooks/cleaned/training.ipynb`.

6) **(Optional) Dashboard demo.** Run `streamlit run app.py` to open a local dashboard with live price-based rankings. Note: the dashboard is a demonstration and does not reproduce the paper's results.

## Expected Outputs
```
results/
├── results_df.csv
├── results_table.tex
├── feature_importance.csv
├── shap_summary.png
├── shap_waterfall.csv
├── ic_by_fold.png
└── ic_over_folds.png
````
Outputs include: <br>
* Model results table
* Feature importance table
* IC by Fold bar chart - visualizes the model vs. baseline performance across folds
* IC Over Folds - shows the cumulative IC over folds

## File Structure
```
ai_sector_signal_research/
├── .devcontainer/
│   └── devcontainer.json
├── .github/
│   └── workflows/
│       └── collect_news.yml          # Daily Tiingo news collector (GitHub Action)
├── data/
│   ├── ai_sector_features_df.csv     # Main panel: prices, volatility, EDGAR features, target
│   ├── ai_sector_features_2026.csv   # 2026 holdout panel
│   ├── ai_sector_features_news_v2.csv
│   ├── clean_prices.csv
│   ├── clean_prices_2026.csv
│   ├── features_df.csv
│   ├── prices_raw.csv
│   ├── prices_raw_2026.csv
│   ├── sentiment_df.csv
│   ├── tickers.csv
│   ├── transcript_signals_df.csv     # Per-filing EDGAR keyword signals
│   ├── fnspid/
│   │   ├── news_19.parquet           # FNSPID headlines for the 19 tickers
│   │   └── news_daily_features.csv   # Daily FinBERT sentiment features
│   ├── news_cache/                   # Legacy per-ticker news cache (TICKER_YEAR.csv)
│   └── news_live/                    # Point-in-time Tiingo news (YYYY-MM.csv)
├── notebooks/
│   ├── cleaned/
│   │   ├── version1/                 # Main study
│   │   │   ├── data_cleaning.ipynb
│   │   │   └── training.ipynb        # Ablation, leakage analysis, figures
│   │   └── version2/                 # Extended features (features_v2)
│   │       ├── data_cleaning_v2.ipynb
│   │       └── training_v2.ipynb
│   └── messy/                        # Development history
│       ├── data_cleaning.ipynb
│       └── training.ipynb
├── outputs/
│   ├── htmls/                        # Notebook exports
│   ├── pdfs/                         # Notebook exports
│   ├── feature_importance.csv
│   ├── ic_by_fold.png
│   ├── ic_over_folds.png
│   ├── leakage_dose_response.csv
│   ├── leakage_dose_response.png
│   ├── leakage_tests.csv
│   ├── main_results.csv
│   ├── results_df.csv
│   ├── results_table.tex
│   ├── shap_summary.png
│   └── shap_waterfall.png
├── scripts/
│   ├── full_model.py                 # Reproduces the main results
│   ├── collect_news.py               # Tiingo news collector
│   ├── fnspid_news.py                # FNSPID filtering and FinBERT scoring
│   ├── features_v2.py                # Next-version text features
│   ├── test_2026.py                  # 2026 holdout evaluation
│   └── tune_2019_2025.py
├── .gitignore
├── README.md
├── app.py                            # Streamlit dashboard (demo)
├── environment.yml
└── requirements.txt
```

## Conclusion
## Key Results

| Question | Finding |
|---|---|
| **RQ1:** Does AI-related filing language add value beyond volatility? | **No.** Text ΔIC = 0.0082 vs. a volatility-only model (Newey-West p = 0.57) |
| **RQ2:** Does lookahead leakage inflate results? | **Yes.** One day of label overlap makes the null highly significant (p < 0.001); IC rises ~0.008 per day of overlap (p = 0.012) |

## Conclusion

### RQ1: No incremental value from AI-related language

This study tested whether AI-related language in SEC 8-K earnings releases adds predictive value beyond volatility when ranking AI-sector stocks over a five-day horizon. With our model, it does not.

- The model achieved a mean fold IC of **0.0185**, but its mean daily IC of **0.015** was not significantly different from zero (Newey-West p = 0.19).
- Relative to a volatility-only model trained on the same folds, the text features added no statistically detectable value (**ΔIC = 0.0082, p = 0.57**).
- This conclusion held when news sentiment from FNSPID or standard momentum features were added.
- Feature importance was dominated by `vol_20d` (20-day rolling volatility), and early stopping retained a single tree in every fold, indicating little learnable structure in the text features.

One possible explanation is that AI terminology became nearly universal after 2022, so keyword counts no longer distinguish between firms. A more promising approach would measure how a company's language **changes relative to its own past filings**, rather than how much AI language it uses in absolute terms.

### RQ2: Lookahead leakage manufactures significance

The study's second finding concerns methodology. An error in the original target construction produced an apparently strong signal (**IC = 0.061**) that disappeared once corrected. The leakage analysis showed that:

- Even **one day** of overlap between the label window and the feature date turns a non-significant result into a highly significant one (p < 0.001).
- IC rises by roughly **0.008 per additional day** of overlap (p = 0.012).
- Because the volatility baseline was inflated in parallel, **outperforming the baseline offered no warning**.

Apparent signals from financial text should therefore be treated with caution until label construction has been verified directly. A carefully tested null result can be as informative as a positive one.

### Interpreting the null result

These findings should not be read as evidence that AI-related disclosure language contains no information. The null result applies to keyword-count features, a 19-stock universe, and a five-day horizon, and the design has limited statistical power: given the standard error of the daily IC, a true IC of roughly **0.03** would be needed for reliable detection, so a smaller but genuine effect could go undetected.

Prior work has shown that textual information in filings can predict returns when measured with richer methods and larger samples (e.g., Cohen, Malloy & Nguyen, 2020). Whether AI-related language carries a similar signal remains an open question, best addressed with context-aware text representations, a larger point-in-time universe, and longer horizons.
