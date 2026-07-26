# NLP-Driven Equity Signals: Predicting AI Sector Stock Performance Using SEC EDGAR Filings

**Contributors:** Heidi Tam <br>
**Creation Date:** June 8, 2026 <br>
**Last Updated:** July 26, 2026

## Overview
[type stuff here]

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
4) The data is collected from https://en.wikipedia.org/wiki/List_of_S%26P_500_companies, FinnHub API, and Edgar. Run the file called `data_cleaning.ipynb` located in `notebooks/cleaned/data_cleaning.ipynb` to ensure all relevant CSVs are created and stored in the correct folder (`data`).

5) To train and test the data and view the corresponding dashboard, in your terminal, run ```streamlit run app.py```. A localhost window will appear.

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
├── data/
│   ├── ai_sector_features_df.csv
│   ├── clean_prices.csv
│   ├── features_df.csv
│   ├── prices_raw.csv
│   ├── sentiment_df.csv
│   ├── tickers.csv
│   └── transcript_signals_df.csv
├── notebooks/
│   ├── cleaned/
│   │   ├── data_cleaning.ipynb
│   │   └── training.ipynb
│   └── messy/                      # For development history
│       ├── data_cleaning.ipynb
│       └── training.ipynb
├── outputs/
│   ├── feature_importance.csv
│   ├── ic_by_fold.png
│   ├── ic_over_folds.png
│   ├── results_df.csv
│   ├── results_table.tex
│   ├── shap_force.png
│   ├── shap_summary.png
│   └── shap_waterfall.png
├── .gitignore
├── README.md
├── app.py
├── environment.yml
└── requirements.txt
```

## Conclusion
This study confirmed that SEC EDGAR 8-K Filings do add measurable predictive value beyond the volatility-only baseline. With a mean model IC of 0.0693 compared to the baseline IC of 0.0542, the overall IC improved by 0.0148, with a +0.0266 higher IC in the fifth fold. Additionally, `ai_sentence_ratio` and `gpu_mentions` were top performers in feature importance, reaffirming the idea that language does contain stock-specific signal. However, although the addition of natural language processing applied to SEC EDGAR 8-K Filings resulted in an overall improvement in model performance, these features were ranked as less significant compared to their volatility and volume counterparts, indicating predictive value is added but volatility and volume still primarily dominate. Moreover, the application of NLP on current events articles for each of the 19 tickers did not have any feature importance to the model, although this is likely an infrastructure issue in having current news text correspond with the historical data drawn, rather than a true lack of sentiment value. Overall, the signal appears to be dependent on the time frame of the test data; lower performance was noted during the COVID-19 Pandemic (2020-2023) and the Federal Reserve rate hiking cycle (March 2022-July 2023). 
