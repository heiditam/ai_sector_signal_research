# NLP-Driven Equity Signals: Predicting AI Sector Stock Performance Using SEC EDGAR Filings

**Contributors:** Heidi Tam <br>
**Creation Date:** June 8, 2026 <br>
**Last Updated:** July 25, 2026

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