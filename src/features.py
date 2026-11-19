import pandas as pd
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

    # Rolling volatility: std of returns
    df['vol_5d']  = close.pct_change().rolling(5).std()
    df['vol_20d'] = close.pct_change().rolling(20).std()

    # RSI (14-day): Relative Strength Index; measures magnitude and recent price changes in stocks
    # whether a stock has been moving up or down more forcefully over its recent trading days
    # > 70: overbought (a lot of upward activity)
    # < 30: oversold (a lot of downward activity)
    # 50: roughly equal up and down movement
    delta = close.pct_change()
    gain  = delta.clip(lower=0).rolling(14).mean() # average of up days
    loss  = (-delta.clip(upper=0)).rolling(14).mean() # average of down days
    df['rsi_14'] = 100 - (100 / (1 + gain / loss))

    # How does the stock compare to SOXX and QQQ?
    df['rel_soxx'] = close.pct_change(20) - soxx.pct_change(20)
    df['rel_qqq']  = close.pct_change(20) - qqq.pct_change(20)

    # Volume spike: number of shares traded on a given day
    df['vol_spike'] = volume / volume.rolling(20).mean()

    # 52-week high proximity: the highest price traded at over the last 252 trading days (1 year)
    df['pct_from_52wk_high'] = close / close.rolling(252).max() - 1

    return df.dropna()