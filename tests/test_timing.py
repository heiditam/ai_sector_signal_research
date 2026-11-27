# IMPORTS
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# SET-UP
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from score_sentiment import assign_trading_date
from features import make_features

PANEL = ROOT / 'outputs/v3/data/ai_sector_features_df.csv'

## RUN TESTS WITH python3 -m pytest tests -q


# Test 1: fwd_5d is the sum of returns from t+1 to t+5
def test_fwd_5d_is_next_five_returns():
    df = pd.read_csv(PANEL, usecols=['Date', 'Ticker', 'ret_1d', 'fwd_5d'], parse_dates=['Date'])
    ret = df.pivot(index='Date', columns='Ticker', values='ret_1d')
    true_fwd = sum(ret.shift(-k) for k in range(1, 6)).stack().rename('true_fwd')   # t+1 .. t+5
    m = df.set_index(['Date', 'Ticker']).join(true_fwd, how='inner')
    assert len(m) > 0.95 * len(df)                      # nearly every row was checked
    assert np.allclose(m['fwd_5d'], m['true_fwd'])


# Test 2: Validate that news and filings are counted on the first trading day whose 4 PM close comes after them
TRADING_DAYS = pd.bdate_range('2025-02-24', '2025-07-31').drop(pd.Timestamp('2025-07-04'))

# hand-picked edge cases
@pytest.mark.parametrize('published, expected', [
    ('2025-03-03T20:59:00Z', '2025-03-03'),   # Mon 3:59pm EST -> same day
    ('2025-03-03T21:00:00Z', '2025-03-04'),   # Mon 4:00pm EST -> next day
    ('2025-03-07T23:00:00Z', '2025-03-10'),   # Fri evening -> Mon
    ('2025-03-08T15:00:00Z', '2025-03-10'),   # Saturday -> Mon
    ('2025-03-03',           '2025-03-04'),   # date only -> treated as after close
    ('2025-07-01T19:59:00Z', '2025-07-01'),   # 3:59pm EDT (summer time)
    ('2025-07-01T20:00:00Z', '2025-07-02'),   # 4:00pm EDT
    ('2025-07-03T21:00:00Z', '2025-07-07'),   # Thu after close, Fri Jul 4 is a holiday
])
def test_assign_trading_date_cases(published, expected):
    got = assign_trading_date(pd.Series([published]), TRADING_DAYS)[0]
    assert got == pd.Timestamp(expected)

# property test: checks any timestamp lands on Day D
def check_first_close_after(t_utc, got, trading_days):
    td = pd.DatetimeIndex(trading_days)
    closes = (td + pd.Timedelta(hours=16)).tz_localize('America/New_York')
    i = td.get_indexer(pd.DatetimeIndex(got))
    assert (i >= 0).all()
    assert (closes[i] > t_utc).all()                               # known before D's close
    assert ((i == 0) | (closes[np.maximum(i - 1, 0)] <= t_utc)).all()   # and not one day too late

def test_random_timestamps():
    rng = np.random.default_rng(0)
    t = pd.Timestamp('2025-03-01', tz='UTC') + pd.to_timedelta(rng.integers(0, 120 * 24 * 60, 5000), unit='min')
    got = assign_trading_date(pd.Series(t.strftime('%Y-%m-%dT%H:%M:%SZ')), TRADING_DAYS)
    check_first_close_after(t, got, TRADING_DAYS)

# Test 3: Verify that the merge only uses past filings 
def test_text_features_come_from_past_filings():
    df = pd.read_csv(PANEL, usecols=['Date', 'FilingDay'], parse_dates=['Date', 'FilingDay'])
    used = df.dropna(subset=['FilingDay'])
    assert (used['FilingDay'] <= used['Date']).all()
    assert ((used['Date'] - used['FilingDay']) <= pd.Timedelta(days=120)).all()


# TEST 4: Perturbation Test - Price features never look ahead
# Checks that each price feature on day t uses only data up to day t
def test_price_features_ignore_the_future():
    n = 600                        # make_features drops the first 251 days (52-week high needs 252 days)
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2024-01-01', periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), dates)
    volume = pd.Series(rng.integers(100_000, 1_000_000, n).astype(float), dates)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), dates)
    t = dates[400]                 # days 251-400 have complete features
    future = dates > t

    base = make_features(close, volume, bench, bench)
    moved = make_features(close.mask(future, close * 3), volume.mask(future, volume * 3),
                          bench.mask(future, bench * 3), bench.mask(future, bench * 3))
    assert len(base.loc[:t]) > 100          # make sure rows were actually compared
    pd.testing.assert_frame_equal(base.loc[:t], moved.loc[:t])


# TEST 5: the development panels contain no 2026 rows, and no rows whose fwd_5d uses 2026 prices
@pytest.mark.parametrize('name', ['features_df.csv', 'ai_sector_features_df.csv'])
def test_development_panel_stops_before_2026(name):
    dates = pd.read_csv(ROOT / 'outputs/v3/data' / name, usecols=['Date'], parse_dates=['Date'])['Date']
    assert dates.max() <= pd.Timestamp('2025-12-31')      # no holdout rows
    assert dates.max() <= pd.Timestamp('2025-12-23')      # no fwd_5d that reaches into 2026


# TEST 6: the sealed files still match the hashes recorded when they were sealed
SEALED = ROOT / 'outputs/v3/sealed'

def test_sealed_files_unchanged():
    import hashlib
    sums = SEALED / 'SHA256SUMS.txt'
    if not sums.exists():
        pytest.skip('holdout not sealed yet')
    latest = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split()[:2]
        latest[name] = digest                             # the last line for each file wins (reseals)
    for name, digest in latest.items():
        assert hashlib.sha256((SEALED / name).read_bytes()).hexdigest() == digest, f'{name} changed'