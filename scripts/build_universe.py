"""
Build the 2015-2025 point-in-time universe: today's S&P 500 plus removed members with usable prices.

    python scripts/build_universe.py

- Run from the repository root, after data_cleaning_v3 has written sp500_membership.csv
  (from 2014) and prices_removed_2014.csv.
- A removed ticker is kept only if yfinance has a price on at least 95% of the days it was
  in the index. This drops symbols that now belong to a different company (e.g. ADT, TE)
  and tickers where yfinance returned only recent data.
- Sectors: current members use Wikipedia's GICS sector (sp500_tickers.csv). Removed members
  use the hand-entered SECTORS below, in today's GICS structure (post-2023), so they are
  comparable with the current list. Entries marked in CHECK are uncertain.
- Writes data/universe_2015.csv (tracked by git).
"""
import pandas as pd

MEMBERSHIP = 'outputs/v3/data/sp500_membership.csv'
CURRENT = 'outputs/v3/data/sp500_tickers.csv'
REMOVED_PRICES = 'outputs/v3/data/prices_removed_2014.csv'
CHANGES = 'data/sp500_changes.csv'
OUT = 'data/universe_2015.csv'
MIN_COVERAGE = 0.95

CD, CS, EN, FI, HC = 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Health Care'
IN, IT, MA, RE, CO = 'Industrials', 'Information Technology', 'Materials', 'Real Estate', 'Communication Services'

SECTORS = {
    'AA': MA, 'AAL': IN, 'AAP': CD, 'AIV': RE, 'ALK': IN, 'AMG': FI, 'AMTM': IN, 'AN': CD,
    'ATI': MA, 'AYI': IN, 'BBWI': CD, 'BHF': FI, 'BIO': HC, 'BLDR': IN, 'BWA': CD, 'CAG': CS,
    'CE': MA, 'CLF': MA, 'CNX': EN, 'COTY': CS, 'CPB': CS, 'CPRI': CD, 'CTVA': MA, 'CZR': CD,
    'DXC': IT, 'EMN': MA, 'ENPH': IT, 'EPAM': IT, 'ETSY': CD, 'FLR': IN, 'FLS': IN, 'FMC': MA,
    'FOSL': CD, 'FTI': EN, 'GHC': CD, 'GME': CD, 'GNW': FI, 'GT': CD, 'HOG': CD, 'HP': EN,
    'HRB': CD, 'IPGP': IT, 'JEF': FI, 'KMX': CD, 'KSS': CD, 'LKQ': CD, 'LNC': FI, 'LUMN': CO,
    'LW': CS, 'M': CD, 'MAC': RE, 'MAT': CD, 'MBC': IN, 'MHK': CD, 'MKTX': FI, 'MOH': HC,
    'MTCH': CO, 'MUR': EN, 'NAVI': FI, 'NBR': EN, 'NKTR': HC, 'NOV': EN, 'NWL': CD, 'OGN': HC,
    'OI': MA, 'PAYC': IN, 'PBI': IN, 'PENN': CD, 'POOL': CD, 'PRGO': HC, 'PVH': CD, 'R': IN,
    'RHI': IN, 'RIG': EN, 'RRC': EN, 'SEDG': IT, 'SIG': CD, 'SLG': RE, 'SLM': FI, 'SOLS': MA,
    'TAP': CS, 'TDC': IT, 'TFX': HC, 'THC': HC, 'TRIP': CO, 'TTD': IT, 'UAA': CD, 'UNM': FI,
    'URBN': CD, 'VFC': CD, 'VNO': RE, 'VNT': IT, 'WHR': CD, 'WU': FI, 'XRAY': HC, 'XRX': IT,
    'ZION': FI,
}

# Uncertain entries: verify before the rebuild
CHECK = {
    'ATI':  'sector: may have moved from Materials to Industrials (aerospace) in a GICS update',
    'TTD':  'sector: Information Technology or Communication Services',
    'AMTM': 'sector: Industrials (government services) assumed',
    'GHC':  'sector: Consumer Discretionary (education) assumed',
    'AA':   'prices: today\'s Alcoa was spun off in 2016; check that 2014-2016 prices are the old Alcoa',
    'MBC':  'identity: confirm the 2022 removal was MasterBrand',
}


def main():
    mem = pd.read_csv(MEMBERSHIP, index_col=0, parse_dates=True)
    close = pd.read_csv(REMOVED_PRICES, header=[0, 1], index_col=0, parse_dates=True)['Close']
    current = pd.read_csv(CURRENT)

    keep, dropped = [], []
    for t in close.columns[close.notna().any()]:
        days = mem.index[mem[t]] if t in mem else pd.DatetimeIndex([])
        days = days[days.isin(close.index)]
        cov = close.loc[days, t].notna().mean() if len(days) else 0.0
        (keep if cov >= MIN_COVERAGE else dropped).append(t)

    missing = sorted(set(keep) - set(SECTORS))
    if missing:
        raise SystemExit(f'Add a sector for: {missing}')

    changes = pd.read_csv(CHANGES, parse_dates=['Date'])
    changes['Removed'] = changes['Removed'].str.replace('.', '-', regex=False)
    names = changes.sort_values('Date').groupby('Removed')['RemovedName'].last()

    removed = pd.DataFrame({
        'Symbol': sorted(keep),
        'Security': [names.get(t, '') for t in sorted(keep)],
        'GICS Sector': [SECTORS[t] for t in sorted(keep)],
        'GICS Sub-Industry': '',
        'Status': 'removed',
        'Sector source': 'manual',
        'Check': [CHECK.get(t, '') for t in sorted(keep)],
    })
    current = current.assign(**{'Status': 'current', 'Sector source': 'Wikipedia', 'Check': ''})
    universe = pd.concat([current, removed], ignore_index=True)
    assert universe['Symbol'].is_unique
    universe.to_csv(OUT, index=False)

    print(f'{len(current)} current + {len(removed)} removed = {len(universe)} tickers -> {OUT}')
    print(f'{len(dropped)} removed tickers dropped (prices cover < {MIN_COVERAGE:.0%} of membership days): '
          f'{sorted(dropped)}')
    print(f'{len(CHECK)} entries to verify: {sorted(CHECK)}')


if __name__ == '__main__':
    main()
