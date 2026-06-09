# Raw data

`src/dynamic_market_regime.py` expects an optional raw price panel at:

```text
data/raw/sp500_prices.csv
```

Required columns:

```text
date,ticker,sector,close
```

If this file is absent, the script creates a deterministic S&P-500-like panel for 2020-01-01 to 2025-06-30 so the full project remains reproducible without external API keys. To use real data, replace `sp500_prices.csv` with adjusted close prices and rerun:

```bash
python src/dynamic_market_regime.py
```
