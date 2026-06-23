"""Shared market universe configuration for the MarketManifold project."""

SECTORS = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "AMD", "INTC", "ORCL", "CSCO", "QCOM", "TXN"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA", "TTWO"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "GM", "F"],
    "Consumer Staples": ["WMT", "COST", "PG", "KO", "PEP", "MDLZ", "CL", "KMB", "GIS", "KR"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "SCHW", "SPGI"],
    "Health Care": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY"],
    "Industrials": ["BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "MMM", "FDX"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "WELL"],
    "Materials": ["LIN", "SHW", "APD", "ECL", "FCX", "NEM", "DOW", "NUE"],
}

def ticker_sector_map():
    return {ticker: sector for sector, tickers in SECTORS.items() for ticker in tickers}
