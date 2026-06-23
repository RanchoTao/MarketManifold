from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
ANIMATIONS_DIR = OUTPUTS_DIR / "animations"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "AVGO", "ORCL", "CRM", "AMD", "INTC", "CSCO", "NFLX", "JPM", "BAC",
    "GS", "MS", "V", "MA", "BRK.B", "UNH", "JNJ", "PFE", "LLY", "XOM",
    "CVX", "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "CAT", "BA",
    "GE", "DIS", "NKE", "T", "VZ", "NEE", "DUK", "AMT", "LIN",
]

SECTORS = {
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AVGO": "Information Technology",
    "ORCL": "Information Technology", "CRM": "Information Technology",
    "AMD": "Information Technology", "INTC": "Information Technology",
    "CSCO": "Information Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "V": "Financials", "MA": "Financials",
    "BRK.B": "Financials",
    "UNH": "Health Care", "JNJ": "Health Care", "PFE": "Health Care",
    "LLY": "Health Care",
    "XOM": "Energy", "CVX": "Energy",
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "CAT": "Industrials", "BA": "Industrials", "GE": "Industrials",
    "LIN": "Materials",
    "NEE": "Utilities", "DUK": "Utilities",
    "AMT": "Real Estate",
}

TICKER_ALIASES = {
    "BRK.B": ["BRK.B", "BRK-B", "BRKB", "BRK.B.US", "BRK-B.US", "BRKB.US"],
    "BF.B": ["BF.B", "BF-B", "BFB", "BF.B.US", "BF-B.US", "BFB.US"],
}

for ticker in DEFAULT_TICKERS:
    TICKER_ALIASES.setdefault(ticker, [ticker, f"{ticker}.US"])


@dataclass(frozen=True)
class RunConfig:
    years: int = 5
    window: int = 90
    step: int = 10
    clusters: int = 6
    min_coverage: float = 0.90
    max_forward_fill_days: int = 2
    winsor_lower: float = 0.005
    winsor_upper: float = 0.995
    min_matched_tickers: int = 30
    random_seed: int = RANDOM_SEED


def ensure_directories() -> None:
    for path in [
        DATA_DIR, PROCESSED_DIR, CACHE_DIR, OUTPUTS_DIR, FIGURES_DIR,
        ANIMATIONS_DIR, TABLES_DIR, LOGS_DIR, REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

