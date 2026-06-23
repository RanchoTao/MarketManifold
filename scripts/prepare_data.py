from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RunConfig
from src.preprocessing import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare target stock data from ZIP without full extraction.")
    parser.add_argument("--archive", default=None)
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()
    summary = prepare_data(
        archive_path=Path(args.archive) if args.archive else None,
        config=RunConfig(years=args.years),
        root=Path(__file__).resolve().parents[1],
    )
    print(f"Kept tickers: {summary['kept_ticker_count']}")
    print(f"Trading days: {summary['final_trading_day_count']}")
    print("Wrote processed CSV files and data quality reports.")


if __name__ == "__main__":
    main()

