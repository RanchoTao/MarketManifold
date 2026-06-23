from __future__ import annotations

import argparse
from pathlib import Path

from src.config import RunConfig
from src.pipeline import run_analysis, run_full_pipeline, run_inspection
from src.preprocessing import prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MarketManifold data mining pipeline.")
    parser.add_argument("--archive", default=None, help="Path to d_us_txt.zip")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--window", type=int, default=90, choices=[60, 90, 120])
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = RunConfig(years=args.years, window=args.window, step=args.step, clusters=args.clusters)
    if args.inspect_only:
        result = run_inspection(args.archive, root)
        print(f"Inspection complete: {result['member_count']} ZIP members.")
        return
    if args.prepare_only:
        result = prepare_data(archive_path=Path(args.archive) if args.archive else None, config=config, root=root)
        print(f"Preparation complete: {result['kept_ticker_count']} tickers, {result['final_trading_day_count']} trading days.")
        return
    if args.analysis_only:
        result = run_analysis(config)
        print(f"Analysis complete: {result['window_count']} windows.")
        return
    result = run_full_pipeline(args.archive, config, root)
    data = result["data_summary"]
    analysis = result["analysis_summary"]
    print("MarketManifold pipeline complete.")
    print(f"Tickers kept: {data['kept_ticker_count']} / {data['requested_ticker_count']}")
    print(f"Trading days: {data['final_trading_day_count']}")
    print(f"Rolling windows: {analysis['window_count']}")
    print(f"GIF: {analysis['animation']['gif']}")
    print(f"MP4 status: {analysis['animation']['mp4_status']}")
    print(f"Report PDF: {analysis['report_pdf_status']}")


if __name__ == "__main__":
    main()

