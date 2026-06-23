from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_inspection


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the MarketManifold ZIP archive without extracting it.")
    parser.add_argument("--archive", default=None, help="Path to ZIP archive. Defaults to first d_us_txt*.zip in project root.")
    args = parser.parse_args()
    result = run_inspection(args.archive, Path(__file__).resolve().parents[1])
    print(f"Archive: {result['archive_path']}")
    print(f"Members: {result['member_count']}")
    print(f"Possible source: {result['possible_data_source']}")
    print("Wrote outputs/tables/archive_inspection.json and archive_sample_report.md")


if __name__ == "__main__":
    main()

