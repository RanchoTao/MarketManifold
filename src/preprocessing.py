from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DEFAULT_TICKERS, PROCESSED_DIR, TABLES_DIR, LOGS_DIR, RunConfig, ensure_directories
from .data_loader import find_archive, inspect_archive, match_target_members, parse_member_dataframe, write_inspection_outputs


def filter_positive_prices(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    prices = pd.to_numeric(df[price_col], errors="coerce")
    return df.loc[prices.notna() & (prices > 0)].copy()


def deduplicate_by_date_ticker(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")


def coverage_ratio(valid_days: int, expected_days: int) -> float:
    return valid_days / expected_days if expected_days else 0.0


def forward_fill_short_gaps(series: pd.Series, first_valid_date: pd.Timestamp, limit: int = 2) -> pd.Series:
    out = series.copy()
    listed_mask = out.index >= first_valid_date
    out.loc[listed_mask] = out.loc[listed_mask].ffill(limit=limit)
    return out


def setup_logging() -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger("marketmanifold.preprocessing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOGS_DIR / "data_processing.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    return logger


def _year_start(end_date: pd.Timestamp, years: int) -> pd.Timestamp:
    return end_date - pd.DateOffset(years=years)


def prepare_data(archive_path: Path | None = None, config: RunConfig | None = None, root: Path | None = None) -> dict:
    config = config or RunConfig()
    root = root or Path.cwd()
    ensure_directories()
    logger = setup_logging()
    archive_path = find_archive(root, str(archive_path) if archive_path else None)
    logger.info("Using archive %s", archive_path)

    inspection = inspect_archive(archive_path)
    write_inspection_outputs(inspection)

    mapping = match_target_members(archive_path, DEFAULT_TICKERS)
    mapping.to_csv(TABLES_DIR / "ticker_mapping_report.csv", index=False)
    matched = mapping[mapping["matched"]].copy()
    if len(matched) < config.min_matched_tickers:
        raise RuntimeError(f"Only {len(matched)} target tickers matched; need at least {config.min_matched_tickers}. See outputs/tables/ticker_mapping_report.csv")

    frames = []
    stats_rows = []
    failed = []
    with zipfile.ZipFile(archive_path) as zf:
        for row in tqdm(matched.to_dict("records"), desc="Reading target members"):
            df, stats = parse_member_dataframe(zf, row["source_member"], row["project_ticker"], row["sector"])
            stats.update(row)
            if df.empty:
                failed.append({"member": row["source_member"], "error": stats.get("error", "empty parsed data")})
            else:
                frames.append(df)
            stats_rows.append(stats)

    if not frames:
        raise RuntimeError("No target ticker data could be parsed from the archive.")

    raw = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    original_first = raw["date"].min()
    original_last = raw["date"].max()
    actual_end = original_last
    actual_start = max(raw["date"].min(), _year_start(actual_end, config.years))
    period = raw[(raw["date"] >= actual_start) & (raw["date"] <= actual_end)].copy()
    trading_days = pd.Index(sorted(period["date"].unique()))

    quality_rows = []
    cleaned_frames = []
    total_anomalies = 0
    for ticker, group in period.groupby("ticker"):
        group = group.sort_values("date").drop_duplicates("date", keep="last")
        sector = group["sector"].iloc[0]
        source_member = group["source_member"].iloc[0]
        raw_valid_days = group["date"].nunique()
        expected = len(trading_days)
        coverage_before = raw_valid_days / expected if expected else 0
        daily = group.set_index("date").reindex(trading_days)
        listed_mask = daily.index >= group["date"].min()
        before_fill_missing = daily["close"].isna().sum()
        daily.loc[listed_mask, "close"] = daily.loc[listed_mask, "close"].ffill(limit=config.max_forward_fill_days)
        filled_days = int(before_fill_missing - daily["close"].isna().sum())
        daily = daily[daily["close"].notna()].copy()
        coverage_after = len(daily) / expected if expected else 0
        returns = np.log(daily["close"]).diff()
        anomalous = int((returns.abs() > 0.5).sum())
        total_anomalies += anomalous
        kept = coverage_after >= config.min_coverage
        drop_reason = "" if kept else f"coverage_after_fill {coverage_after:.3f} < {config.min_coverage:.3f}"
        if kept:
            out = daily.reset_index(names="date")
            out["ticker"] = ticker
            out["sector"] = sector
            out["source_member"] = source_member
            cleaned_frames.append(out[["date", "ticker", "sector", "close", "source_member"]])
        stat = next((s for s in stats_rows if s.get("project_ticker") == ticker), {})
        quality_rows.append({
            "ticker": ticker,
            "sector": sector,
            "source_member": source_member,
            "first_date": group["date"].min().date().isoformat(),
            "last_date": group["date"].max().date().isoformat(),
            "expected_trading_days": expected,
            "raw_valid_days": raw_valid_days,
            "coverage_before_fill": coverage_before,
            "filled_days": filled_days,
            "coverage_after_fill": coverage_after,
            "duplicate_count": stat.get("duplicate_count", 0),
            "invalid_date_count": stat.get("invalid_date_count", 0),
            "invalid_price_count": stat.get("invalid_price_count", 0),
            "anomalous_return_count": anomalous,
            "kept": kept,
            "drop_reason": drop_reason,
        })

    quality = pd.DataFrame(quality_rows).sort_values("ticker")
    quality.to_csv(TABLES_DIR / "data_quality_report.csv", index=False)
    if not cleaned_frames:
        raise RuntimeError("All matched tickers were dropped by coverage rules.")
    prices_long = pd.concat(cleaned_frames, ignore_index=True).sort_values(["date", "ticker"])
    kept_tickers = sorted(prices_long["ticker"].unique())
    common_counts = prices_long.groupby("date")["ticker"].nunique()
    common_days = common_counts[common_counts == len(kept_tickers)].index
    prices_long = prices_long[prices_long["date"].isin(common_days)].copy()
    prices_wide = prices_long.pivot(index="date", columns="ticker", values="close").sort_index()
    prices_wide = prices_wide.dropna(axis=0, how="any")
    prices_long = prices_wide.reset_index().melt(id_vars="date", var_name="ticker", value_name="close")
    sector_map = mapping.set_index("project_ticker")["sector"].to_dict()
    member_map = mapping.set_index("project_ticker")["source_member"].to_dict()
    prices_long["sector"] = prices_long["ticker"].map(sector_map)
    prices_long["source_member"] = prices_long["ticker"].map(member_map)
    prices_long = prices_long[["date", "ticker", "sector", "close"]].sort_values(["date", "ticker"])

    returns = np.log(prices_wide).diff().dropna()
    lower = returns.stack().quantile(config.winsor_lower)
    upper = returns.stack().quantile(config.winsor_upper)
    returns = returns.clip(lower=lower, upper=upper)

    anomalies = []
    for ticker in prices_wide.columns:
        raw_ret = np.log(prices_wide[ticker]).diff()
        for date, value in raw_ret[raw_ret.abs() > 0.5].items():
            anomalies.append({"date": date.date().isoformat(), "ticker": ticker, "log_return": value})
    pd.DataFrame(anomalies).to_csv(TABLES_DIR / "data_anomalies.csv", index=False)

    prices_long.to_csv(PROCESSED_DIR / "prices_long.csv", index=False)
    prices_wide.to_csv(PROCESSED_DIR / "prices_wide.csv")
    returns.to_csv(PROCESSED_DIR / "log_returns.csv")

    invalid_price_count = int(sum(s.get("invalid_price_count", 0) for s in stats_rows))
    duplicate_count = int(sum(s.get("duplicate_count", 0) for s in stats_rows))
    summary = {
        "source": inspection["possible_data_source"],
        "archive_path": str(archive_path.resolve()),
        "detected_schema": inspection["detected_schemas"],
        "price_field_used": "close",
        "price_adjustment_status": "unknown",
        "original_first_date": original_first.date().isoformat(),
        "original_last_date": original_last.date().isoformat(),
        "actual_start_date": pd.Timestamp(prices_wide.index.min()).date().isoformat(),
        "actual_end_date": pd.Timestamp(prices_wide.index.max()).date().isoformat(),
        "requested_ticker_count": len(DEFAULT_TICKERS),
        "matched_ticker_count": int(mapping["matched"].sum()),
        "kept_ticker_count": len(kept_tickers),
        "dropped_ticker_count": int((~quality["kept"]).sum()),
        "final_trading_day_count": int(len(prices_wide)),
        "final_row_count": int(len(prices_long)),
        "sector_count": int(prices_long["sector"].nunique()),
        "duplicate_count": duplicate_count,
        "invalid_price_count": invalid_price_count,
        "anomalous_return_count": int(len(anomalies) or total_anomalies),
        "failed_members": failed,
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (TABLES_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Prepared %s tickers and %s trading days", len(kept_tickers), len(prices_wide))
    return summary
