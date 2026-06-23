"""Download and clean representative S&P 500 daily prices from Yahoo Finance."""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised in network-enabled runtime
    yf = None

try:
    from market_config import SECTORS, ticker_sector_map
except ImportError:
    from src.market_config import SECTORS, ticker_sector_map
DEFAULT_START = "2021-06-23"
DEFAULT_END = "2026-06-23"
DEFAULT_OUTPUT = "data/raw/sp500_prices.csv"
CACHE_DIR = Path("data/cache/yfinance")
REPORT_PATH = Path("results/data_quality_report.csv")
SUMMARY_PATH = Path("results/data_download_summary.json")
LOG_PATH = Path("results/data_download_log.txt")


def yahoo_ticker(ticker: str) -> str:
    """Map project tickers to Yahoo Finance ticker syntax."""
    return ticker.upper().replace(".", "-")


def chunks(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def end_exclusive(end: str) -> str:
    return (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).date().isoformat()


def normalize_yfinance_frame(df: pd.DataFrame, request_to_project: Dict[str, str]) -> pd.DataFrame:
    """Convert yfinance single/multi ticker output to date,ticker,close,volume rows."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "volume"])
    data = df.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.tz_localize(None) if data.index.tz is not None else data.index
    rows = []
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(map(str, data.columns.get_level_values(0)))
        field_first = bool({"Close", "Adj Close", "Volume"} & level0)
        for req, project in request_to_project.items():
            try:
                sub = data.xs(req, axis=1, level=1) if field_first else data.xs(req, axis=1, level=0)
            except (KeyError, ValueError):
                continue
            close_col = "Close" if "Close" in sub.columns else "Adj Close" if "Adj Close" in sub.columns else None
            if close_col is None:
                continue
            tmp = pd.DataFrame({"date": sub.index, "ticker": project, "close": sub[close_col]})
            tmp["volume"] = sub["Volume"] if "Volume" in sub.columns else np.nan
            rows.append(tmp)
    else:
        close_col = "Close" if "Close" in data.columns else "Adj Close" if "Adj Close" in data.columns else None
        if close_col is not None and len(request_to_project) == 1:
            project = next(iter(request_to_project.values()))
            tmp = pd.DataFrame({"date": data.index, "ticker": project, "close": data[close_col]})
            tmp["volume"] = data["Volume"] if "Volume" in data.columns else np.nan
            rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "close", "volume"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out[["date", "ticker", "close", "volume"]]


def cache_path(ticker: str, start: str, end_excl: str) -> Path:
    return CACHE_DIR / f"{ticker}_{start}_{end_excl}.csv"


def write_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")
    print(message)


def download_batch(tickers: List[str], start: str, end_excl: str, retries: int = 3) -> pd.DataFrame:
    req = {yahoo_ticker(t): t for t in tickers}
    for attempt in range(1, retries + 1):
        try:
            write_log(f"Downloading batch {tickers} attempt {attempt}/{retries}")
            df = yf.download(list(req), start=start, end=end_excl, interval="1d", auto_adjust=True, group_by="column", progress=False, threads=True)
            long = normalize_yfinance_frame(df, req)
            success = sorted(set(long.loc[long["close"].notna(), "ticker"]))
            failed = sorted(set(tickers) - set(success))
            write_log(f"Batch success={success}; failed={failed}")
            if success:
                return long
        except Exception as exc:  # network/library errors should not abort whole run
            write_log(f"Batch error for {tickers}: {exc!r}")
        time.sleep(1.5 * attempt)
    return pd.DataFrame(columns=["date", "ticker", "close", "volume"])


def load_or_download(tickers: List[str], start: str, end_excl: str, batch_size: int, force: bool) -> Tuple[pd.DataFrame, List[str]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames, missing = [], []
    for t in tickers:
        cp = cache_path(t, start, end_excl)
        if cp.exists() and not force:
            frames.append(pd.read_csv(cp))
        else:
            missing.append(t)
    for batch_no, batch in enumerate(chunks(missing, batch_size), 1):
        write_log(f"Starting batch {batch_no}: {batch}")
        got = download_batch(batch, start, end_excl)
        success = set(got["ticker"].unique()) if not got.empty else set()
        for t in batch:
            part = got[got["ticker"] == t]
            if part.empty:
                write_log(f"Falling back to single ticker download for {t}")
                part = download_batch([t], start, end_excl)
            if not part.empty:
                part.to_csv(cache_path(t, start, end_excl), index=False)
                frames.append(part)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "close", "volume"])
    failed = sorted(set(tickers) - set(raw.loc[raw["close"].notna(), "ticker"].unique()))
    return raw, failed


def clean_panel(raw: pd.DataFrame, min_coverage: float, max_ffill: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    sector_map = ticker_sector_map()
    tickers = sorted(sector_map)
    df = raw.copy()
    before = len(df)
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = df.get("ticker", "").astype(str).str.upper()
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    if "volume" in df:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    duplicate_count = int(df.duplicated(["date", "ticker"], keep=False).sum())
    df = df.dropna(subset=["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    valid_mask = np.isfinite(df["close"]) & (df["close"] > 0)
    invalid_price_rows = int((~valid_mask).sum())
    df = df.loc[valid_mask & df["ticker"].isin(tickers), [c for c in ["date", "ticker", "close", "volume"] if c in df.columns]]
    base_dates = sorted(df["date"].unique())
    expected = len(base_dates)
    wide = df.pivot(index="date", columns="ticker", values="close").reindex(base_dates)
    volume_wide = df.pivot(index="date", columns="ticker", values="volume").reindex(base_dates) if "volume" in df else None
    report_rows, kept = [], []
    filled_close = pd.DataFrame(index=base_dates)
    filled_volume = pd.DataFrame(index=base_dates) if volume_wide is not None else None
    for t in tickers:
        s = wide[t] if t in wide else pd.Series(index=base_dates, dtype=float)
        raw_valid = int(s.notna().sum())
        cov_before = raw_valid / expected if expected else 0.0
        first = s.first_valid_index() or ""
        last = s.last_valid_index() or ""
        # Limit filling to gaps after the first observed price; never bfill pre-history.
        sf = s.ffill(limit=max_ffill)
        filled_days = int((s.isna() & sf.notna()).sum())
        remaining = int(sf.isna().sum())
        cov_after = int(sf.notna().sum()) / expected if expected else 0.0
        kept_flag = cov_before >= min_coverage and raw_valid > 0
        reason = "" if kept_flag else ("download_failed" if raw_valid == 0 else "coverage_below_threshold")
        if kept_flag:
            kept.append(t)
            filled_close[t] = sf
            if volume_wide is not None:
                filled_volume[t] = volume_wide[t].ffill(limit=max_ffill) if t in volume_wide else np.nan
        report_rows.append({
            "ticker": t, "sector": sector_map[t], "request_ticker": yahoo_ticker(t),
            "first_date": first, "last_date": last, "expected_days": expected,
            "raw_valid_days": raw_valid, "coverage_before_fill": round(cov_before, 6),
            "filled_days": filled_days if kept_flag else 0, "remaining_missing_days": remaining if kept_flag else expected - raw_valid,
            "coverage_after_fill": round(cov_after if kept_flag else cov_before, 6), "kept": kept_flag, "drop_reason": reason,
        })
    common_dates = filled_close.index[filled_close[kept].notna().all(axis=1)].tolist() if kept else []
    final = filled_close.loc[common_dates, kept].stack().reset_index()
    final.columns = ["date", "ticker", "close"]
    final["sector"] = final["ticker"].map(sector_map)
    final = final[["date", "ticker", "sector", "close"]].sort_values(["date", "ticker"]).reset_index(drop=True)
    stats = {"input_rows": before, "duplicate_rows": duplicate_count, "invalid_price_rows": invalid_price_rows, "base_trading_days": expected, "common_trading_days": len(common_dates), "total_filled_days": int(sum(r["filled_days"] for r in report_rows))}
    return final, pd.DataFrame(report_rows), stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.add_argument("--max-ffill", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if yf is None:
        raise RuntimeError("yfinance is not installed; run pip install -r requirements.txt before downloading Yahoo Finance data.")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    sector_map = ticker_sector_map()
    tickers = sorted(sector_map)
    end_excl = end_exclusive(args.end)
    raw, failed = load_or_download(tickers, args.start, end_excl, args.batch_size, args.force)
    final, report, stats = clean_panel(raw, args.min_coverage, args.max_ffill)
    if final["ticker"].nunique() < 80:
        write_log(f"WARNING: fewer than 80 tickers kept ({final['ticker'].nunique()}); inspect report for reasons.")
    if final.empty:
        raise RuntimeError("No valid market data remained after cleaning.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    kept = sorted(final["ticker"].unique())
    summary = {
        "source": "Yahoo Finance via yfinance", "requested_start": args.start, "requested_end": args.end,
        "download_end_exclusive": end_excl, "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "auto_adjust": True, "interval": "1d", "requested_ticker_count": len(tickers),
        "downloaded_ticker_count": int(report.query('raw_valid_days > 0').shape[0]), "kept_ticker_count": len(kept),
        "dropped_ticker_count": len(tickers) - len(kept), "failed_tickers": failed,
        "kept_sectors": sorted(final["sector"].unique()), "final_first_date": str(final["date"].min()),
        "final_last_date": str(final["date"].max()), "final_trading_day_count": int(final["date"].nunique()),
        "final_row_count": int(len(final)), "min_coverage": args.min_coverage, "max_forward_fill_days": args.max_ffill,
        "cleaning": stats,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_log(f"Wrote {args.output}: {len(kept)} tickers, {summary['final_trading_day_count']} days, {len(final)} rows")

if __name__ == "__main__":
    main()
