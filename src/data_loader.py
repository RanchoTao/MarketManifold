from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd

from .config import DEFAULT_TICKERS, SECTORS, TABLES_DIR, TICKER_ALIASES, ensure_directories


TEXT_EXTENSIONS = {".txt", ".csv", ".dat"}
STANDARD_COLUMNS = {
    "<TICKER>": "ticker",
    "<PER>": "period",
    "<DATE>": "date",
    "<TIME>": "time",
    "<OPEN>": "open",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
    "<VOL>": "volume",
    "<OPENINT>": "open_interest",
    "ticker": "ticker",
    "per": "period",
    "date": "date",
    "time": "time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adjusted_close",
    "adjusted close": "adjusted_close",
    "volume": "volume",
    "vol": "volume",
    "openint": "open_interest",
    "open interest": "open_interest",
}


def find_archive(root: Path, archive: str | None = None) -> Path:
    if archive:
        path = Path(archive)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Archive not found: {path}")
        return path
    candidates = sorted(set(root.glob("*.zip")) | set(root.glob("d_us_txt*")), key=lambda p: p.name.lower())
    candidates = [p for p in candidates if p.is_file() and p.suffix.lower() == ".zip"]
    if not candidates:
        raise FileNotFoundError("No ZIP archive found. Place d_us_txt.zip in the project root or pass --archive.")
    return candidates[0]


def normalize_ticker(value: str | None) -> str:
    if value is None:
        return ""
    ticker = str(value).strip().upper()
    ticker = re.sub(r"\.US$", "", ticker)
    ticker = ticker.replace("_", "-")
    if ticker in {"BRK-B", "BRKB"}:
        return "BRK.B"
    if ticker in {"BF-B", "BFB"}:
        return "BF.B"
    return ticker


def ticker_from_member(member: str) -> str:
    name = PurePosixPath(member).name
    if "." in name:
        name = name.rsplit(".", 1)[0]
    return normalize_ticker(name)


def _decode_sample(raw: bytes) -> tuple[str, str]:
    for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace"), "latin1-replace"


def sniff_text(raw: bytes) -> dict:
    text, encoding = _decode_sample(raw)
    lines = text.splitlines()
    sample = "\n".join(lines[:20])
    delimiter = ","
    has_header = False
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        delimiter = max(counts, key=counts.get)
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        first = lines[0] if lines else ""
        has_header = bool(re.search(r"[A-Za-z<>]", first))
    first_line = lines[0] if lines else ""
    fields = [x.strip() for x in first_line.split(delimiter)] if first_line else []
    date_format = "YYYYMMDD" if re.search(r"\b\d{8}\b", sample) else "unknown"
    return {
        "encoding": encoding,
        "delimiter": delimiter,
        "has_header": has_header,
        "first_line": first_line,
        "fields": fields,
        "date_format": date_format,
        "sample_lines": lines[:20],
    }


def inspect_archive(archive_path: Path, max_samples: int = 5) -> dict:
    ensure_directories()
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        files = [i for i in infos if not i.is_dir()]
        ext_counts = Counter(PurePosixPath(i.filename).suffix.lower() or "<no_ext>" for i in files)
        depth_counts = Counter(len(PurePosixPath(i.filename).parts) for i in infos)
        dir_counts = Counter("/".join(PurePosixPath(i.filename).parts[:d]) for i in infos for d in [1, 2, 3] if len(PurePosixPath(i.filename).parts) >= d)
        text_files = [i for i in files if PurePosixPath(i.filename).suffix.lower() in TEXT_EXTENSIONS]
        chosen = []
        for idx in [0, len(text_files) // 2, len(text_files) - 1]:
            if 0 <= idx < len(text_files):
                chosen.append(text_files[idx])
        for ticker in ["aapl", "msft", "brk", "googl", "xom"]:
            hit = next((i for i in text_files if ticker in PurePosixPath(i.filename).name.lower()), None)
            if hit:
                chosen.append(hit)
        dedup = []
        seen = set()
        for item in chosen:
            if item.filename not in seen:
                dedup.append(item)
                seen.add(item.filename)
            if len(dedup) >= max_samples:
                break
        sample_reports = {}
        for item in dedup:
            with zf.open(item) as fh:
                sample_reports[item.filename] = sniff_text(fh.read(8192))

    schemas = []
    for report in sample_reports.values():
        cols = [STANDARD_COLUMNS.get(c.strip().lower(), STANDARD_COLUMNS.get(c.strip(), c.strip())) for c in report["fields"]]
        schemas.append(cols)

    all_member_names = [i.filename for i in infos]
    likely_stooq = any("data/daily/us/" in name.lower() for name in all_member_names) and any(
        "<TICKER>" in report["first_line"].upper() and "<PER>" in report["first_line"].upper()
        for report in sample_reports.values()
    )
    possible_source = "Stooq-style daily US text archive (not independently confirmed)" if likely_stooq else "unknown"
    organization = "one file per instrument" if len(text_files) > 1000 and all(PurePosixPath(i.filename).name.lower().endswith(".txt") for i in text_files[:50]) else "unknown"

    result = {
        "archive_path": str(archive_path.resolve()),
        "archive_size_bytes": archive_path.stat().st_size,
        "member_count": len(infos),
        "uncompressed_size_bytes": sum(i.file_size for i in infos),
        "file_extensions": dict(ext_counts),
        "directory_depths": dict(sorted(depth_counts.items())),
        "top_internal_directories": dict(dir_counts.most_common(30)),
        "first_30_members": all_member_names[:30],
        "sample_members": [i.filename for i in dedup],
        "detected_encodings": {k: v["encoding"] for k, v in sample_reports.items()},
        "detected_delimiters": {k: v["delimiter"] for k, v in sample_reports.items()},
        "detected_schemas": {k: schemas[idx] for idx, k in enumerate(sample_reports)},
        "possible_date_formats": sorted(set(v["date_format"] for v in sample_reports.values())),
        "possible_ticker_formats": ["AAPL.US in <TICKER> field", "lowercase filename such as aapl.us.txt"],
        "possible_data_source": possible_source,
        "organization": organization,
        "sample_reports": sample_reports,
        "inspection_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result


def write_inspection_outputs(inspection: dict) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "archive_inspection.json").write_text(json.dumps(inspection, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Archive Sample Report",
        "",
        f"- Archive: `{inspection['archive_path']}`",
        f"- Members: {inspection['member_count']}",
        f"- Uncompressed bytes: {inspection['uncompressed_size_bytes']}",
        f"- Organization: {inspection['organization']}",
        f"- Possible source: {inspection['possible_data_source']}",
        "",
        "## First 30 Members",
        "",
    ]
    lines.extend(f"- `{name}`" for name in inspection["first_30_members"])
    lines.append("")
    lines.append("## Samples")
    for member, report in inspection["sample_reports"].items():
        lines.extend([
            "",
            f"### `{member}`",
            "",
            f"- Encoding: `{report['encoding']}`",
            f"- Delimiter: `{report['delimiter']}`",
            f"- Header: `{report['has_header']}`",
            f"- Date format: `{report['date_format']}`",
            f"- Fields: `{', '.join(report['fields'])}`",
            "",
            "```text",
        ])
        lines.extend(report["sample_lines"][:20])
        lines.append("```")
    (TABLES_DIR / "archive_sample_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_member_index(archive_path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            suffix = PurePosixPath(info.filename).suffix.lower()
            if suffix not in TEXT_EXTENSIONS:
                continue
            member_ticker = ticker_from_member(info.filename)
            rows.append({
                "source_member": info.filename,
                "filename_ticker": member_ticker,
                "member_size": info.file_size,
            })
    return pd.DataFrame(rows)


def match_target_members(archive_path: Path, tickers: Iterable[str] = DEFAULT_TICKERS) -> pd.DataFrame:
    index = build_member_index(archive_path)
    by_ticker = defaultdict(list)
    for row in index.to_dict("records"):
        by_ticker[row["filename_ticker"]].append(row)
    rows = []
    for project_ticker in tickers:
        aliases = [normalize_ticker(a) for a in TICKER_ALIASES.get(project_ticker, [project_ticker])]
        hits = []
        for alias in aliases:
            hits.extend(by_ticker.get(alias, []))
        if hits:
            chosen = sorted(hits, key=lambda r: (0 if "stocks" in r["source_member"].lower() else 1, r["source_member"]))[0]
            rows.append({
                "project_ticker": project_ticker,
                "source_member": chosen["source_member"],
                "source_ticker": PurePosixPath(chosen["source_member"]).stem.upper(),
                "normalized_ticker": normalize_ticker(chosen["filename_ticker"]),
                "sector": SECTORS.get(project_ticker, "Unknown"),
                "matched": True,
                "reason": "matched by filename ticker/alias",
            })
        else:
            rows.append({
                "project_ticker": project_ticker,
                "source_member": "",
                "source_ticker": "",
                "normalized_ticker": "",
                "sector": SECTORS.get(project_ticker, "Unknown"),
                "matched": False,
                "reason": "no matching member found",
            })
    return pd.DataFrame(rows)


def parse_member_dataframe(zf: zipfile.ZipFile, member: str, project_ticker: str, sector: str) -> tuple[pd.DataFrame, dict]:
    stats = {"member": member, "error": "", "invalid_date_count": 0, "invalid_price_count": 0, "duplicate_count": 0}
    try:
        with zf.open(member) as fh:
            raw = fh.read()
    except Exception as exc:
        stats["error"] = str(exc)
        return pd.DataFrame(), stats
    sample = sniff_text(raw[:8192])
    text, _ = _decode_sample(raw)
    sample_fields = [str(field).strip().lower() for field in sample.get("fields", [])]
    recognized_header = any(field in STANDARD_COLUMNS for field in sample_fields) or any(
        str(field).strip() in STANDARD_COLUMNS for field in sample.get("fields", [])
    )
    header = 0 if sample["has_header"] and recognized_header else None
    names = None
    if header is None:
        names = ["ticker", "period", "date", "time", "open", "high", "low", "close", "volume", "open_interest"]
    df = pd.read_csv(io.StringIO(text), sep=sample["delimiter"], header=header, names=names, engine="python")
    df.columns = [STANDARD_COLUMNS.get(str(c).strip().lower(), STANDARD_COLUMNS.get(str(c).strip(), str(c).strip().lower())) for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        stats["error"] = f"required columns missing: {list(df.columns)}"
        return pd.DataFrame(), stats
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    stats["invalid_date_count"] = int(df["date"].isna().sum())
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    invalid_price = df["close"].isna() | (df["close"] <= 0)
    stats["invalid_price_count"] = int(invalid_price.sum())
    if "ticker" in df.columns:
        source_ticker = df["ticker"].dropna().astype(str).head(1)
        source_ticker = source_ticker.iloc[0] if len(source_ticker) else PurePosixPath(member).stem.upper()
    else:
        source_ticker = PurePosixPath(member).stem.upper()
    df = df.loc[df["date"].notna() & ~invalid_price, ["date", "close"]].copy()
    before = len(df)
    df = df.drop_duplicates(["date"], keep="last")
    stats["duplicate_count"] = int(before - len(df))
    df["ticker"] = project_ticker
    df["source_ticker"] = source_ticker
    df["sector"] = sector
    df["source_member"] = member
    df = df.sort_values("date")
    return df[["date", "ticker", "sector", "close", "source_member", "source_ticker"]], stats
