from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .clustering import agglomerative_labels, ari, kmeans_labels, safe_silhouette, sector_scores
from .config import PROCESSED_DIR, REPORTS_DIR, TABLES_DIR, RunConfig, ensure_directories
from .data_loader import find_archive, inspect_archive, write_inspection_outputs
from .manifold import align_to_previous, correlation_distance, mds_coordinates, pca_coordinates, standardize_window
from .metrics import window_metrics
from .prediction import run_prediction
from .preprocessing import prepare_data
from .reporting import write_report_tex, write_structural_interpretation
from .rolling_windows import rolling_window_slices
from .visualization import create_animation, plot_data_coverage, plot_key_snapshots, plot_metric_timeseries, plot_prediction


def _coordinate_norm(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0)
    return float(np.sqrt(np.sum(centered * centered) / len(centered)))


def _mean_pairwise_distance(values: np.ndarray) -> float:
    distances = []
    for idx in range(len(values)):
        diff = values[idx + 1:] - values[idx]
        distances.extend(np.sqrt(np.sum(diff * diff, axis=1)))
    return float(np.mean(distances)) if distances else float("nan")


def write_alignment_diagnostics(coords: pd.DataFrame) -> pd.DataFrame:
    rows = []
    aligned_cols = ["x_aligned", "y_aligned"] if {"x_aligned", "y_aligned"}.issubset(coords.columns) else ["x", "y"]
    for window_id, group in coords.groupby("window_id"):
        raw = group[["x_raw", "y_raw"]].to_numpy(float)
        aligned = group[aligned_cols].to_numpy(float)
        raw_norm = _coordinate_norm(raw)
        aligned_norm = _coordinate_norm(aligned)
        raw_distance = _mean_pairwise_distance(raw)
        aligned_distance = _mean_pairwise_distance(aligned)
        rows.append({
            "window_id": int(window_id),
            "raw_coordinate_norm": raw_norm,
            "aligned_coordinate_norm": aligned_norm,
            "norm_ratio": aligned_norm / raw_norm if raw_norm else np.nan,
            "raw_mean_pairwise_distance": raw_distance,
            "aligned_mean_pairwise_distance": aligned_distance,
            "distance_ratio": aligned_distance / raw_distance if raw_distance else np.nan,
        })
    diagnostics = pd.DataFrame(rows).sort_values("window_id")
    diagnostics.to_csv(TABLES_DIR / "alignment_diagnostics.csv", index=False)
    return diagnostics


def run_inspection(archive: str | None = None, root: Path | None = None) -> dict:
    root = root or Path.cwd()
    archive_path = find_archive(root, archive)
    result = inspect_archive(archive_path)
    write_inspection_outputs(result)
    return result


def run_analysis(config: RunConfig | None = None) -> dict:
    config = config or RunConfig()
    ensure_directories()
    returns_path = PROCESSED_DIR / "log_returns.csv"
    prices_path = PROCESSED_DIR / "prices_long.csv"
    summary_path = TABLES_DIR / "data_summary.json"
    if not returns_path.exists() or not prices_path.exists():
        raise FileNotFoundError("Processed data not found. Run: python scripts/prepare_data.py")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    prices_long = pd.read_csv(prices_path, parse_dates=["date"])
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    sectors = prices_long.drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()
    windows = rolling_window_slices(returns.index, config.window, config.step)
    if not windows:
        raise RuntimeError(f"Not enough return rows for window={config.window}.")

    metric_rows = []
    coord_rows = []
    cluster_rows = []
    prev_aligned = None
    prev_labels = None
    tickers = list(returns.columns)
    n_clusters = min(config.clusters, max(2, len(tickers) - 1))

    for window_id, (start, end, start_date, end_date) in enumerate(tqdm(windows, desc="Analyzing rolling windows")):
        wret = returns.iloc[start:end].copy()
        standardized = standardize_window(wret)
        corr = wret.corr().fillna(0.0)
        distance = correlation_distance(corr)
        raw_coords = mds_coordinates(distance, config.random_seed + window_id)
        aligned = align_to_previous(prev_aligned, raw_coords)
        labels = agglomerative_labels(distance, n_clusters=n_clusters)
        pca_coords, pca_ratio_by_stock = pca_coordinates(wret, config.random_seed)
        km_labels = kmeans_labels(standardized.T, n_clusters=n_clusters, random_state=config.random_seed)
        silhouette = safe_silhouette(distance, labels)
        adjacent = ari(prev_labels, labels)
        sector_ari, sector_nmi = sector_scores([sectors.get(t, "Unknown") for t in tickers], labels)
        stock_vol = wret.std(ddof=1) * np.sqrt(252)
        metrics = window_metrics(wret, corr, distance, aligned)
        metrics.update({
            "window_id": window_id,
            "window_start": start_date.date().isoformat(),
            "window_end": end_date.date().isoformat(),
            "cluster_count": int(len(set(labels))),
            "silhouette_score": silhouette,
            "adjacent_window_ari": adjacent,
            "sector_cluster_ari": sector_ari,
            "sector_cluster_nmi": sector_nmi,
            "pca_stock_projection_pc1_ratio": pca_ratio_by_stock,
        })
        metric_rows.append(metrics)
        for idx, ticker in enumerate(tickers):
            coord_rows.append({
                "window_id": window_id,
                "window_start": start_date.date().isoformat(),
                "window_end": end_date.date().isoformat(),
                "ticker": ticker,
                "sector": sectors.get(ticker, "Unknown"),
                "x_raw": raw_coords.loc[ticker, "x_raw"],
                "y_raw": raw_coords.loc[ticker, "y_raw"],
                "x": aligned.loc[ticker, "x"],
                "y": aligned.loc[ticker, "y"],
                "x_aligned": aligned.loc[ticker, "x"],
                "y_aligned": aligned.loc[ticker, "y"],
                "x_pca": pca_coords.loc[ticker, "x_pca"],
                "y_pca": pca_coords.loc[ticker, "y_pca"],
                "cluster": int(labels[idx]),
                "kmeans_cluster": int(km_labels[idx]),
                "stock_volatility": float(stock_vol[ticker]),
            })
            cluster_rows.append({
                "window_id": window_id,
                "window_start": start_date.date().isoformat(),
                "window_end": end_date.date().isoformat(),
                "ticker": ticker,
                "sector": sectors.get(ticker, "Unknown"),
                "cluster": int(labels[idx]),
                "kmeans_cluster": int(km_labels[idx]),
            })
        prev_aligned = aligned
        prev_labels = labels

    metrics_df = pd.DataFrame(metric_rows).sort_values("window_id")
    coords_df = pd.DataFrame(coord_rows).sort_values(["window_id", "ticker"])
    clusters_df = pd.DataFrame(cluster_rows).sort_values(["window_id", "ticker"])
    metrics_df.to_csv(TABLES_DIR / "window_metrics.csv", index=False)
    coords_df.to_csv(TABLES_DIR / "window_coordinates.csv", index=False)
    clusters_df.to_csv(TABLES_DIR / "window_clusters.csv", index=False)
    diagnostics_df = write_alignment_diagnostics(coords_df)

    quality = pd.read_csv(TABLES_DIR / "data_quality_report.csv")
    plot_data_coverage(quality)
    plot_metric_timeseries(metrics_df)
    key_ids = plot_key_snapshots(coords_df, metrics_df)
    animation = create_animation(coords_df, metrics_df)
    pred_metrics, pred_values = run_prediction(metrics_df, target="market_volatility")
    pred_metrics.to_csv(TABLES_DIR / "prediction_metrics.csv", index=False)
    pred_values.to_csv(TABLES_DIR / "prediction_values.csv", index=False)
    plot_prediction(pred_values)
    write_structural_interpretation(metrics_df)
    write_report_tex(summary, metrics_df, pred_metrics)
    pdf_status = compile_report()
    result = {
        "window_count": len(metrics_df),
        "ticker_count": len(tickers),
        "key_snapshot_window_ids": key_ids,
        "animation": animation,
        "prediction_models": pred_metrics["model"].tolist(),
        "report_pdf_status": pdf_status,
        "alignment_last_distance_ratio": float(diagnostics_df["distance_ratio"].iloc[-1]),
    }
    (TABLES_DIR / "pipeline_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def compile_report() -> str:
    tex = REPORTS_DIR / "report.tex"
    if not tex.exists():
        return "report.tex not found"
    try:
        proc = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "report.tex"],
            cwd=REPORTS_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except FileNotFoundError:
        return "xelatex not found"
    except Exception as exc:
        return f"xelatex failed: {exc}"
    return "compiled" if proc.returncode == 0 else f"xelatex returned {proc.returncode}"


def run_full_pipeline(archive: str | None = None, config: RunConfig | None = None, root: Path | None = None) -> dict:
    config = config or RunConfig()
    root = root or Path.cwd()
    archive_path = find_archive(root, archive)
    summary = prepare_data(archive_path=archive_path, config=config, root=root)
    analysis = run_analysis(config)
    return {"data_summary": summary, "analysis_summary": analysis}
