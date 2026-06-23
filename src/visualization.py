from __future__ import annotations

import shutil
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ANIMATIONS_DIR, FIGURES_DIR


LABEL_TICKERS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "BRK.B"}


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_data_coverage(quality: pd.DataFrame) -> None:
    q = quality.sort_values("coverage_after_fill")
    plt.figure(figsize=(10, 6))
    colors = ["#377eb8" if kept else "#e41a1c" for kept in q["kept"]]
    plt.barh(q["ticker"], q["coverage_after_fill"], color=colors)
    plt.axvline(0.9, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Coverage after fill")
    plt.ylabel("Ticker")
    plt.title("Target ticker data coverage")
    _savefig(FIGURES_DIR / "data_coverage.png")


def plot_metric_timeseries(metrics: pd.DataFrame) -> None:
    m = metrics.copy()
    m["window_end"] = pd.to_datetime(m["window_end"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(m["window_end"], m["mean_correlation"], color="#1b9e77")
    axes[0].set_ylabel("Mean corr.")
    axes[1].plot(m["window_end"], m["market_volatility"], color="#d95f02")
    axes[1].set_ylabel("Market vol.")
    axes[2].plot(m["window_end"], m["pca_first_component_ratio"], color="#7570b3")
    axes[2].set_ylabel("PC1 ratio")
    axes[2].set_xlabel("Window end")
    fig.suptitle("Rolling market structure metrics")
    _savefig(FIGURES_DIR / "market_metrics_timeseries.png")

    plt.figure(figsize=(10, 4))
    plt.plot(m["window_end"], m["mean_correlation"], color="#1b9e77")
    plt.title("Rolling average correlation")
    plt.xlabel("Window end")
    plt.ylabel("Mean correlation")
    _savefig(FIGURES_DIR / "correlation_timeseries.png")

    plt.figure(figsize=(10, 4))
    plt.plot(m["window_end"], m["market_volatility"], color="#d95f02")
    plt.title("Rolling equal-weight market volatility")
    plt.xlabel("Window end")
    plt.ylabel("Annualized volatility")
    _savefig(FIGURES_DIR / "volatility_timeseries.png")

    plt.figure(figsize=(10, 4))
    plt.plot(m["window_end"], m["pca_first_component_ratio"], color="#7570b3")
    plt.title("Rolling first principal component strength")
    plt.xlabel("Window end")
    plt.ylabel("Explained variance ratio")
    _savefig(FIGURES_DIR / "pca_factor_strength.png")

    plt.figure(figsize=(10, 4))
    plt.plot(m["window_end"], m["adjacent_window_ari"], label="Adjacent ARI", color="#4daf4a")
    plt.plot(m["window_end"], m["sector_cluster_ari"], label="Sector-cluster ARI", color="#984ea3")
    plt.legend()
    plt.title("Cluster stability and sector alignment")
    plt.xlabel("Window end")
    plt.ylabel("Score")
    _savefig(FIGURES_DIR / "cluster_stability.png")


def plot_snapshot(coords: pd.DataFrame, metrics: pd.DataFrame, window_id: int, path: Path, color_by: str = "cluster") -> None:
    frame = coords[coords["window_id"] == window_id].copy()
    row = metrics[metrics["window_id"] == window_id].iloc[0]
    plt.figure(figsize=(8, 6))
    groups = frame[color_by].astype(str)
    unique = sorted(groups.unique())
    cmap = plt.get_cmap("tab10")
    for idx, group in enumerate(unique):
        sub = frame[groups == group]
        sizes = 60 + 900 * (sub["stock_volatility"] / max(frame["stock_volatility"].max(), 1e-12))
        plt.scatter(sub["x"], sub["y"], s=sizes, alpha=0.75, label=str(group), color=cmap(idx % 10), edgecolor="white", linewidth=0.6)
    for _, r in frame[frame["ticker"].isin(LABEL_TICKERS)].iterrows():
        plt.text(r["x"], r["y"], r["ticker"], fontsize=8, ha="center", va="center")
    plt.axhline(0, color="#dddddd", linewidth=0.8)
    plt.axvline(0, color="#dddddd", linewidth=0.8)
    plt.title(f"Market structure, {row['window_start']} to {row['window_end']}")
    plt.xlabel("MDS dimension 1 (aligned)")
    plt.ylabel("MDS dimension 2 (aligned)")
    plt.legend(title=color_by, fontsize=8, loc="best", frameon=True)
    _savefig(path)


def plot_key_snapshots(coords: pd.DataFrame, metrics: pd.DataFrame) -> list[int]:
    ids = []
    for col in ["market_volatility", "mean_correlation"]:
        ids.append(int(metrics.loc[metrics[col].idxmax(), "window_id"]))
    ids.append(int(metrics.loc[metrics["mean_correlation"].idxmin(), "window_id"]))
    latest = int(metrics["window_id"].max())
    ids = list(dict.fromkeys(ids))
    while len(ids) < 3:
        ids.append(latest)
        ids = list(dict.fromkeys(ids))
    for idx, window_id in enumerate(ids[:3], start=1):
        plot_snapshot(coords, metrics, window_id, FIGURES_DIR / f"key_snapshot_{idx}.png")
    plot_snapshot(coords, metrics, latest, FIGURES_DIR / "latest_market_structure.png")
    return ids[:3]


def plot_prediction(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "Prediction skipped: too few windows", ha="center", va="center")
        plt.axis("off")
        _savefig(FIGURES_DIR / "prediction_comparison.png")
        return
    p = predictions.copy()
    p["window_end"] = pd.to_datetime(p["window_end"])
    plt.figure(figsize=(10, 5))
    plt.plot(p["window_end"], p["actual"], label="Actual", color="black", linewidth=2)
    for col in [c for c in p.columns if c not in {"window_id", "window_end", "actual"}]:
        plt.plot(p["window_end"], p[col], label=col, linewidth=1.5)
    plt.legend()
    plt.title("Next-window market volatility prediction")
    plt.xlabel("Window end")
    plt.ylabel("Market volatility")
    _savefig(FIGURES_DIR / "prediction_comparison.png")


def create_animation(coords: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    xmin, xmax = coords["x"].min(), coords["x"].max()
    ymin, ymax = coords["y"].min(), coords["y"].max()
    padx = (xmax - xmin) * 0.12 or 1
    pady = (ymax - ymin) * 0.12 or 1
    tmp_dir = ANIMATIONS_DIR / "_frames"
    tmp_dir.mkdir(exist_ok=True)
    cmap = plt.get_cmap("tab10")
    for window_id in sorted(coords["window_id"].unique()):
        frame = coords[coords["window_id"] == window_id]
        row = metrics[metrics["window_id"] == window_id].iloc[0]
        fig, ax = plt.subplots(figsize=(7, 5))
        for idx, group in enumerate(sorted(frame["cluster"].astype(str).unique())):
            sub = frame[frame["cluster"].astype(str) == group]
            sizes = 45 + 650 * (sub["stock_volatility"] / max(coords["stock_volatility"].max(), 1e-12))
            ax.scatter(sub["x"], sub["y"], s=sizes, alpha=0.75, label=group, color=cmap(idx % 10), edgecolor="white", linewidth=0.5)
        for _, r in frame[frame["ticker"].isin(LABEL_TICKERS)].iterrows():
            ax.text(r["x"], r["y"], r["ticker"], fontsize=7, ha="center", va="center")
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)
        ax.set_title(f"Market structure window ending {row['window_end']}")
        ax.set_xlabel("MDS dimension 1")
        ax.set_ylabel("MDS dimension 2")
        ax.legend(title="cluster", fontsize=7, loc="upper right")
        path = tmp_dir / f"frame_{int(window_id):04d}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        frames.append(imageio.imread(path))
    gif_path = ANIMATIONS_DIR / "market_structure.gif"
    imageio.mimsave(gif_path, frames, duration=0.18)
    result = {"gif": str(gif_path), "mp4": "", "mp4_status": "not attempted"}
    if shutil.which("ffmpeg"):
        mp4_path = ANIMATIONS_DIR / "market_structure.mp4"
        try:
            imageio.mimsave(mp4_path, frames, fps=6)
            result.update({"mp4": str(mp4_path), "mp4_status": "generated"})
        except Exception as exc:
            result["mp4_status"] = f"failed: {exc}"
    else:
        result["mp4_status"] = "ffmpeg not found"
    for file in tmp_dir.glob("*.png"):
        file.unlink(missing_ok=True)
    tmp_dir.rmdir()
    return result

