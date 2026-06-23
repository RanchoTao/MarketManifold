from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageSequence
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import agglomerative_labels
from src.config import PROCESSED_DIR, RunConfig
from src.manifold import align_to_previous, correlation_distance, mds_coordinates
from src.metrics import window_metrics
from src.rolling_windows import rolling_window_slices


CACHE_VERSION = 1
OUTPUT_GIF = PROJECT_ROOT / "outputs" / "market_structure_daily_full_slow.gif"
OUTPUT_MP4 = PROJECT_ROOT / "outputs" / "market_structure_daily_full_slow.mp4"
OUTPUT_PREVIEW = PROJECT_ROOT / "outputs" / "market_structure_daily_full_slow_preview.png"
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "daily_full_market_structure.pkl"
FAILURE_LOG = PROJECT_ROOT / "outputs" / "logs" / "daily_full_animation_failures.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "daily_full_animation_summary.json"

LABEL_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "BRK.B"]
CLUSTER_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#999999", "#F0E442"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a full-range daily-step market structure animation.")
    parser.add_argument("--window", type=int, default=90, help="Rolling window length in trading days.")
    parser.add_argument("--fps", type=int, default=8, help="Playback frames per second.")
    parser.add_argument("--clusters", type=int, default=6, help="Number of agglomerative clusters.")
    parser.add_argument("--recompute", action="store_true", help="Ignore the cached daily analysis.")
    return parser.parse_args()


def cache_metadata(returns_path: Path, returns: pd.DataFrame, args: argparse.Namespace) -> dict:
    stat = returns_path.stat()
    return {
        "version": CACHE_VERSION,
        "returns_size": stat.st_size,
        "returns_mtime_ns": stat.st_mtime_ns,
        "window": args.window,
        "step": 1,
        "clusters": args.clusters,
        "random_seed": RunConfig().random_seed,
        "tickers": list(returns.columns),
        "first_return_date": returns.index.min().date().isoformat(),
        "last_return_date": returns.index.max().date().isoformat(),
    }


def match_cluster_labels(previous: np.ndarray | None, current: np.ndarray) -> np.ndarray:
    """Map current labels to previous colors by maximizing ticker overlap."""
    if previous is None:
        return current.astype(int)
    previous_ids = sorted(np.unique(previous))
    current_ids = sorted(np.unique(current))
    overlap = np.zeros((len(previous_ids), len(current_ids)), dtype=int)
    for row, previous_id in enumerate(previous_ids):
        for col, current_id in enumerate(current_ids):
            overlap[row, col] = int(np.sum((previous == previous_id) & (current == current_id)))
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {current_ids[col]: previous_ids[row] for row, col in zip(rows, cols)}
    unused = [label for label in range(max(len(previous_ids), len(current_ids)) + len(current_ids)) if label not in mapping.values()]
    for current_id in current_ids:
        if current_id not in mapping:
            mapping[current_id] = unused.pop(0)
    return np.array([mapping[label] for label in current], dtype=int)


def load_inputs() -> tuple[pd.DataFrame, dict[str, str], Path]:
    returns_path = PROCESSED_DIR / "log_returns.csv"
    prices_path = PROCESSED_DIR / "prices_long.csv"
    if not returns_path.exists() or not prices_path.exists():
        raise FileNotFoundError("Processed data is missing. Run: python scripts/prepare_data.py")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    prices = pd.read_csv(prices_path, usecols=["ticker", "sector"])
    sectors = prices.drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()
    return returns, sectors, returns_path


def compute_daily_analysis(
    returns: pd.DataFrame,
    sectors: dict[str, str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    windows = rolling_window_slices(returns.index, window=args.window, step=1)
    tickers = list(returns.columns)
    n_clusters = min(args.clusters, max(2, len(tickers) - 1))
    previous_aligned = None
    previous_labels = None
    coordinate_rows: list[dict] = []
    metric_rows: list[dict] = []
    failures: list[dict] = []
    seed = RunConfig().random_seed

    for frame_id, (start, end, start_date, end_date) in enumerate(tqdm(windows, desc="Computing daily windows")):
        try:
            window_returns = returns.iloc[start:end].copy()
            corr = window_returns.corr().fillna(0.0)
            distance = correlation_distance(corr)
            raw_coordinates = mds_coordinates(distance, random_state=seed)
            aligned = align_to_previous(previous_aligned, raw_coordinates)
            raw_labels = agglomerative_labels(distance, n_clusters=n_clusters)
            stable_labels = match_cluster_labels(previous_labels, raw_labels)
            metrics = window_metrics(window_returns, corr, distance, aligned)
            stock_volatility = window_returns.std(ddof=1) * np.sqrt(252)

            metric_rows.append({
                "frame_id": frame_id,
                "window_start": start_date.date().isoformat(),
                "window_end": end_date.date().isoformat(),
                "mean_correlation": metrics["mean_correlation"],
                "market_volatility": metrics["market_volatility"],
                "pca_first_component_ratio": metrics["pca_first_component_ratio"],
            })
            for ticker_index, ticker in enumerate(tickers):
                coordinate_rows.append({
                    "frame_id": frame_id,
                    "window_start": start_date.date().isoformat(),
                    "window_end": end_date.date().isoformat(),
                    "ticker": ticker,
                    "sector": sectors.get(ticker, "Unknown"),
                    "x": float(aligned.loc[ticker, "x"]),
                    "y": float(aligned.loc[ticker, "y"]),
                    "cluster": int(stable_labels[ticker_index]),
                    "stock_volatility": float(stock_volatility[ticker]),
                })
            previous_aligned = aligned
            previous_labels = stable_labels
        except Exception as exc:
            failures.append({"frame_id": frame_id, "window_end": end_date.date().isoformat(), "error": str(exc)})

    coordinates = pd.DataFrame(coordinate_rows).sort_values(["frame_id", "ticker"])
    metrics = pd.DataFrame(metric_rows).sort_values("frame_id")
    if metrics.empty:
        raise RuntimeError("Every daily rolling window failed; no animation can be rendered.")
    return coordinates, metrics, failures


def load_or_compute_cache(
    returns: pd.DataFrame,
    sectors: dict[str, str],
    returns_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], bool]:
    metadata = cache_metadata(returns_path, returns, args)
    if CACHE_PATH.exists() and not args.recompute:
        try:
            cached = pd.read_pickle(CACHE_PATH)
            if cached.get("metadata") == metadata:
                print(f"Using cache: {CACHE_PATH}")
                return cached["coordinates"], cached["metrics"], cached.get("failures", []), True
            print("Cache metadata changed; recomputing daily windows.")
        except Exception as exc:
            print(f"Cache could not be read ({exc}); recomputing daily windows.")

    coordinates, metrics, failures = compute_daily_analysis(returns, sectors, args)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({
        "metadata": metadata,
        "coordinates": coordinates,
        "metrics": metrics,
        "failures": failures,
    }, CACHE_PATH)
    print(f"Wrote cache: {CACHE_PATH}")
    return coordinates, metrics, failures, False


def global_axis_limits(coordinates: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, xmax = float(coordinates["x"].min()), float(coordinates["x"].max())
    ymin, ymax = float(coordinates["y"].min()), float(coordinates["y"].max())
    xpad = max((xmax - xmin) * 0.10, 0.05)
    ypad = max((ymax - ymin) * 0.10, 0.05)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)


def frame_repeat_count(frame_id: int, last_frame_id: int, key_frames: set[int], fps: int) -> int:
    if frame_id == last_frame_id:
        return max(1, round(2.0 * fps))
    if frame_id == 0:
        return max(1, round(1.5 * fps))
    if frame_id in key_frames:
        return 1 + round(0.9 * fps)
    return 1


def gif_frame_delays_centiseconds(frame_count: int, key_frames: set[int]) -> list[int]:
    """Return GIF delays in 1/100 seconds; 12/13 alternation averages 8 fps."""
    delays = [12 if frame_id % 2 == 0 else 13 for frame_id in range(frame_count)]
    for frame_id in key_frames:
        if 0 <= frame_id < frame_count:
            delays[frame_id] = 100
    if frame_count:
        delays[0] = 150
        delays[-1] = 200
    return delays


def set_gif_frame_delays(path: Path, delays_centiseconds: list[int]) -> int:
    """Patch GIF Graphic Control Extensions without loading all frames into memory."""
    data = bytearray(path.read_bytes())
    if bytes(data[:6]) not in {b"GIF87a", b"GIF89a"}:
        raise ValueError(f"Not a GIF file: {path}")
    position = 13
    packed = data[10]
    if packed & 0x80:
        position += 3 * (2 ** ((packed & 0x07) + 1))
    frames: list[tuple[int, int | None]] = []
    pending_delay_position: int | None = None
    while position < len(data):
        block = data[position]
        if block == 0x3B:
            break
        if block == 0x21:
            label = data[position + 1]
            if label == 0xF9:
                if data[position + 2] != 4:
                    raise ValueError("Unexpected GIF graphic control extension size.")
                pending_delay_position = position + 4
                position += 8
                continue
            position += 2
            while True:
                size = data[position]
                position += 1
                if size == 0:
                    break
                position += size
            continue
        if block == 0x2C:
            image_position = position
            frames.append((image_position, pending_delay_position))
            pending_delay_position = None
            descriptor_packed = data[position + 9]
            position += 10
            if descriptor_packed & 0x80:
                position += 3 * (2 ** ((descriptor_packed & 0x07) + 1))
            position += 1  # LZW minimum code size
            while True:
                size = data[position]
                position += 1
                if size == 0:
                    break
                position += size
            continue
        raise ValueError(f"Unexpected GIF block 0x{block:02x} at byte {position}.")

    if len(frames) != len(delays_centiseconds):
        raise ValueError(f"GIF frame count {len(frames)} does not match delay count {len(delays_centiseconds)}.")
    insertions: list[tuple[int, bytes]] = []
    for (image_position, delay_position), delay in zip(frames, delays_centiseconds):
        delay = max(1, min(int(delay), 65535))
        low, high = delay & 0xFF, (delay >> 8) & 0xFF
        if delay_position is None:
            insertions.append((image_position, bytes([0x21, 0xF9, 0x04, 0x00, low, high, 0x00, 0x00])))
        else:
            data[delay_position] = low
            data[delay_position + 1] = high
    for image_position, extension in sorted(insertions, reverse=True):
        data[image_position:image_position] = extension
    path.write_bytes(data)
    return len(frames)


def render_outputs(
    coordinates: pd.DataFrame,
    metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    xlim, ylim = global_axis_limits(coordinates)
    vol_low = float(coordinates["stock_volatility"].quantile(0.05))
    vol_high = float(coordinates["stock_volatility"].quantile(0.95))
    vol_span = max(vol_high - vol_low, 1e-12)
    cluster_ids = sorted(int(value) for value in coordinates["cluster"].unique())
    color_lookup = {cluster: CLUSTER_COLORS[index % len(CLUSTER_COLORS)] for index, cluster in enumerate(cluster_ids)}
    key_frames = {
        int(metrics.loc[metrics["market_volatility"].idxmax(), "frame_id"]),
        int(metrics.loc[metrics["mean_correlation"].idxmax(), "frame_id"]),
        int(metrics.loc[metrics["mean_correlation"].idxmin(), "frame_id"]),
    }
    last_frame_id = int(metrics["frame_id"].max())

    fig = plt.figure(figsize=(12, 7.5), dpi=80, facecolor="white")
    ax = fig.add_axes([0.075, 0.13, 0.69, 0.72])
    ax.set_facecolor("white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("MDS Dimension 1 (Procrustes Aligned)", fontsize=11)
    ax.set_ylabel("MDS Dimension 2 (Procrustes Aligned)", fontsize=11)
    ax.grid(color="#E6E6E6", linewidth=0.7, alpha=0.8)
    ax.axhline(0, color="#BEBEBE", linewidth=0.7)
    ax.axvline(0, color="#BEBEBE", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#8A8A8A")
    # A one-point seed keeps Matplotlib's marker path initialized for later updates.
    scatter = ax.scatter([0], [0], s=[1], c=["white"], alpha=0.82, edgecolors="white", linewidths=0.7)
    labels = {
        ticker: ax.annotate(
            ticker,
            xy=(0, 0),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.68},
        )
        for ticker in LABEL_TICKERS
        if ticker in set(coordinates["ticker"])
    }
    main_title = fig.text(0.5, 0.955, "Market Structure Evolution", ha="center", va="top", fontsize=20, weight="bold")
    subtitle = fig.text(0.5, 0.905, "", ha="center", va="top", fontsize=12, color="#444444")
    info_text = fig.text(
        0.795,
        0.78,
        "",
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#F7F7F7", "edgecolor": "#BDBDBD", "linewidth": 0.8},
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_lookup[cluster], markeredgecolor="white", markersize=9, label=f"Cluster {cluster}")
        for cluster in cluster_ids
    ]
    fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(0.79, 0.42), frameon=False, title="Matched Clusters")
    fig.text(0.5, 0.035, "Each frame is estimated from the previous 90 trading days.", ha="center", fontsize=9, color="#777777")

    gif_writer = imageio.get_writer(
        OUTPUT_GIF,
        mode="I",
        # The Pillow GIF backend expects frame duration in milliseconds.
        duration=1000.0 / args.fps,
        loop=0,
        palettesize=128,
        subrectangles=True,
    )
    mp4_writer = None
    mp4_status = "not generated"
    try:
        mp4_writer = imageio.get_writer(
            OUTPUT_MP4,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=2,
            pixelformat="yuv420p",
        )
        mp4_status = "generated"
    except Exception as exc:
        print(f"MP4 writer unavailable: {exc}")
        OUTPUT_MP4.unlink(missing_ok=True)
        mp4_writer = None
        mp4_status = f"failed to initialize: {exc}"

    encoded_frame_count = 0
    total_repeats = 0
    try:
        for _, metric in tqdm(metrics.iterrows(), total=len(metrics), desc="Rendering animation"):
            frame_id = int(metric["frame_id"])
            frame = coordinates[coordinates["frame_id"] == frame_id].sort_values("ticker")
            offsets = frame[["x", "y"]].to_numpy()
            normalized_volatility = np.clip((frame["stock_volatility"].to_numpy() - vol_low) / vol_span, 0, 1)
            sizes = 65 + 235 * normalized_volatility
            colors = [color_lookup[int(cluster)] for cluster in frame["cluster"]]
            scatter.set_offsets(offsets)
            scatter.set_sizes(sizes)
            scatter.set_facecolors(colors)
            for ticker, annotation in labels.items():
                row = frame[frame["ticker"] == ticker]
                annotation.set_visible(not row.empty)
                if not row.empty:
                    annotation.xy = (float(row.iloc[0]["x"]), float(row.iloc[0]["y"]))

            end_date = str(metric["window_end"])
            subtitle.set_text(f"90-Trading-Day Rolling Window | Window Ending: {end_date}")
            info_text.set_text(
                f"Date: {end_date}\n"
                f"Mean Correlation: {metric['mean_correlation']:.3f}\n"
                f"Annualized Market\nVolatility: {metric['market_volatility']:.3f}\n"
                f"PC1 Explained Variance: {metric['pca_first_component_ratio']:.3f}"
            )
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgb = np.ascontiguousarray(rgba[:, :, :3])
            repeat_count = frame_repeat_count(frame_id, last_frame_id, key_frames, args.fps)
            gif_writer.append_data(rgb)
            for _ in range(repeat_count):
                if mp4_writer is not None:
                    try:
                        mp4_writer.append_data(rgb)
                    except Exception as exc:
                        print(f"MP4 encoding stopped: {exc}")
                        mp4_status = f"failed while encoding: {exc}"
                        mp4_writer.close()
                        mp4_writer = None
                        OUTPUT_MP4.unlink(missing_ok=True)
                encoded_frame_count += 1
            total_repeats += repeat_count
            if frame_id == last_frame_id:
                fig.savefig(OUTPUT_PREVIEW, dpi=120, facecolor="white")
    finally:
        gif_writer.close()
        if mp4_writer is not None:
            mp4_writer.close()
        plt.close(fig)

    gif_delays = gif_frame_delays_centiseconds(len(metrics), key_frames)
    set_gif_frame_delays(OUTPUT_GIF, gif_delays)
    actual_gif_duration = 0.0
    gif_encoded_images = 0
    with Image.open(OUTPUT_GIF) as gif:
        for image in ImageSequence.Iterator(gif):
            actual_gif_duration += image.info.get("duration", 0) / 1000.0
            gif_encoded_images += 1

    return {
        "gif_path": str(OUTPUT_GIF.resolve()),
        "mp4_path": str(OUTPUT_MP4.resolve()) if OUTPUT_MP4.exists() else "",
        "preview_path": str(OUTPUT_PREVIEW.resolve()),
        "mp4_status": mp4_status,
        "fps": args.fps,
        "unique_frame_count": int(len(metrics)),
        "submitted_frame_count_with_holds": int(encoded_frame_count),
        "gif_encoded_image_count": int(gif_encoded_images),
        "intended_duration_seconds": float(total_repeats / args.fps),
        "actual_gif_duration_seconds": float(actual_gif_duration),
        "key_frame_ids": sorted(key_frames),
        "global_xlim": [float(xlim[0]), float(xlim[1])],
        "global_ylim": [float(ylim[0]), float(ylim[1])],
        "position_smoothing": "none",
    }


def main() -> None:
    args = parse_args()
    if args.window != 90:
        print(f"Warning: requested window={args.window}; the project default is 90 trading days.")
    returns, sectors, returns_path = load_inputs()
    coordinates, metrics, failures, used_cache = load_or_compute_cache(returns, sectors, returns_path, args)
    FAILURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["frame_id", "window_end", "error"]).to_csv(FAILURE_LOG, index=False)
    rendering = render_outputs(coordinates, metrics, args)
    summary = {
        "data_first_return_date": returns.index.min().date().isoformat(),
        "data_last_return_date": returns.index.max().date().isoformat(),
        "first_window_end": metrics["window_end"].iloc[0],
        "last_window_end": metrics["window_end"].iloc[-1],
        "trading_day_count": int(len(returns)),
        "rolling_window": args.window,
        "step": 1,
        "valid_frame_count": int(len(metrics)),
        "failed_frame_count": int(len(failures)),
        "cache_path": str(CACHE_PATH.resolve()),
        "used_cache": used_cache,
        **rendering,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
