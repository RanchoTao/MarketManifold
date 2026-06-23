from __future__ import annotations

import zipfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from src.data_loader import inspect_archive, normalize_ticker, parse_member_dataframe
from src.manifold import align_to_previous, correlation_distance
from src.preprocessing import coverage_ratio, deduplicate_by_date_ticker, filter_positive_prices, forward_fill_short_gaps
from src.rolling_windows import rolling_window_slices
from src.visualization import create_animation


def make_zip(path: Path, member: str, text: str) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, text)
    return path


def test_zip_member_format_check(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "tiny.zip",
        "data/daily/us/nasdaq stocks/1/aapl.us.txt",
        "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
        "AAPL.US,D,20200102,000000,1,2,1,1.5,100,0\n",
    )
    result = inspect_archive(archive)
    assert result["member_count"] == 1
    assert ".txt" in result["file_extensions"]
    assert result["detected_delimiters"]["data/daily/us/nasdaq stocks/1/aapl.us.txt"] == ","


def test_ticker_normalization() -> None:
    assert normalize_ticker("aapl.us") == "AAPL"
    assert normalize_ticker("BRK-B.US") == "BRK.B"
    assert normalize_ticker("brkb") == "BRK.B"


def test_date_parsing_with_header(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "header.zip",
        "aapl.us.txt",
        "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n"
        "AAPL.US,D,20200102,000000,1,2,1,1.5,100,0\n",
    )
    with zipfile.ZipFile(archive) as zf:
        df, stats = parse_member_dataframe(zf, "aapl.us.txt", "AAPL", "Tech")
    assert stats["invalid_date_count"] == 0
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-02")


def test_no_header_parsing(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "noheader.zip", "msft.us.txt", "MSFT.US,D,20200102,000000,1,2,1,1.5,100,0\n")
    with zipfile.ZipFile(archive) as zf:
        df, stats = parse_member_dataframe(zf, "msft.us.txt", "MSFT", "Tech")
    assert stats["error"] == ""
    assert df["ticker"].iloc[0] == "MSFT"


def test_non_positive_price_filter() -> None:
    df = pd.DataFrame({"close": [10, 0, -1, None, 5]})
    assert filter_positive_prices(df)["close"].tolist() == [10, 5]


def test_duplicate_record_keeps_last() -> None:
    df = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02")],
        "ticker": ["AAPL", "AAPL"],
        "close": [1, 2],
    })
    out = deduplicate_by_date_ticker(df)
    assert len(out) == 1
    assert out["close"].iloc[0] == 2


def test_coverage_filter_ratio() -> None:
    assert coverage_ratio(90, 100) == 0.9
    assert coverage_ratio(0, 0) == 0.0


def test_short_gap_forward_fill() -> None:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    s = pd.Series([1.0, np.nan, np.nan, np.nan], index=idx)
    out = forward_fill_short_gaps(s, pd.Timestamp("2020-01-01"), limit=2)
    assert out.iloc[1] == 1.0
    assert out.iloc[2] == 1.0
    assert np.isnan(out.iloc[3])


def test_no_backward_fill_before_listing() -> None:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    s = pd.Series([np.nan, 2.0, np.nan], index=idx)
    out = forward_fill_short_gaps(s, pd.Timestamp("2020-01-02"), limit=2)
    assert np.isnan(out.iloc[0])
    assert out.iloc[2] == 2.0


def test_correlation_distance_matrix_validity() -> None:
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"])
    dist = correlation_distance(corr)
    assert np.allclose(np.diag(dist), 0)
    assert np.allclose(dist.values, dist.values.T)
    assert np.isclose(dist.loc["A", "B"], 1.0)


def test_procrustes_alignment() -> None:
    prev = pd.DataFrame({"x": [0.0, 1.0, 0.0], "y": [0.0, 0.0, 1.0]}, index=["A", "B", "C"])
    curr = pd.DataFrame({"x_raw": [0.0, 0.0, -1.0], "y_raw": [0.0, 1.0, 0.0]}, index=["A", "B", "C"])
    aligned = align_to_previous(prev, curr)
    assert aligned.shape == (3, 2)
    assert np.linalg.norm(aligned.loc["A"] - prev.loc["A"]) < 1.0


def mean_pairwise(values: np.ndarray) -> float:
    distances = []
    for idx in range(len(values)):
        diff = values[idx + 1:] - values[idx]
        distances.extend(np.sqrt(np.sum(diff * diff, axis=1)))
    return float(np.mean(distances))


def test_alignment_preserves_pairwise_distance_scale() -> None:
    prev = pd.DataFrame({"x": [0.0, 2.0, 0.0, 2.0], "y": [0.0, 0.0, 1.0, 1.0]}, index=list("ABCD"))
    curr = pd.DataFrame({"x_raw": [5.0, 5.0, 4.0, 4.0], "y_raw": [8.0, 6.0, 8.0, 6.0]}, index=list("ABCD"))
    aligned = align_to_previous(prev, curr)
    ratio = mean_pairwise(aligned[["x", "y"]].to_numpy()) / mean_pairwise(curr[["x_raw", "y_raw"]].to_numpy())
    assert 0.999 < ratio < 1.001


def test_alignment_norm_does_not_decay_across_windows() -> None:
    base = pd.DataFrame({"x_raw": [0.0, 2.0, 0.0, 2.0], "y_raw": [0.0, 0.0, 1.0, 1.0]}, index=list("ABCD"))
    previous = align_to_previous(None, base)
    norms = []
    for step in range(1, 80):
        angle = step * 0.2
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shifted = base[["x_raw", "y_raw"]].to_numpy() @ rotation + np.array([step * 0.05, -step * 0.03])
        current = pd.DataFrame(shifted, index=base.index, columns=["x_raw", "y_raw"])
        previous = align_to_previous(previous, current)
        centered = previous[["x", "y"]].to_numpy() - previous[["x", "y"]].to_numpy().mean(axis=0)
        norms.append(np.sqrt(np.sum(centered * centered) / len(centered)))
    assert min(norms) > 0.5
    assert norms[-1] / norms[0] > 0.99


def test_aligned_last_window_span_not_machine_precision() -> None:
    base = pd.DataFrame({"x_raw": [0.0, 1.0, 0.0], "y_raw": [0.0, 0.0, 1.0]}, index=list("ABC"))
    aligned = align_to_previous(None, base)
    for _ in range(20):
        current = pd.DataFrame({"x_raw": [0.1, 1.1, 0.1], "y_raw": [-0.2, -0.2, 0.8]}, index=list("ABC"))
        aligned = align_to_previous(aligned, current)
    span = aligned["x"].max() - aligned["x"].min() + aligned["y"].max() - aligned["y"].min()
    assert span > 1e-6


def test_animation_frames_have_position_variance(tmp_path: Path, monkeypatch) -> None:
    import src.visualization as visualization

    monkeypatch.setattr(visualization, "ANIMATIONS_DIR", tmp_path)
    coords = pd.DataFrame([
        {"window_id": 0, "ticker": "A", "x": 0.0, "y": 0.0, "cluster": 0, "stock_volatility": 0.2},
        {"window_id": 0, "ticker": "B", "x": 1.0, "y": 0.0, "cluster": 1, "stock_volatility": 0.3},
        {"window_id": 0, "ticker": "C", "x": 0.0, "y": 1.0, "cluster": 1, "stock_volatility": 0.4},
        {"window_id": 1, "ticker": "A", "x": 0.2, "y": 0.1, "cluster": 0, "stock_volatility": 0.2},
        {"window_id": 1, "ticker": "B", "x": 1.2, "y": 0.2, "cluster": 1, "stock_volatility": 0.3},
        {"window_id": 1, "ticker": "C", "x": 0.3, "y": 1.1, "cluster": 1, "stock_volatility": 0.4},
    ])
    metrics = pd.DataFrame([
        {"window_id": 0, "window_end": "2020-01-01"},
        {"window_id": 1, "window_end": "2020-01-02"},
    ])
    result = create_animation(coords, metrics)
    reader = imageio.get_reader(result["gif"])
    first = np.asarray(reader.get_data(0), dtype=np.int16)[..., :3]
    second = np.asarray(reader.get_data(1), dtype=np.int16)[..., :3]
    reader.close()
    assert first.var() > 0
    assert second.var() > 0
    assert np.mean(np.abs(first - second)) > 0.1


def test_rolling_window_count() -> None:
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    assert len(rolling_window_slices(idx, window=20, step=10)) == 9


def test_formal_pipeline_uses_real_data_only() -> None:
    root = Path(__file__).resolve().parents[1]
    checked = "\n".join((root / path).read_text(encoding="utf-8").lower() for path in ["run_pipeline.py", "src/pipeline.py", "src/preprocessing.py"])
    assert ("sim" + "ulate") not in checked
    assert ("synt" + "hetic") not in checked
