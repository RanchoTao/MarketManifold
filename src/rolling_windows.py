from __future__ import annotations

import pandas as pd


def rolling_window_slices(index: pd.Index, window: int = 90, step: int = 10) -> list[tuple[int, int, pd.Timestamp, pd.Timestamp]]:
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if step <= 0:
        raise ValueError("step must be positive")
    slices = []
    for start in range(0, len(index) - window + 1, step):
        end = start + window
        slices.append((start, end, pd.Timestamp(index[start]), pd.Timestamp(index[end - 1])))
    return slices

