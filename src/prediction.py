from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURES = [
    "mean_correlation",
    "market_volatility",
    "pca_first_component_ratio",
    "cross_sectional_dispersion",
    "adjacent_window_ari",
]


def run_prediction(metrics: pd.DataFrame, target: str = "market_volatility") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = metrics.sort_values("window_id").copy()
    for col in FEATURES:
        df[f"lag_{col}"] = df[col].shift(1)
    df["target"] = df[target].shift(-1)
    model_df = df.dropna(subset=[f"lag_{c}" for c in FEATURES] + ["target"]).copy()
    if len(model_df) < 12:
        msg = pd.DataFrame([{
            "model": "skipped",
            "target": target,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "note": "too few rolling windows for a time-ordered prediction experiment",
        }])
        return msg, pd.DataFrame()
    split = max(1, int(len(model_df) * 0.7))
    train = model_df.iloc[:split]
    test = model_df.iloc[split:]
    x_cols = [f"lag_{c}" for c in FEATURES]
    rows = []
    predictions = pd.DataFrame({
        "window_id": test["window_id"].values,
        "window_end": test["window_end"].values,
        "actual": test["target"].values,
    })
    naive_pred = np.repeat(train["target"].iloc[-1], len(test))
    candidates = {
        "naive_last_train": naive_pred,
        "LinearRegression": LinearRegression().fit(train[x_cols], train["target"]).predict(test[x_cols]),
    }
    if len(train) >= 20:
        candidates["RandomForestRegressor"] = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2).fit(train[x_cols], train["target"]).predict(test[x_cols])
    for name, pred in candidates.items():
        rows.append({
            "model": name,
            "target": target,
            "mae": float(mean_absolute_error(test["target"], pred)),
            "rmse": float(np.sqrt(mean_squared_error(test["target"], pred))),
            "r2": float(r2_score(test["target"], pred)) if len(test) > 1 else np.nan,
            "note": "",
        })
        predictions[name] = pred
    return pd.DataFrame(rows), predictions
