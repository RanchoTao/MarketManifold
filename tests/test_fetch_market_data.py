import math
import pandas as pd

from src.fetch_market_data import yahoo_ticker, normalize_yfinance_frame, clean_panel


def test_yahoo_ticker_mapping():
    assert yahoo_ticker("BRK.B") == "BRK-B"
    assert yahoo_ticker("BF.B") == "BF-B"
    assert yahoo_ticker("aapl") == "AAPL"


def test_multiindex_conversion():
    dates = pd.date_range("2024-01-02", periods=2)
    cols = pd.MultiIndex.from_product([["Close", "Volume"], ["AAPL", "MSFT"]])
    df = pd.DataFrame([[10, 20, 100, 200], [11, 21, 110, 210]], index=dates, columns=cols)
    out = normalize_yfinance_frame(df, {"AAPL": "AAPL", "MSFT": "MSFT"})
    assert set(out["ticker"]) == {"AAPL", "MSFT"}
    assert set(["date", "ticker", "close", "volume"]).issubset(out.columns)


def test_single_ticker_conversion():
    dates = pd.date_range("2024-01-02", periods=2)
    df = pd.DataFrame({"Close": [10.0, 11.0], "Volume": [100, 110]}, index=dates)
    out = normalize_yfinance_frame(df, {"AAPL": "AAPL"})
    assert out["close"].tolist() == [10.0, 11.0]
    assert out["ticker"].unique().tolist() == ["AAPL"]


def test_clean_panel_coverage_fill_and_consistency(monkeypatch):
    import src.fetch_market_data as fmd
    sectors = {"Tech": ["AAA", "BBB"], "Finance": ["CCC"]}
    monkeypatch.setattr(fmd, "ticker_sector_map", lambda: {"AAA": "Tech", "BBB": "Tech", "CCC": "Finance"})
    rows = []
    dates = pd.date_range("2024-01-01", periods=6, freq="D").strftime("%Y-%m-%d").tolist()
    # AAA has one small gap that should be filled.
    for i, d in enumerate(dates):
        if i != 2:
            rows.append({"date": d, "ticker": "AAA", "close": 10 + i})
    # duplicate keeps last valid record; one gap filled.
    for i, d in enumerate(dates):
        if i != 4:
            rows.append({"date": d, "ticker": "BBB", "close": 20 + i})
    rows.append({"date": dates[0], "ticker": "BBB", "close": 99})
    # CCC has pre-history/long missing and below coverage, so dropped.
    rows.append({"date": dates[-1], "ticker": "CCC", "close": 30})
    raw = pd.DataFrame(rows)
    final, report, stats = fmd.clean_panel(raw, min_coverage=0.80, max_ffill=2)
    assert stats["duplicate_rows"] > 0
    assert set(final["ticker"].unique()) == {"AAA", "BBB"}
    assert final.duplicated(["date", "ticker"]).sum() == 0
    assert final["close"].map(lambda x: math.isfinite(x) and x > 0).all()
    counts = final.groupby("ticker")["date"].nunique()
    assert counts.nunique() == 1
    assert report.loc[report["ticker"] == "AAA", "filled_days"].iloc[0] == 1
    assert not report.loc[report["ticker"] == "CCC", "kept"].iloc[0]
