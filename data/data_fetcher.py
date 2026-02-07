import datetime as dt
from typing import Tuple, Union

import pandas as pd


def _map_to_yahoo_candidates(symbol: str) -> list:
    """Return a list of candidate Yahoo tickers to try for a given symbol.

    E.g., map `NQH26` -> `NQ=F` as a continuous futures proxy.
    """
    candidates = [symbol]
    s = symbol.strip().upper()
    if s.startswith("NQ") and "=F" not in s:
        candidates.append("NQ=F")
    # add more mapping rules here as needed
    return candidates


def fetch_intraday_ohlcv(
    symbol: str,
    lookback_days: Union[int, dt.date, Tuple[dt.date, dt.date]] = 5,
) -> Tuple[pd.DataFrame, str]:
    """Fetch intraday OHLCV using yfinance.

    This function only returns real market data. If `yfinance` is not
    installed, or if the provider returns no intraday data for the symbol
    / period requested, an empty DataFrame with the standard columns is
    returned (no simulated data).

    - When `lookback_days` is an int, it is treated as a lookback in days.
    - When `lookback_days` is a date, the function fetches that calendar day.
    - When `lookback_days` is a (start_date, end_date) tuple, it fetches that range.
    """
    try:
        import yfinance as yf  # type: ignore

        start = end = None
        period = None
        if isinstance(lookback_days, tuple) and len(lookback_days) == 2:
            start_date, end_date = lookback_days
            start = dt.datetime.combine(start_date, dt.time(0, 0))
            end = dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time(0, 0))
        elif isinstance(lookback_days, dt.date) and not isinstance(lookback_days, dt.datetime):
            start = dt.datetime.combine(lookback_days, dt.time(0, 0))
            end = start + dt.timedelta(days=1)
        else:
            period = f"{max(1, int(lookback_days))}d"

        candidates = _map_to_yahoo_candidates(symbol)
        for candidate in candidates:
            if start and end:
                raw = yf.download(
                    tickers=candidate,
                    start=start,
                    end=end,
                    interval="5m",
                    progress=False,
                    threads=False,
                )
            else:
                raw = yf.download(
                    tickers=candidate,
                    period=period,
                    interval="5m",
                    progress=False,
                    threads=False,
                )
            if raw is None or raw.empty:
                continue

            # yfinance may return a DataFrame with MultiIndex columns when
            # the ticker is included as the second level (e.g., ('Close', 'NQ=F')).
            # Normalize into a flat-frame with columns: timestamp, open, high, low, close, volume.
            if isinstance(raw.columns, pd.MultiIndex):
                # Try to pick columns where second level == candidate; otherwise take first available level-0 column
                cols = {}
                for level0 in ("Open", "High", "Low", "Close", "Volume"):
                    if (level0, candidate) in raw.columns:
                        cols[level0] = raw[(level0, candidate)]
                    else:
                        # fallback: find any column whose first level matches level0
                        matches = [col for col in raw.columns if col[0] == level0]
                        if matches:
                            cols[level0] = raw[matches[0]]
                        else:
                            cols[level0] = pd.Series(pd.NA, index=raw.index)

                df = pd.DataFrame(cols)
                # bring the index (Datetime) into a column
                df = df.reset_index()
                df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
            else:
                df = raw.reset_index()
                if "Datetime" in df.columns:
                    df.rename(columns={"Datetime": "timestamp"}, inplace=True)
                elif "index" in df.columns:
                    df.rename(columns={"index": "timestamp"}, inplace=True)

                # If columns are already named Open/High/Low/Close/Volume, keep them

            # Normalize column names to lowercase expected by the app
            df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
            ts = pd.to_datetime(df["timestamp"])
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
            df["timestamp"] = ts

            # Ensure required columns exist
            for c in ("open", "high", "low", "close"):
                if c not in df.columns:
                    df[c] = pd.NA
            if "volume" not in df.columns:
                df["volume"] = 0

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            return df.sort_values("timestamp").reset_index(drop=True), candidate

        # none of the candidates produced data
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), ""
    except Exception:
        # No simulation: return empty standardized frame so the app shows 'no live data'
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), ""


def filter_date(df: pd.DataFrame, date: dt.date) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    mask = df["timestamp"].dt.date == date
    return df.loc[mask].copy()