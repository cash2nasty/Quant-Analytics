import datetime as dt
from typing import Dict, Optional

import pandas as pd

from data.session_reference import get_session_windows_for_date
from storage.history_manager import SessionStats


def slice_session(df: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    return df.loc[mask].copy()


def compute_session_stats(
    df_today: pd.DataFrame,
    trading_date: Optional[dt.date] = None,
) -> Dict[str, SessionStats]:
    """
    Build Asia / London / US session stats for the selected day.
    """
    if df_today.empty:
        return {}

    date = trading_date or df_today["timestamp"].dt.date.iloc[0]
    windows = get_session_windows_for_date(date)

    sessions: Dict[str, SessionStats] = {}
    for name, win in windows.items():
        sdf = slice_session(df_today, win["start"], win["end"])
        if sdf.empty:
            continue
        high = float(sdf["high"].max())
        low = float(sdf["low"].min())
        open_ = float(sdf["open"].iloc[0])
        close = float(sdf["close"].iloc[-1])
        rng = high - low
        vol = float(sdf["volume"].sum())
        sessions[name] = SessionStats(
            name=name,
            high=high,
            low=low,
            open=open_,
            close=close,
            range=rng,
            volume=vol,
        )
    return sessions