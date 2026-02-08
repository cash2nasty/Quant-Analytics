from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from indicators.moving_averages import compute_daily_vwap, compute_weekly_vwap, compute_anchored_vwap
from indicators.volatility import rolling_volatility, classify_volatility, atr_like
from indicators.momentum import roc, trend_strength
from indicators.volume import rvol
from indicators.trend import classify_trend
from indicators.statistics import zscore
from data.news_fetcher import combine_news_signals
from data.event_calendar import load_event_calendar
from engines.patterns import detect_patterns
from storage.history_manager import SessionStats, BiasSummary
from engines.zones import ZoneConfluence, resample_ohlcv


@dataclass
class VWAPPosture:
    daily_above: bool
    weekly_above: bool
    comment: str


def analyze_vwap_posture(df_today: pd.DataFrame) -> VWAPPosture:
    dvwap = compute_daily_vwap(df_today)
    wvwap = compute_weekly_vwap(df_today)
    last_price = df_today["close"].iloc[-1]
    last_dvwap = dvwap.iloc[-1]
    last_wvwap = wvwap.iloc[-1]

    daily_above = last_price > last_dvwap if pd.notna(last_dvwap) else False
    weekly_above = last_price > last_wvwap if pd.notna(last_wvwap) else False

    if daily_above and weekly_above:
        comment = (
            "Price is above both daily and weekly VWAP (strong bullish posture). "
            "Suggestion: If price opens above daily VWAP, then a retracement to daily VWAP is the first check; "
            "if it holds, then continuation can push away from weekly VWAP support. "
            "If price opens below daily VWAP, then expect a deeper pullback toward weekly VWAP."
        )
    elif not daily_above and not weekly_above:
        comment = (
            "Price is below both daily and weekly VWAP (strong bearish posture). "
            "Suggestion: If price opens below daily VWAP, then a retracement up to daily VWAP is the first check; "
            "if it rejects, then continuation can push away from weekly VWAP resistance. "
            "If price opens above daily VWAP, then expect a deeper pullback toward weekly VWAP."
        )
    elif daily_above and not weekly_above:
        comment = (
            "Price is above daily but below weekly VWAP (short-term strength, long-term weakness). "
            "Suggestion: If price opens above daily VWAP, then a retracement to daily VWAP is the first check; "
            "if it holds, then continuation aims for a push through weekly VWAP. "
            "If price opens below daily VWAP, then weekly VWAP remains the ceiling."
        )
    else:
        comment = (
            "Price is below daily but above weekly VWAP (short-term weakness, long-term strength). "
            "Suggestion: If price opens below daily VWAP, then a retracement to weekly VWAP is the first check; "
            "if it holds, then continuation aims for a push through daily VWAP. "
            "If price opens above daily VWAP, then daily VWAP may flip to support."
        )

    return VWAPPosture(
        daily_above=daily_above,
        weekly_above=weekly_above,
        comment=comment,
    )


def _session_direction(sess: SessionStats) -> float:
    return sess.close - sess.open


def _direction_label(value: float, threshold: float = 0.0) -> str:
    if value > threshold:
        return "Bullish"
    if value < -threshold:
        return "Bearish"
    return "Neutral"


def _direction_value(label: str) -> int:
    if label == "Bullish":
        return 1
    if label == "Bearish":
        return -1
    return 0


def _classify_day_type(
    df_today: pd.DataFrame,
    prev_range: float,
    overnight_high: Optional[float],
    overnight_low: Optional[float],
    last_roc: float,
    mom_strength: float,
) -> str:
    if df_today is None or df_today.empty:
        return "Normal"
    if overnight_high is None or overnight_low is None or prev_range <= 0:
        return "Normal"
    overnight_range = float(overnight_high - overnight_low)
    ratio = overnight_range / max(prev_range, 1e-6)
    if ratio < 0.35 and abs(last_roc) < 0.0015:
        return "Chop"
    if ratio > 0.75 and abs(last_roc) > 0.003:
        return "Trend"
    return "Normal"


def _classify_regime(vol_regime: str, last_roc: float, mom_strength: float) -> str:
    if vol_regime == "expanded" and abs(last_roc) > 0.003 and abs(mom_strength) > 0.0:
        return "Trend"
    if vol_regime == "compressed" and abs(last_roc) < 0.0015:
        return "MeanRevert"
    return "Mixed"


def _opening_range_signal(df_today: pd.DataFrame, minutes: int = 60) -> Tuple[str, Optional[float], Optional[float]]:
    if df_today is None or df_today.empty or "timestamp" not in df_today.columns:
        return "n/a", None, None
    date = df_today["timestamp"].iloc[0].date()
    start = pd.Timestamp.combine(date, pd.Timestamp("09:30").time())
    end = start + pd.Timedelta(minutes=minutes)
    or_df = df_today[(df_today["timestamp"] >= start) & (df_today["timestamp"] <= end)]
    if or_df.empty or df_today["timestamp"].iloc[-1] < end:
        return "pending", None, None
    or_high = float(or_df["high"].max())
    or_low = float(or_df["low"].min())
    last_price = float(df_today["close"].iloc[-1])
    tol = max(last_price * 0.0005, 1e-6)
    if last_price > or_high + tol:
        return "Bullish", or_high, or_low
    if last_price < or_low - tol:
        return "Bearish", or_high, or_low
    return "Neutral", or_high, or_low


def _anchor_time_from_extreme(df: pd.DataFrame, column: str, mode: str) -> Optional[pd.Timestamp]:
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    if column not in df.columns:
        return None
    if mode == "max":
        idx = df[column].idxmax()
    else:
        idx = df[column].idxmin()
    if pd.isna(idx):
        return None
    return pd.to_datetime(df.loc[idx, "timestamp"])


def _anchored_vwap_signal(
    df_all: pd.DataFrame,
    anchor_ts: Optional[pd.Timestamp],
    last_price: Optional[float],
    atr_value: Optional[float],
    near_threshold: float = 0.8,
    far_threshold: float = 1.6,
) -> Tuple[str, Optional[float], Optional[float]]:
    if anchor_ts is None or last_price is None or atr_value is None or atr_value <= 0:
        return "Neutral", None, None
    vwap = compute_anchored_vwap(df_all, anchor_ts)
    if vwap is None or vwap.empty:
        return "Neutral", None, None
    last_vwap = float(vwap.iloc[-1])
    distance_units = abs(last_price - last_vwap) / atr_value
    if distance_units > far_threshold:
        return "Far", last_vwap, distance_units
    if last_price > last_vwap and distance_units <= near_threshold:
        return "Bullish", last_vwap, distance_units
    if last_price < last_vwap and distance_units <= near_threshold:
        return "Bearish", last_vwap, distance_units
    return "Neutral", last_vwap, distance_units


def anchored_vwap_anchor_times(
    df_prev: Optional[pd.DataFrame],
    df_today: Optional[pd.DataFrame],
    df_all: Optional[pd.DataFrame],
) -> Dict[str, Optional[pd.Timestamp]]:
    anchors: Dict[str, Optional[pd.Timestamp]] = {
        "Prior Day High": None,
        "Prior Day Low": None,
        "Overnight High": None,
        "Overnight Low": None,
        "US Open 30m High": None,
        "US Open 30m Low": None,
    }

    if df_prev is not None and not df_prev.empty:
        anchors["Prior Day High"] = _anchor_time_from_extreme(df_prev, "high", "max")
        anchors["Prior Day Low"] = _anchor_time_from_extreme(df_prev, "low", "min")

    if df_all is not None and not df_all.empty and df_today is not None and not df_today.empty:
        day = pd.to_datetime(df_today["timestamp"].iloc[-1]).date()
        overnight_start = pd.Timestamp.combine(day - pd.Timedelta(days=1), pd.Timestamp("18:00").time())
        overnight_end = pd.Timestamp.combine(day, pd.Timestamp("09:30").time())
        overnight_df = df_all[
            (df_all["timestamp"] >= overnight_start)
            & (df_all["timestamp"] <= overnight_end)
        ]
        if not overnight_df.empty:
            anchors["Overnight High"] = _anchor_time_from_extreme(overnight_df, "high", "max")
            anchors["Overnight Low"] = _anchor_time_from_extreme(overnight_df, "low", "min")

        or_start = pd.Timestamp.combine(day, pd.Timestamp("09:30").time())
        or_end = or_start + pd.Timedelta(minutes=30)
        or_df = df_all[(df_all["timestamp"] >= or_start) & (df_all["timestamp"] <= or_end)]
        if not or_df.empty:
            anchors["US Open 30m High"] = _anchor_time_from_extreme(or_df, "high", "max")
            anchors["US Open 30m Low"] = _anchor_time_from_extreme(or_df, "low", "min")

    return anchors


def build_anchored_vwap_rows(
    df_all: Optional[pd.DataFrame],
    anchor_times: Dict[str, Optional[pd.Timestamp]],
    last_price: Optional[float],
    atr_value: Optional[float],
    near_threshold: float = 0.8,
    far_threshold: float = 1.6,
) -> List[Dict[str, object]]:
    if df_all is None or df_all.empty or last_price is None or atr_value is None or atr_value <= 0:
        return []

    rows: List[Dict[str, object]] = []
    for name, anchor_ts in anchor_times.items():
        if anchor_ts is None:
            continue
        label, last_vwap, distance_units = _anchored_vwap_signal(
            df_all,
            anchor_ts,
            last_price,
            atr_value,
            near_threshold=near_threshold,
            far_threshold=far_threshold,
        )
        if last_vwap is None or distance_units is None:
            continue

        if label == "Bullish":
            relation = "Above"
            suggestion = "If price retraces, expect support near this anchored VWAP. Likely bullish defense or continuation."
            activity = "Bullish"
        elif label == "Bearish":
            relation = "Below"
            suggestion = "If price retraces up, expect resistance near this anchored VWAP. Likely bearish defense or continuation."
            activity = "Bearish"
        elif label == "Far":
            relation = "Far"
            suggestion = "Price is stretched far from this anchored VWAP; mean-revert risk is higher. Likely snap-back or pause."
            activity = "MeanRevert"
        else:
            relation = "Near"
            suggestion = "Price is near this anchored VWAP; direction is less clear. Likely balance or chop."
            activity = "Neutral"

        rows.append(
            {
                "Anchor": name,
                "Anchor Time": anchor_ts.strftime("%Y-%m-%d %H:%M"),
                "Anchored VWAP": float(last_vwap),
                "Price vs VWAP": relation,
                "Distance (ATR)": float(distance_units),
                "Likely Activity": activity,
                "Suggestion": suggestion,
            }
        )

    return rows


def _premarket_signal(
    df_today: pd.DataFrame,
    amd_bias: str,
    pattern_bias: str,
    htf_bias: str = "Neutral",
    zone_confluence: Optional[ZoneConfluence] = None,
) -> Tuple[str, Dict[str, str], float]:
    if df_today is None or df_today.empty or "timestamp" not in df_today.columns:
        return "Neutral", {}, 0
    date = df_today["timestamp"].iloc[0].date()
    start = pd.Timestamp.combine(date, pd.Timestamp("18:00").time()) - pd.Timedelta(days=1)
    end = pd.Timestamp.combine(date, pd.Timestamp("09:10").time())
    pm = df_today[(df_today["timestamp"] >= start) & (df_today["timestamp"] <= end)]
    if pm.empty:
        return "Neutral", {}, 0

    pm_open = float(pm["open"].iloc[0])
    pm_close = float(pm["close"].iloc[-1])
    pm_high = float(pm["high"].max())
    pm_low = float(pm["low"].min())
    pm_dir = "Bullish" if pm_close > pm_open else "Bearish" if pm_close < pm_open else "Neutral"

    # Gap vs prior close
    gap_dir = "Neutral"
    gap = 0.0
    if len(df_today) > 1:
        gap = pm_open - float(df_today["open"].iloc[0])
    if gap > 0:
        gap_dir = "Bullish"
    elif gap < 0:
        gap_dir = "Bearish"

    # VWAP posture
    vwap = compute_daily_vwap(pm)
    vwap_bias = "Neutral"
    open_vwap_bias = "Neutral"
    if len(vwap):
        last_vwap = float(vwap.iloc[-1])
        vwap_bias = "Bullish" if pm_close > last_vwap else "Bearish" if pm_close < last_vwap else "Neutral"
        open_vwap_bias = "Bullish" if pm_open > last_vwap else "Bearish" if pm_open < last_vwap else "Neutral"

    # Moving averages (premarket)
    sma20 = pm["close"].rolling(20).mean()
    sma50 = pm["close"].rolling(50).mean()
    last_sma20 = float(sma20.iloc[-1]) if len(sma20) and pd.notna(sma20.iloc[-1]) else None
    last_sma50 = float(sma50.iloc[-1]) if len(sma50) and pd.notna(sma50.iloc[-1]) else None
    sma20_bias = "Neutral"
    sma50_bias = "Neutral"
    if last_sma20 is not None:
        sma20_bias = "Bullish" if pm_close > last_sma20 else "Bearish" if pm_close < last_sma20 else "Neutral"
    if last_sma50 is not None:
        sma50_bias = "Bullish" if pm_close > last_sma50 else "Bearish" if pm_close < last_sma50 else "Neutral"

    # Momentum (premarket)
    roc_series = roc(pm["close"], length=10) if len(pm) >= 10 else pd.Series(dtype=float)
    mom_strength = trend_strength(pm["close"], length=20) if len(pm) >= 20 else 0.0
    last_roc = float(roc_series.iloc[-1]) if len(roc_series) else 0.0
    roc_mom_bias = "Neutral"
    if last_roc > 0 and mom_strength > 0:
        roc_mom_bias = "Bullish"
    elif last_roc < 0 and mom_strength < 0:
        roc_mom_bias = "Bearish"

    # Volatility regime (premarket)
    vol_series = rolling_volatility(pm["close"], length=20) if len(pm) >= 20 else pd.Series(dtype=float)
    vol_regime = classify_volatility(vol_series) if len(vol_series) else "mixed"
    vol_bias = pm_dir if vol_regime == "expanded" and pm_dir in ("Bullish", "Bearish") else "Neutral"

    # RVOL (premarket)
    rvol_series = rvol(pm, length=20) if len(pm) >= 20 else pd.Series(dtype=float)
    last_rvol = float(rvol_series.iloc[-1]) if len(rvol_series) else 1.0
    rvol_bias = pm_dir if last_rvol > 1.2 and pm_dir in ("Bullish", "Bearish") else "Neutral"

    # Z-score (premarket)
    z_series = zscore(pm["close"], length=20) if len(pm) >= 20 else pd.Series(dtype=float)
    last_z = float(z_series.iloc[-1]) if len(z_series) else 0.0
    zscore_bias = "Bullish" if last_z > 0.5 else "Bearish" if last_z < -0.5 else "Neutral"

    # Standard deviation (premarket)
    std_series = pm["close"].rolling(20).std()
    last_std = float(std_series.iloc[-1]) if len(std_series) and pd.notna(std_series.iloc[-1]) else None
    std_mean = float(std_series.dropna().mean()) if len(std_series.dropna()) else None
    std_bias = "Neutral"
    if last_std is not None and std_mean is not None and last_std > std_mean:
        std_bias = pm_dir if pm_dir in ("Bullish", "Bearish") else "Neutral"

    # HTF slope (from broader data)
    htf_slope_bias = htf_bias if htf_bias in ("Bullish", "Bearish") else "Neutral"

    # HTF zones (order blocks / breaker blocks / FVGs)
    zone_bias = "Neutral"
    zone_weight = 0.0
    if zone_confluence is not None and zone_confluence.bias in ("Bullish", "Bearish"):
        zone_bias = zone_confluence.bias
        zone_weight = min(0.15, 0.04 * abs(zone_confluence.score))

    # Overnight range position
    overnight_bias = "Neutral"
    tol = max(pm_close * 0.0005, 1e-6)
    if pm_close > pm_high - tol:
        overnight_bias = "Bullish"
    elif pm_close < pm_low + tol:
        overnight_bias = "Bearish"

    weights = {
        "premarket_trend": 0.30,
        "overnight": 0.25,
        "vwap": 0.15,
        "open_vwap": 0.05,
        "sma20": 0.05,
        "sma50": 0.05,
        "roc_mom": 0.05,
        "vol": 0.05,
        "rvol": 0.05,
        "zscore": 0.05,
        "std": 0.05,
        "htf": 0.05,
        "zone": zone_weight,
        "gap": 0.10,
        "amd_sweep": 0.20,
        "pattern_bias": 0.10,
    }
    signals = [
        (pm_dir, "premarket_trend"),
        (gap_dir, "gap"),
        (vwap_bias, "vwap"),
        (open_vwap_bias, "open_vwap"),
        (sma20_bias, "sma20"),
        (sma50_bias, "sma50"),
        (roc_mom_bias, "roc_mom"),
        (vol_bias, "vol"),
        (rvol_bias, "rvol"),
        (zscore_bias, "zscore"),
        (std_bias, "std"),
        (htf_slope_bias, "htf"),
        (zone_bias, "zone"),
        (overnight_bias, "overnight"),
        (amd_bias, "amd_sweep"),
        (pattern_bias, "pattern_bias"),
    ]
    score = 0.0
    for s, key in signals:
        score += _direction_value(s) * weights.get(key, 0.0)
    if score > 0.0:
        bias = "Bullish"
    elif score < 0.0:
        bias = "Bearish"
    else:
        bias = "Neutral"

    # Directional fallback: use overnight edge when mixed
    if bias == "Neutral" and overnight_bias in ("Bullish", "Bearish"):
        bias = overnight_bias

    details = {
        "premarket_trend": pm_dir,
        "gap": gap_dir,
        "vwap": vwap_bias,
        "open_vwap": open_vwap_bias,
        "sma20": sma20_bias,
        "sma50": sma50_bias,
        "roc_mom": roc_mom_bias,
        "vol": vol_bias,
        "rvol": rvol_bias,
        "zscore": zscore_bias,
        "std": std_bias,
        "htf": htf_slope_bias,
        "zone": zone_bias,
        "overnight": overnight_bias,
        "amd_sweep": amd_bias,
        "pattern_bias": pattern_bias,
    }
    return bias, details, float(score)


def _amd_signal(asia: Optional[SessionStats], london: Optional[SessionStats]) -> Tuple[str, str]:
    if not asia or not london:
        return "Neutral", "Unknown"
    swept_high = london.high >= asia.high
    swept_low = london.low <= asia.low
    if swept_high and not swept_low:
        sweep_bias = "Bearish"
    elif swept_low and not swept_high:
        sweep_bias = "Bullish"
    else:
        sweep_bias = "Neutral"

    if london.close > asia.high:
        close_position = "OutsideHigh"
    elif london.close < asia.low:
        close_position = "OutsideLow"
    else:
        close_position = "Inside"

    if close_position == "OutsideHigh":
        close_bias = "Bullish"
    elif close_position == "OutsideLow":
        close_bias = "Bearish"
    else:
        close_bias = "Neutral"

    if sweep_bias != "Neutral" and close_bias == "Neutral":
        return sweep_bias, close_position
    if sweep_bias == "Neutral" and close_bias != "Neutral":
        return close_bias, close_position
    if sweep_bias == close_bias and sweep_bias != "Neutral":
        return sweep_bias, close_position
    return "Neutral", close_position


def _kalman_trend_signal(series: pd.Series) -> Tuple[str, float]:
    if series is None or series.empty:
        return "n/a", 0.0
    data = series.dropna().values.astype(float)
    if len(data) < 20:
        return "n/a", 0.0

    # Simple constant-velocity Kalman filter
    x = np.array([data[0], 0.0])
    P = np.eye(2)
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[1e-4, 0.0], [0.0, 1e-5]])
    R = np.array([[np.var(data[-20:]) + 1e-6]])

    for z in data[1:]:
        x = A @ x
        P = A @ P @ A.T + Q
        y = z - (H @ x)[0]
        S = H @ P @ H.T + R
        K = (P @ H.T) / S
        x = x + (K.flatten() * y)
        P = (np.eye(2) - K @ H) @ P

    slope = float(x[1])
    threshold = max(data[-1] * 0.0002, 1e-6)
    if slope > threshold:
        return "Bullish", slope
    if slope < -threshold:
        return "Bearish", slope
    return "Neutral", slope


def _hmm_regime_signal(series: pd.Series, max_len: int = 200) -> Tuple[str, float]:
    if series is None or series.empty:
        return "Mixed", 0.0
    returns = series.pct_change().dropna().values.astype(float)
    if len(returns) < 50:
        return "Mixed", 0.0
    if len(returns) > max_len:
        returns = returns[-max_len:]

    n = len(returns)
    k = 2
    mu = np.array([np.mean(returns) - np.std(returns), np.mean(returns) + np.std(returns)])
    var = np.array([np.var(returns) + 1e-6, np.var(returns) * 1.5 + 1e-6])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi = np.array([0.5, 0.5])

    def _gauss(x, mean, variance):
        return np.exp(-0.5 * ((x - mean) ** 2) / variance) / np.sqrt(2 * np.pi * variance)

    for _ in range(5):
        B = np.vstack([_gauss(returns, mu[0], var[0]), _gauss(returns, mu[1], var[1])])
        alpha = np.zeros((k, n))
        scale = np.zeros(n)
        alpha[:, 0] = pi * B[:, 0]
        scale[0] = np.sum(alpha[:, 0]) + 1e-12
        alpha[:, 0] /= scale[0]
        for t in range(1, n):
            alpha[:, t] = (A.T @ alpha[:, t - 1]) * B[:, t]
            scale[t] = np.sum(alpha[:, t]) + 1e-12
            alpha[:, t] /= scale[t]

        beta = np.zeros((k, n))
        beta[:, -1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[:, t] = A @ (B[:, t + 1] * beta[:, t + 1])
            beta[:, t] /= np.sum(beta[:, t]) + 1e-12

        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=0, keepdims=True) + 1e-12

        xi_sum = np.zeros((k, k))
        for t in range(n - 1):
            xi = A * np.outer(alpha[:, t], B[:, t + 1] * beta[:, t + 1])
            denom = np.sum(xi) + 1e-12
            xi_sum += xi / denom

        pi = gamma[:, 0]
        A = xi_sum / (np.sum(xi_sum, axis=1, keepdims=True) + 1e-12)
        for state in range(k):
            weight = gamma[state]
            mu[state] = np.sum(weight * returns) / (np.sum(weight) + 1e-12)
            var[state] = np.sum(weight * (returns - mu[state]) ** 2) / (np.sum(weight) + 1e-12) + 1e-6

    last_gamma = gamma[:, -1]
    strong_state = int(np.argmax(last_gamma))
    abs_mu = np.abs(mu)
    if np.max(abs_mu) < 0.0005:
        return "Mixed", float(last_gamma[strong_state])
    trend_state = int(np.argmax(abs_mu))
    if strong_state == trend_state:
        return "Trend", float(last_gamma[strong_state])
    return "MeanRevert", float(last_gamma[strong_state])


def _bayes_bias(signals: List[str], weight: float, weight_scale: float) -> Tuple[str, float]:
    log_odds = 0.0
    for signal in signals:
        if signal == "Bullish":
            log_odds += weight * weight_scale
        elif signal == "Bearish":
            log_odds -= weight * weight_scale
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    if prob > 0.55:
        return "Bullish", prob
    if prob < 0.45:
        return "Bearish", prob
    return "Neutral", prob


def _load_pattern_weights() -> Dict[str, float]:
    path = Path("data") / "pattern_weights.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    weights = data.get("weights", data)
    if not isinstance(weights, dict):
        return {}
    return {str(k): float(v) for k, v in weights.items() if isinstance(v, (int, float))}




def _atr_direction_threshold(
    df_today: pd.DataFrame,
    df_prev: pd.DataFrame,
    length: int = 20,
    multiplier: float = 0.10,
) -> float:
    ref = df_today if df_today is not None and not df_today.empty else df_prev
    if ref is None or ref.empty:
        return 0.0
    atr_series = atr_like(ref, length=length)
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return 0.0
    return float(atr_series.iloc[-1]) * multiplier


def build_bias(
    df_today: pd.DataFrame,
    df_prev: pd.DataFrame,
    sessions: Dict[str, SessionStats],
    zone_confluence: Optional[ZoneConfluence] = None,
    vol_thresholds: Optional[Tuple[float, float]] = None,
) -> BiasSummary:
    """
    Implements the final bias priority order:
    1. Previous day
    2. Asia
    3. London
    4. Early US (US Open Bias)
    5. Daily & Weekly VWAP posture
    6. Patterns (handled separately, but can be referenced)
    7. Volatility regime
    8. Momentum
    9. Volume (RVOL)
    10. News (modifier only)
    """
    asia = sessions.get("Asia")
    london = sessions.get("London")
    us = sessions.get("US")

    atr_threshold = _atr_direction_threshold(df_today, df_prev)

    # 1. Previous day direction
    prev_dir = 0.0
    prev_range = 0.0
    if not df_prev.empty:
        prev_open = df_prev["open"].iloc[0]
        prev_close = df_prev["close"].iloc[-1]
        prev_dir = prev_close - prev_open
        prev_range = float(df_prev["high"].max() - df_prev["low"].min())
    prev_label = _direction_label(prev_dir, threshold=atr_threshold)

    # 2. Asia
    asia_label = _direction_label(_session_direction(asia), threshold=atr_threshold) if asia else "Neutral"

    # 3. London
    london_label = _direction_label(_session_direction(london), threshold=atr_threshold) if london else "Neutral"

    # Combine into Daily Bias (session-weighted vote)
    prev_weight = max(prev_range, 0.0)
    asia_weight = max(float(getattr(asia, "range", 0.0)), 0.0) if asia else 0.0
    london_weight = max(float(getattr(london, "range", 0.0)), 0.0) if london else 0.0
    total_weight = prev_weight + asia_weight + london_weight
    if total_weight > 0:
        weighted_score = (
            prev_weight * _direction_value(prev_label)
            + asia_weight * _direction_value(asia_label)
            + london_weight * _direction_value(london_label)
        )
        bias_threshold = 0.05 * total_weight
        if weighted_score > bias_threshold:
            daily_bias = "Bullish"
        elif weighted_score < -bias_threshold:
            daily_bias = "Bearish"
        else:
            daily_bias = "Neutral"
        daily_bias_source = "weighted by session range"
    else:
        votes = [prev_label, asia_label, london_label]
        daily_bias = max(set(votes), key=votes.count)
        daily_bias_source = "simple majority vote"

    # Patterns (session + intraday)
    patterns = detect_patterns(sessions, df_today, df_prev)
    pattern_signals = [
        getattr(patterns, "london_continuation_bias", None),
        getattr(patterns, "asia_range_sweep_bias", None),
    ]
    pattern_signals = [s for s in pattern_signals if s in ("Bullish", "Bearish")]
    pattern_bias = max(set(pattern_signals), key=pattern_signals.count) if pattern_signals else "Neutral"

    # Higher-timeframe slope check (1H/4H)
    htf_bias = "n/a"
    htf_1h = None
    htf_4h = None
    if df_today is not None and not df_today.empty:
        rs_1h = resample_ohlcv(df_today, "1H")
        rs_4h = resample_ohlcv(df_today, "4H")
        if len(rs_1h) >= 10:
            htf_1h = float(trend_strength(rs_1h["close"], length=10))
        if len(rs_4h) >= 6:
            htf_4h = float(trend_strength(rs_4h["close"], length=6))
        if htf_1h is not None and htf_4h is not None:
            if htf_1h > 0 and htf_4h > 0:
                htf_bias = "Bullish"
            elif htf_1h < 0 and htf_4h < 0:
                htf_bias = "Bearish"
            else:
                htf_bias = "Neutral"

    # 4. Early US (US Open Bias) – pre-9:30 finalized at 09:10 ET
    us_open_bias_30 = "Neutral"
    us_open_bias_60 = "Neutral"
    amd_bias, london_close_pos = _amd_signal(asia, london)
    pre_us_open_bias, pre_details, pre_score = _premarket_signal(
        df_today,
        amd_bias,
        pattern_bias,
        htf_bias=htf_bias,
        zone_confluence=zone_confluence,
    )
    pre_details["london_close"] = london_close_pos
    if pre_us_open_bias in ("Bullish", "Bearish"):
        us_open_bias_30 = pre_us_open_bias
        us_open_bias_60 = pre_us_open_bias

    # Keep legacy US Open Bias as premarket signal for compatibility
    us_open_bias = us_open_bias_60

    # 5. VWAP posture
    vwap_posture = analyze_vwap_posture(df_today)

    # Anchored VWAP signals
    avwap_weight = 0.07
    avwap_far_penalty = 0.05
    last_price = None
    atr_value = None
    if df_today is not None and not df_today.empty:
        last_price = float(df_today["close"].iloc[-1])
        atr_series = atr_like(df_today, length=20)
        if len(atr_series) and pd.notna(atr_series.iloc[-1]):
            atr_value = float(atr_series.iloc[-1])
    elif df_prev is not None and not df_prev.empty:
        last_price = float(df_prev["close"].iloc[-1])
        atr_series = atr_like(df_prev, length=20)
        if len(atr_series) and pd.notna(atr_series.iloc[-1]):
            atr_value = float(atr_series.iloc[-1])

    df_all = None
    if df_prev is not None and not df_prev.empty and df_today is not None and not df_today.empty:
        df_all = pd.concat([df_prev, df_today], ignore_index=True).sort_values("timestamp")
    elif df_today is not None and not df_today.empty:
        df_all = df_today.copy()
    elif df_prev is not None and not df_prev.empty:
        df_all = df_prev.copy()

    # 6. Overnight range position
    overnight_label = "n/a"
    overnight_high = None
    overnight_low = None
    if asia or london:
        highs = [s.high for s in (asia, london) if s]
        lows = [s.low for s in (asia, london) if s]
        if highs and lows:
            overnight_high = max(highs)
            overnight_low = min(lows)
    if us and overnight_high is not None and overnight_low is not None:
        open_price = float(us.open)
        tol = max(open_price * 0.0005, 1e-6)
        if open_price > overnight_high + tol:
            overnight_label = "Above"
        elif open_price < overnight_low - tol:
            overnight_label = "Below"
        else:
            overnight_label = "Inside"

    # Anchors for US open bias (prior day + overnight highs/lows)
    avwap_us_signals: List[str] = []
    avwap_us_far = False
    if df_all is not None and last_price is not None and atr_value is not None:
        prev_high_ts = _anchor_time_from_extreme(df_prev, "high", "max") if df_prev is not None else None
        prev_low_ts = _anchor_time_from_extreme(df_prev, "low", "min") if df_prev is not None else None

        overnight_start = None
        overnight_end = None
        if df_today is not None and not df_today.empty and "timestamp" in df_today.columns:
            day = pd.to_datetime(df_today["timestamp"].iloc[-1]).date()
            overnight_start = pd.Timestamp.combine(day - pd.Timedelta(days=1), pd.Timestamp("18:00").time())
            overnight_end = pd.Timestamp.combine(day, pd.Timestamp("09:30").time())

        overnight_df = None
        if overnight_start is not None and overnight_end is not None and df_all is not None:
            overnight_df = df_all[
                (df_all["timestamp"] >= overnight_start)
                & (df_all["timestamp"] <= overnight_end)
            ]
        overnight_high_ts = _anchor_time_from_extreme(overnight_df, "high", "max") if overnight_df is not None else None
        overnight_low_ts = _anchor_time_from_extreme(overnight_df, "low", "min") if overnight_df is not None else None

        for anchor_ts in (prev_high_ts, prev_low_ts, overnight_high_ts, overnight_low_ts):
            label, _, dist_units = _anchored_vwap_signal(df_all, anchor_ts, last_price, atr_value)
            if label == "Far":
                avwap_us_far = True
            elif label in ("Bullish", "Bearish"):
                avwap_us_signals.append(label)

    # Anchors for daily bias (prior day + overnight + US open range)
    avwap_daily_signals: List[str] = []
    avwap_daily_far = False
    if df_all is not None and last_price is not None and atr_value is not None:
        or_start = None
        or_end = None
        if df_today is not None and not df_today.empty and "timestamp" in df_today.columns:
            day = pd.to_datetime(df_today["timestamp"].iloc[-1]).date()
            or_start = pd.Timestamp.combine(day, pd.Timestamp("09:30").time())
            or_end = or_start + pd.Timedelta(minutes=30)
        or_df = None
        if or_start is not None and or_end is not None and df_all is not None:
            or_df = df_all[(df_all["timestamp"] >= or_start) & (df_all["timestamp"] <= or_end)]
        or_high_ts = _anchor_time_from_extreme(or_df, "high", "max") if or_df is not None else None
        or_low_ts = _anchor_time_from_extreme(or_df, "low", "min") if or_df is not None else None

        prev_high_ts = _anchor_time_from_extreme(df_prev, "high", "max") if df_prev is not None else None
        prev_low_ts = _anchor_time_from_extreme(df_prev, "low", "min") if df_prev is not None else None

        overnight_start = None
        overnight_end = None
        if df_today is not None and not df_today.empty and "timestamp" in df_today.columns:
            day = pd.to_datetime(df_today["timestamp"].iloc[-1]).date()
            overnight_start = pd.Timestamp.combine(day - pd.Timedelta(days=1), pd.Timestamp("18:00").time())
            overnight_end = pd.Timestamp.combine(day, pd.Timestamp("09:30").time())

        overnight_df = None
        if overnight_start is not None and overnight_end is not None and df_all is not None:
            overnight_df = df_all[
                (df_all["timestamp"] >= overnight_start)
                & (df_all["timestamp"] <= overnight_end)
            ]
        overnight_high_ts = _anchor_time_from_extreme(overnight_df, "high", "max") if overnight_df is not None else None
        overnight_low_ts = _anchor_time_from_extreme(overnight_df, "low", "min") if overnight_df is not None else None

        for anchor_ts in (prev_high_ts, prev_low_ts, overnight_high_ts, overnight_low_ts, or_high_ts, or_low_ts):
            label, _, dist_units = _anchored_vwap_signal(df_all, anchor_ts, last_price, atr_value)
            if label == "Far":
                avwap_daily_far = True
            elif label in ("Bullish", "Bearish"):
                avwap_daily_signals.append(label)

    # 7. Volatility regime
    vol_series = rolling_volatility(df_today["close"], length=20)
    if vol_thresholds:
        vol_regime = classify_volatility(vol_series, p_low=vol_thresholds[0], p_high=vol_thresholds[1])
    else:
        vol_regime = classify_volatility(vol_series)

    # 8. Momentum
    roc_series = roc(df_today["close"], length=10)
    mom_strength = trend_strength(df_today["close"], length=20)

    # 9. Volume (RVOL)
    rvol_series = rvol(df_today, length=20)
    last_rvol = rvol_series.iloc[-1] if len(rvol_series) else 1.0

    # 10. News (modifier only)
    news_signal = combine_news_signals()

    # Event risk filter
    event_label = "none"
    events = load_event_calendar()
    event_date = None
    if df_today is not None and not df_today.empty and "timestamp" in df_today.columns:
        event_date = df_today["timestamp"].iloc[-1].date().isoformat()
    elif df_prev is not None and not df_prev.empty and "timestamp" in df_prev.columns:
        event_date = df_prev["timestamp"].iloc[-1].date().isoformat()
    if event_date and event_date in events:
        names = events[event_date]
        event_label = ", ".join(names)

    # Gap classification
    gap_label = "n/a"
    gap_size = "n/a"
    gap_dir = "flat"
    if not df_prev.empty and not df_today.empty:
        prev_close = float(df_prev["close"].iloc[-1])
        today_open = float(df_today["open"].iloc[0])
        gap = today_open - prev_close
        if gap > 0:
            gap_dir = "up"
        elif gap < 0:
            gap_dir = "down"
        scale = max(prev_range, 1e-6)
        gap_ratio = abs(gap) / scale
        if gap_ratio >= 0.5:
            gap_size = "large"
        elif gap_ratio >= 0.2:
            gap_size = "medium"
        else:
            gap_size = "small"
        gap_label = f"{gap_size} {gap_dir}"

    # Opening range signal (30/60m) for daily bias finalization
    or30_signal, _, _ = _opening_range_signal(df_today, minutes=30)
    or60_signal, _, _ = _opening_range_signal(df_today, minutes=60)
    daily_or_signal = "pending"
    if df_today is not None and not df_today.empty and "timestamp" in df_today.columns:
        last_ts = pd.to_datetime(df_today["timestamp"].iloc[-1])
        cutoff = pd.Timestamp.combine(last_ts.date(), pd.Timestamp("10:45").time())
        if last_ts >= cutoff:
            daily_or_signal = or60_signal

    # Kalman trend signal
    kalman_bias, kalman_slope = _kalman_trend_signal(df_today["close"] if df_today is not None else None)

    # HMM regime signal (trend vs mean-revert)
    hmm_regime, hmm_conf = _hmm_regime_signal(df_today["close"] if df_today is not None else None)

    # Confidence scoring (simple heuristic)
    daily_conf = 0.5
    us_conf = 0.5

    # Alignment boosts
    if daily_bias == prev_label:
        daily_conf += 0.1
    if daily_bias == asia_label:
        daily_conf += 0.1
    if daily_bias == london_label:
        daily_conf += 0.1
    if daily_bias == us_open_bias:
        daily_conf += 0.1

    # Anchored VWAP (daily)
    if avwap_daily_signals:
        avwap_daily_bias = max(set(avwap_daily_signals), key=avwap_daily_signals.count)
        if daily_bias == avwap_daily_bias:
            daily_conf += avwap_weight
        else:
            daily_conf -= avwap_weight
    if avwap_daily_far:
        daily_conf -= avwap_far_penalty

    # Overnight range alignment
    if overnight_label == "Above" and daily_bias == "Bullish":
        daily_conf += 0.05
    elif overnight_label == "Below" and daily_bias == "Bearish":
        daily_conf += 0.05
    elif overnight_label in ("Above", "Below"):
        daily_conf -= 0.05

    # Gap alignment
    if gap_size == "large":
        if gap_dir == "up" and daily_bias == "Bullish":
            daily_conf += 0.05
        elif gap_dir == "down" and daily_bias == "Bearish":
            daily_conf += 0.05
        elif gap_dir in ("up", "down"):
            daily_conf -= 0.05

    # HTF alignment
    if htf_bias in ("Bullish", "Bearish"):
        if htf_bias == daily_bias:
            daily_conf += 0.05
        else:
            daily_conf -= 0.05

    # VWAP alignment
    if daily_bias == "Bullish" and vwap_posture.daily_above and vwap_posture.weekly_above:
        daily_conf += 0.1
    if daily_bias == "Bearish" and not vwap_posture.daily_above and not vwap_posture.weekly_above:
        daily_conf += 0.1

    # Volatility regime
    if vol_regime == "expanded":
        daily_conf += 0.05
    elif vol_regime == "compressed":
        daily_conf -= 0.05

    # Momentum
    last_roc = roc_series.iloc[-1] if len(roc_series) else 0.0
    if daily_bias == "Bullish" and last_roc > 0 and mom_strength > 0:
        daily_conf += 0.05
    if daily_bias == "Bearish" and last_roc < 0 and mom_strength < 0:
        daily_conf += 0.05

    # Regime switch (trend vs mean-revert)
    regime = _classify_regime(vol_regime, last_roc, mom_strength)
    roc_bias = "Bullish" if last_roc > 0 else "Bearish" if last_roc < 0 else "Neutral"
    if regime == "Trend" and roc_bias in ("Bullish", "Bearish"):
        if daily_bias == roc_bias:
            daily_conf += 0.05
        else:
            daily_conf -= 0.05
    elif regime == "MeanRevert" and roc_bias in ("Bullish", "Bearish"):
        if daily_bias == roc_bias:
            daily_conf -= 0.05
        else:
            daily_conf += 0.03

    # Day-type adjustment (soft weighting)
    day_type = _classify_day_type(
        df_today,
        prev_range,
        overnight_high,
        overnight_low,
        last_roc,
        mom_strength,
    )
    if day_type == "Chop":
        daily_conf -= 0.15
    elif day_type == "Trend":
        daily_conf += 0.05

    # Event risk adjustment
    if event_label != "none":
        daily_conf -= 0.10

    # Volume
    if last_rvol > 1.2:
        daily_conf += 0.05
    elif last_rvol < 0.8:
        daily_conf -= 0.05

    # Anchored VWAP (US open confidence mirror)
    if avwap_us_signals:
        avwap_us_bias = max(set(avwap_us_signals), key=avwap_us_signals.count)
        if us_open_bias in ("Bullish", "Bearish"):
            if us_open_bias == avwap_us_bias:
                us_conf += avwap_weight
            else:
                us_conf -= avwap_weight
    if avwap_us_far:
        us_conf -= avwap_far_penalty

    # News tone (modifier only)
    if news_signal.tone == "risk_on" and daily_bias == "Bullish":
        daily_conf += 0.05
    elif news_signal.tone == "risk_off" and daily_bias == "Bearish":
        daily_conf += 0.05
    elif news_signal.tone in ("risk_on", "risk_off"):
        daily_conf -= 0.05

    # Volatility-adjusted prior scaling
    if vol_regime == "expanded":
        prior_scale = 1.1
    elif vol_regime == "compressed":
        prior_scale = 0.9
    else:
        prior_scale = 1.0

    daily_pattern_signals = [
        pattern_bias,
        getattr(patterns, "us_open_gap_fill_bias", None),
        getattr(patterns, "orb_60_bias", None),
        getattr(patterns, "failed_orb_60_bias", None),
        getattr(patterns, "vwap_reclaim_reject_bias", None),
        getattr(patterns, "power_hour_bias", None),
    ]
    daily_pattern_signals = [s for s in daily_pattern_signals if s in ("Bullish", "Bearish")]
    alignment_set = [daily_bias, htf_bias, roc_bias, ("Bullish" if vwap_posture.daily_above else "Bearish")]
    pattern_bias_today = None
    if daily_pattern_signals:
        pattern_bias_today = max(set(daily_pattern_signals), key=daily_pattern_signals.count)
        alignment_count = sum(1 for s in alignment_set if s == pattern_bias_today)
        if alignment_count < 2:
            pattern_bias_today = None

    # Ensemble scoring (equal weights)
    signal_list = [
        prev_label,
        asia_label,
        london_label,
        us_open_bias if us_open_bias in ("Bullish", "Bearish") else "Neutral",
        "Bullish" if vwap_posture.daily_above and vwap_posture.weekly_above else "Bearish" if (not vwap_posture.daily_above and not vwap_posture.weekly_above) else "Neutral",
        "Bullish" if overnight_label == "Above" else "Bearish" if overnight_label == "Below" else "Neutral",
        "Bullish" if gap_size == "large" and gap_dir == "up" else "Bearish" if gap_size == "large" and gap_dir == "down" else "Neutral",
        htf_bias if htf_bias in ("Bullish", "Bearish") else "Neutral",
        roc_bias,
        kalman_bias if kalman_bias in ("Bullish", "Bearish") else "Neutral",
        daily_or_signal if daily_or_signal in ("Bullish", "Bearish") else "Neutral",
    ]
    pattern_weight = None
    if pattern_bias_today:
        weights = _load_pattern_weights()
        pattern_weight = weights.get(pattern_bias_today.lower(), weights.get("default", 0.5))
        if pattern_weight >= 0.6:
            signal_list.extend([pattern_bias_today, pattern_bias_today])
        elif pattern_weight >= 0.4:
            signal_list.append(pattern_bias_today)
    score = sum(_direction_value(s) for s in signal_list)
    if score > 1:
        ensemble_bias = "Bullish"
    elif score < -1:
        ensemble_bias = "Bearish"
    else:
        ensemble_bias = "Neutral"

    # Bayesian bias update (equal weights)
    bayes_bias, bayes_prob = _bayes_bias(signal_list, weight=0.25, weight_scale=prior_scale)

    # Finalize bias using Bayesian result when confident
    final_bias = daily_bias
    final_source = daily_bias_source
    if bayes_bias != "Neutral":
        final_bias = bayes_bias
        final_source = "bayesian equal-weight update"
    elif ensemble_bias != "Neutral":
        final_bias = ensemble_bias
        final_source = "ensemble score"

    # Clamp
    daily_conf = max(0.1, min(0.95, daily_conf))
    us_conf = max(0.1, min(0.95, us_conf))

    impact_notes = [
        "Session weighting favors larger-range sessions over quieter ones.",
        "ATR-scaled thresholds reduce false bias on quiet days.",
    ]
    if overnight_label == "Above":
        impact_notes.append("US open above overnight range favors trend continuation.")
    elif overnight_label == "Below":
        impact_notes.append("US open below overnight range favors downside follow-through.")
    elif overnight_label == "Inside":
        impact_notes.append("US open inside overnight range favors mean reversion early.")

    if gap_size == "large" and gap_dir in ("up", "down"):
        impact_notes.append("Large opening gap reduces the odds of an immediate fade.")

    if htf_bias in ("Bullish", "Bearish"):
        impact_notes.append("HTF slope alignment supports follow-through; conflict adds chop risk.")

    if day_type == "Chop":
        impact_notes.append("Chop day detected: bias kept but confidence reduced.")
    elif day_type == "Trend":
        impact_notes.append("Trend day detected: confidence boosted for follow-through.")

    if regime == "Trend":
        impact_notes.append("Trend regime: continuation signals weighted higher.")
    elif regime == "MeanRevert":
        impact_notes.append("Mean-revert regime: fade risk higher on extremes.")

    if avwap_daily_signals:
        impact_notes.append("Anchored VWAP aligned with daily bias for fair-value context.")
    if avwap_daily_far:
        impact_notes.append("Price stretched far from anchored VWAP: mean-revert risk higher.")

    if pre_us_open_bias in ("Bullish", "Bearish"):
        impact_notes.append(
            "Premarket bias (09:10 ET) uses overnight range, gap, VWAP, and premarket trend."
        )
    elif pre_us_open_bias == "Neutral":
        impact_notes.append("US Open bias is neutral due to mixed premarket signals.")

    if amd_bias in ("Bullish", "Bearish"):
        impact_notes.append(
            f"AMD signal favors {amd_bias.lower()} distribution; London close {london_close_pos.lower()}."
        )

    if daily_or_signal in ("Bullish", "Bearish"):
        impact_notes.append(f"Daily bias confirmed after 10:45 ET by 60m OR ({daily_or_signal.lower()}).")
    elif daily_or_signal == "pending":
        impact_notes.append("Daily bias not finalized until 10:45 ET (60m OR pending).")

    if pattern_bias_today:
        impact_notes.append("Patterns aligned with other confluences and tilted the bias.")

    if kalman_bias in ("Bullish", "Bearish"):
        impact_notes.append("Kalman trend aligns bias with smoothed slope.")

    if hmm_regime in ("Trend", "MeanRevert"):
        impact_notes.append(f"HMM regime detected: {hmm_regime.lower()} state active.")

    if event_label != "none":
        impact_notes.append(f"Event risk ({event_label}): confidence reduced.")

    us_open_note = ""
    if pre_us_open_bias == "Neutral":
        lean = "Neutral"
        if pre_score > 0:
            lean = "Bullish"
        elif pre_score < 0:
            lean = "Bearish"
        bull_reasons = [k for k, v in pre_details.items() if v == "Bullish"]
        bear_reasons = [k for k, v in pre_details.items() if v == "Bearish"]
        if lean == "Bullish" and bull_reasons:
            us_open_note = f" US Open neutral but leaning bullish due to {', '.join(bull_reasons)}."
        elif lean == "Bearish" and bear_reasons:
            us_open_note = f" US Open neutral but leaning bearish due to {', '.join(bear_reasons)}."
        else:
            us_open_note = " US Open neutral due to mixed premarket signals."

    explanation = (
        f"Daily Bias is {final_bias} based on previous day ({prev_label}), "
        f"Asia ({asia_label}), and London ({london_label}) with {final_source}. "
        f"US Open Bias is {us_open_bias_30} (30m) / {us_open_bias_60} (60m)."
        f"{us_open_note} "
        f"Overnight range position is {overnight_label}. "
        f"Gap is {gap_label}. HTF slope bias is {htf_bias}. "
        f"Day type is {day_type}. Regime is {regime} / HMM {hmm_regime}. Event risk is {event_label}. "
        f"Volatility regime is '{vol_regime}', momentum and ROC aligned at {last_roc:.3f}, "
        f"RVOL at {last_rvol:.2f}. Kalman slope at {kalman_slope:.4f}. "
        f"Bayes prob {bayes_prob:.2f} from equal weights. "
        f"News tone is {news_signal.tone} and only adjusts confidence, not direction. "
        f"Day impact: {' '.join(impact_notes)}"
    )

    if zone_confluence is not None and zone_confluence.bias != "Neutral":
        score = abs(zone_confluence.score)
        bias_boost = min(0.15, 0.04 * score)
        daily_conf += bias_boost
        if score >= 2 and final_bias != zone_confluence.bias:
            final_bias = zone_confluence.bias
        explanation += (
            f" HTF zones suggest {zone_confluence.bias} bias "
            f"(score {zone_confluence.score:.2f}; {zone_confluence.notes})."
        )

    daily_conf = max(0.1, min(0.95, daily_conf))
    us_conf = max(0.1, min(0.95, us_conf))

    return BiasSummary(
        daily_bias=final_bias,
        daily_confidence=daily_conf,
        us_open_bias=us_open_bias,
        us_open_confidence=us_conf,
        us_open_bias_30=us_open_bias_30,
        us_open_confidence_30=us_conf if us_open_bias_30 in ("Bullish", "Bearish") else 0.0,
        us_open_bias_60=us_open_bias_60,
        us_open_confidence_60=us_conf if us_open_bias_60 in ("Bullish", "Bearish") else 0.0,
        explanation=explanation,
        vwap_comment=vwap_posture.comment,
        news_comment=news_signal.explanation,
        amd_summary=(
            f"Sweep {amd_bias.lower()}; London close {london_close_pos.lower()}"
            if amd_bias in ("Bullish", "Bearish") or london_close_pos != "Unknown"
            else None
        ),
    )