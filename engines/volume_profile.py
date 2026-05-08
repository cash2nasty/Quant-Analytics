import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data.session_reference import get_session_windows_for_date


@dataclass
class VolumeNode:
    low: float
    high: float
    center: float
    volume: float
    prominence: float
    method: str = "Primary"
    confidence: str = "High"


@dataclass
class VolumeProfileResult:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    poc: Optional[float]
    vah: Optional[float]
    val: Optional[float]
    profile_high: Optional[float]
    profile_low: Optional[float]
    total_volume: float
    value_area_pct: float
    value_area_width: float
    close_price: Optional[float]
    close_location: str
    shape: str
    hvns: List[VolumeNode]
    lvns: List[VolumeNode]
    histogram: pd.DataFrame


def trading_day_bounds(trading_day: dt.date) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp.combine(trading_day - dt.timedelta(days=1), dt.time(18, 0))
    end = pd.Timestamp.combine(trading_day, dt.time(16, 59, 59))
    return start, end


def _slice_window(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])  # defensive cleanup
    mask = (out["timestamp"] >= start_ts) & (out["timestamp"] <= end_ts)
    return out.loc[mask].sort_values("timestamp").reset_index(drop=True)


def _build_histogram(df: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["price", "volume"])

    rows = []
    for _, r in df.iterrows():
        low = float(r.get("low", 0.0))
        high = float(r.get("high", 0.0))
        vol = float(r.get("volume", 0.0))
        if high < low:
            low, high = high, low
        if tick_size <= 0:
            tick_size = 0.25

        start = round(low / tick_size) * tick_size
        end = round(high / tick_size) * tick_size
        steps = int(round((end - start) / tick_size))

        if steps <= 0:
            price = round(float(r.get("close", low)) / tick_size) * tick_size
            rows.append((price, vol))
            continue

        share = vol / float(steps + 1)
        for i in range(steps + 1):
            rows.append((round(start + i * tick_size, 8), share))

    if not rows:
        return pd.DataFrame(columns=["price", "volume"])

    hist = pd.DataFrame(rows, columns=["price", "volume"]).groupby("price", as_index=False)["volume"].sum()
    return hist.sort_values("price").reset_index(drop=True)


def _value_area(hist: pd.DataFrame, value_area_pct: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if hist is None or hist.empty:
        return None, None, None

    h = hist.copy()
    h = h.sort_values("price").reset_index(drop=True)
    total = float(h["volume"].sum())
    if total <= 0:
        return None, None, None

    poc_idx = int(h["volume"].idxmax())
    poc = float(h.loc[poc_idx, "price"])

    target = max(0.0, min(1.0, float(value_area_pct))) * total
    included = {poc_idx}
    acc = float(h.loc[poc_idx, "volume"])
    left = poc_idx - 1
    right = poc_idx + 1

    while acc < target and (left >= 0 or right < len(h)):
        left_vol = float(h.loc[left, "volume"]) if left >= 0 else -1.0
        right_vol = float(h.loc[right, "volume"]) if right < len(h) else -1.0
        if right_vol >= left_vol:
            if right < len(h):
                included.add(right)
                acc += right_vol
                right += 1
            elif left >= 0:
                included.add(left)
                acc += left_vol
                left -= 1
        else:
            if left >= 0:
                included.add(left)
                acc += left_vol
                left -= 1
            elif right < len(h):
                included.add(right)
                acc += right_vol
                right += 1

    prices = h.loc[sorted(included), "price"]
    return float(prices.min()), float(prices.max()), poc


def _find_nodes(hist: pd.DataFrame, find_high: bool = True) -> List[VolumeNode]:
    if hist is None or len(hist) < 5:
        return []

    h = hist.copy()
    h["smooth"] = h["volume"].rolling(3, center=True, min_periods=1).mean()
    vals = h["smooth"].tolist()

    nodes: List[VolumeNode] = []
    prices = h["price"].astype(float).tolist()

    tick = 0.25
    if len(prices) >= 2:
        diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1) if (prices[i + 1] - prices[i]) > 0]
        if diffs:
            tick = min(diffs)
    def _build_tier(multiplier: float, allow_fallback: bool, method: str, confidence: str) -> List[VolumeNode]:
        tier_nodes: List[VolumeNode] = []
        for i in range(1, len(vals) - 1):
            l, c, r = vals[i - 1], vals[i], vals[i + 1]
            if find_high:
                cond = c > l and c > r
                prominence = c - max(l, r)
            else:
                cond = c < l and c < r
                prominence = min(l, r) - c

            if not cond:
                continue
            if prominence <= 0:
                continue

            left_idx = i
            right_idx = i
            c_val = float(c)

            if find_high:
                threshold = c_val - float(prominence) * multiplier
                while left_idx > 0 and float(vals[left_idx - 1]) >= threshold:
                    left_idx -= 1
                while right_idx < len(vals) - 1 and float(vals[right_idx + 1]) >= threshold:
                    right_idx += 1
            else:
                threshold = c_val + float(prominence) * multiplier
                while left_idx > 0 and float(vals[left_idx - 1]) <= threshold:
                    left_idx -= 1
                while right_idx < len(vals) - 1 and float(vals[right_idx + 1]) <= threshold:
                    right_idx += 1

            low = float(h.loc[left_idx, "price"])
            high = float(h.loc[right_idx, "price"])

            if low == high:
                if not allow_fallback:
                    continue
                if left_idx > 0:
                    low = float(h.loc[left_idx - 1, "price"])
                if right_idx < len(h) - 1:
                    high = float(h.loc[right_idx + 1, "price"])
                if low == high:
                    low = low - tick
                    high = high + tick

            zone_vol = float(h.loc[left_idx:right_idx, "volume"].sum()) if right_idx >= left_idx else float(h.loc[i, "volume"])
            center = (low + high) / 2.0
            tier_nodes.append(
                VolumeNode(
                    low=low,
                    high=high,
                    center=center,
                    volume=zone_vol,
                    prominence=float(prominence),
                    method=method,
                    confidence=confidence,
                )
            )

        if not tier_nodes:
            return []

        q = 0.4 if method != "Fallback" else 0.2
        prom_threshold = pd.Series([n.prominence for n in tier_nodes]).quantile(q)
        tier_nodes = [n for n in tier_nodes if n.prominence >= float(prom_threshold)]
        tier_nodes = sorted(tier_nodes, key=lambda n: (n.prominence, n.volume), reverse=True)
        return tier_nodes

    def _overlaps(a: VolumeNode, b: VolumeNode) -> bool:
        return (a.low <= b.high) and (a.high >= b.low)

    def _add_non_overlapping(target: List[VolumeNode], source: List[VolumeNode]) -> List[VolumeNode]:
        out = list(target)
        for n in source:
            if any(_overlaps(n, e) for e in out):
                continue
            out.append(n)
        return out

    # Tier 1: strict primary prominence rule.
    primary_nodes = _build_tier(0.5, False, "Primary", "High")
    nodes = _add_non_overlapping(nodes, primary_nodes)

    # Tier 2: adaptive but still natural prominence ranges.
    if len(nodes) < 3:
        adaptive_nodes_35 = _build_tier(0.35, False, "Adaptive", "Medium")
        nodes = _add_non_overlapping(nodes, adaptive_nodes_35)
    if len(nodes) < 3:
        adaptive_nodes_20 = _build_tier(0.2, False, "Adaptive", "Medium")
        nodes = _add_non_overlapping(nodes, adaptive_nodes_20)

    # Tier 3 fallback: only when no natural zones exist.
    if not nodes:
        fallback_nodes = _build_tier(0.5, True, "Fallback", "Low")
        nodes = _add_non_overlapping(nodes, fallback_nodes)

    if not nodes:
        return []

    method_priority = {"Primary": 0, "Adaptive": 1, "Fallback": 2}
    nodes = sorted(
        nodes,
        key=lambda n: (method_priority.get(n.method, 9), -float(n.prominence), -float(n.volume)),
    )
    return nodes[:5]


def classify_profile_shape(hist: pd.DataFrame) -> str:
    if hist is None or hist.empty:
        return "n/a"
    h = hist.copy().sort_values("price").reset_index(drop=True)
    v = h["volume"]
    if float(v.sum()) <= 0:
        return "n/a"

    mean = float((h["price"] * h["volume"]).sum() / v.sum())
    var = float((((h["price"] - mean) ** 2) * h["volume"]).sum() / v.sum())
    std = max(var ** 0.5, 1e-9)
    skew = float((((h["price"] - mean) ** 3) * h["volume"]).sum() / v.sum()) / (std ** 3)

    highs = _find_nodes(h, find_high=True)
    if len(highs) >= 2:
        p1, p2 = highs[0].center, highs[1].center
        pr = float(h["price"].max() - h["price"].min())
        if pr > 0 and abs(p1 - p2) >= pr * 0.25:
            return "B-shape"

    if skew > 0.35:
        return "P-shape"
    if skew < -0.35:
        return "b-shape"

    top_share = float(v.max() / v.sum())
    if top_share < 0.12:
        return "Trend-like"
    return "D-shape"


def build_profile(
    name: str,
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    tick_size: float = 0.25,
    value_area_pct: float = 0.70,
) -> VolumeProfileResult:
    window = _slice_window(df, start_ts, end_ts)
    hist = _build_histogram(window, tick_size=tick_size)

    val, vah, poc = _value_area(hist, value_area_pct=value_area_pct)
    close_price = float(window["close"].iloc[-1]) if not window.empty else None
    profile_high = float(hist["price"].max()) if not hist.empty else None
    profile_low = float(hist["price"].min()) if not hist.empty else None
    total_volume = float(hist["volume"].sum()) if not hist.empty else 0.0

    if close_price is None or val is None or vah is None:
        close_location = "n/a"
    elif close_price > vah:
        close_location = "Above VAH"
    elif close_price < val:
        close_location = "Below VAL"
    else:
        close_location = "Within Value"

    hvns = _find_nodes(hist, find_high=True)
    lvns = _find_nodes(hist, find_high=False)
    shape = classify_profile_shape(hist)

    return VolumeProfileResult(
        name=name,
        start=start_ts,
        end=end_ts,
        poc=poc,
        vah=vah,
        val=val,
        profile_high=profile_high,
        profile_low=profile_low,
        total_volume=total_volume,
        value_area_pct=float(value_area_pct),
        value_area_width=float((vah - val) if (vah is not None and val is not None) else 0.0),
        close_price=close_price,
        close_location=close_location,
        shape=shape,
        hvns=hvns,
        lvns=lvns,
        histogram=hist,
    )


def build_session_profiles(
    df: pd.DataFrame,
    trading_day: dt.date,
    tick_size: float = 0.25,
    value_area_pct: float = 0.70,
) -> Dict[str, VolumeProfileResult]:
    windows = get_session_windows_for_date(trading_day)
    td_start, td_end = trading_day_bounds(trading_day)

    profiles = {
        "Asia": build_profile("Asia", df, pd.Timestamp(windows["Asia"]["start"]), pd.Timestamp(windows["Asia"]["end"]), tick_size, value_area_pct),
        "London": build_profile("London", df, pd.Timestamp(windows["London"]["start"]), pd.Timestamp(windows["London"]["end"]), tick_size, value_area_pct),
        "NY": build_profile("NY", df, pd.Timestamp(windows["US"]["start"]), pd.Timestamp(windows["US"]["end"]), tick_size, value_area_pct),
        "Full Day": build_profile("Full Day", df, td_start, td_end, tick_size, value_area_pct),
    }
    return profiles


def summarize_previous_day(
    full_day: VolumeProfileResult,
    ny: VolumeProfileResult,
) -> Tuple[str, str]:
    direction = "balanced"
    if full_day.close_location == "Above VAH":
        direction = "upside acceptance"
    elif full_day.close_location == "Below VAL":
        direction = "downside acceptance"

    summary = (
        f"Previous day full profile closed {full_day.close_location} with {full_day.shape}. "
        f"Previous NY profile closed {ny.close_location} with {ny.shape}."
    )

    if direction == "upside acceptance":
        expectation = (
            "Expect upside continuation attempt on the next open. Watch for pullback holds above value and acceptance above POC."
        )
    elif direction == "downside acceptance":
        expectation = (
            "Expect downside continuation attempt on the next open. Watch for failed rallies into value and acceptance below POC."
        )
    else:
        expectation = (
            "Expect rotational open unless early value migration forms. Watch VAH/VAL rejection quality for directional commitment."
        )

    return summary, expectation


def anticipate_ny_from_sessions(asia: VolumeProfileResult, london: VolumeProfileResult) -> str:
    if asia.close_location == "n/a" or london.close_location == "n/a":
        return "NY anticipation unavailable (insufficient session data)."

    if asia.close_location == "Above VAH" and london.close_location == "Above VAH":
        return "Asia and London both accepted above value. Anticipate NY continuation unless open fails back inside value."
    if asia.close_location == "Below VAL" and london.close_location == "Below VAL":
        return "Asia and London both accepted below value. Anticipate NY downside continuation unless open reclaims value."

    if asia.close_location != london.close_location:
        return "Asia/London disagree on value acceptance. Anticipate NY two-sided rotation until one side gains acceptance."

    return "Asia and London suggest balance. Anticipate NY rotational behavior around value unless value migrates early."
