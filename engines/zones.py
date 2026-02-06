from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class Zone:
    kind: str
    side: str
    low: float
    high: float
    start: pd.Timestamp
    end: pd.Timestamp
    timeframe: str


@dataclass
class ZoneConfluence:
    score: float
    bias: str
    bullish_hits: int
    bearish_hits: int
    notes: str


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    data = df.copy()
    if "timestamp" not in data.columns:
        return pd.DataFrame()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").set_index("timestamp")
    ohlc = data.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return ohlc


def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def detect_liquidity_lines(df: pd.DataFrame, window: int = 3) -> List[float]:
    if df is None or df.empty:
        return []
    highs = df["high"].values
    lows = df["low"].values
    lines: List[float] = []
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            lines.append(float(highs[i]))
        if lows[i] == min(lows[i - window : i + window + 1]):
            lines.append(float(lows[i]))
    return lines


def detect_fvg(df: pd.DataFrame, timeframe: str) -> List[Zone]:
    zones: List[Zone] = []
    if df is None or len(df) < 3:
        return zones
    end_time = df["timestamp"].iloc[-1]
    for i in range(len(df) - 2):
        c1 = df.iloc[i]
        c3 = df.iloc[i + 2]
        if c1.high < c3.low:
            low = float(c1.high)
            high = float(c3.low)
            if high > low:
                zones.append(
                    Zone(
                        kind="fvg",
                        side="bullish",
                        low=low,
                        high=high,
                        start=c1.timestamp,
                        end=end_time,
                        timeframe=timeframe,
                    )
                )
        if c1.low > c3.high:
            low = float(c3.high)
            high = float(c1.low)
            if high > low:
                zones.append(
                    Zone(
                        kind="fvg",
                        side="bearish",
                        low=low,
                        high=high,
                        start=c1.timestamp,
                        end=end_time,
                        timeframe=timeframe,
                    )
                )
    return zones


def detect_order_blocks(df: pd.DataFrame, timeframe: str) -> List[Zone]:
    zones: List[Zone] = []
    if df is None or len(df) < 5:
        return zones
    atr = compute_atr(df, length=14)
    end_time = df["timestamp"].iloc[-1]
    for i in range(len(df) - 4):
        c0 = df.iloc[i]
        c3 = df.iloc[i + 3]
        if pd.isna(atr.iloc[i]):
            continue
        displacement = abs(float(c3.close) - float(c0.close))
        if displacement < float(atr.iloc[i]) * 1.2:
            continue
        low = float(c0.low)
        high = float(c0.high)
        if high <= low:
            continue
        if c0.close < c0.open and c3.close > c0.high:
            zones.append(
                Zone(
                    kind="order_block",
                    side="bullish",
                    low=low,
                    high=high,
                    start=c0.timestamp,
                    end=end_time,
                    timeframe=timeframe,
                )
            )
        if c0.close > c0.open and c3.close < c0.low:
            zones.append(
                Zone(
                    kind="order_block",
                    side="bearish",
                    low=low,
                    high=high,
                    start=c0.timestamp,
                    end=end_time,
                    timeframe=timeframe,
                )
            )
    return zones


def detect_breakers(df: pd.DataFrame, order_blocks: List[Zone], timeframe: str) -> List[Zone]:
    zones: List[Zone] = []
    if df is None or df.empty:
        return zones
    for ob in order_blocks:
        after = df[df["timestamp"] > ob.start]
        if after.empty:
            continue
        if ob.side == "bullish":
            broken = after[after["close"] < ob.low]
            if not broken.empty:
                b = broken.iloc[0]
                if ob.high > ob.low:
                    zones.append(
                        Zone(
                            kind="breaker",
                            side="bearish",
                            low=ob.low,
                            high=ob.high,
                            start=b.timestamp,
                            end=df["timestamp"].iloc[-1],
                            timeframe=timeframe,
                        )
                    )
        if ob.side == "bearish":
            broken = after[after["close"] > ob.high]
            if not broken.empty:
                b = broken.iloc[0]
                if ob.high > ob.low:
                    zones.append(
                        Zone(
                            kind="breaker",
                            side="bullish",
                            low=ob.low,
                            high=ob.high,
                            start=b.timestamp,
                            end=df["timestamp"].iloc[-1],
                            timeframe=timeframe,
                        )
                    )
    return zones


def _filter_major_15m(zones: List[Zone], df_15m: pd.DataFrame) -> List[Zone]:
    if df_15m is None or df_15m.empty:
        return []
    lines = detect_liquidity_lines(df_15m, window=3)
    if not lines:
        return []
    last_price = float(df_15m["close"].iloc[-1])
    tol = max(last_price * 0.0015, 1e-6)
    major: List[Zone] = []
    for z in zones:
        mid = (z.low + z.high) / 2.0
        if any(abs(mid - line) <= tol for line in lines):
            major.append(z)
    return major


def build_htf_zones(df: pd.DataFrame) -> List[Zone]:
    zones: List[Zone] = []
    if df is None or df.empty:
        return zones
    tf_map = {
        "4H": "4H",
        "1H": "1H",
        "30m": "30min",
        "15m": "15min",
    }
    for tf, rule in tf_map.items():
        rs = resample_ohlcv(df, rule)
        if rs.empty:
            continue
        fvg = detect_fvg(rs, tf)
        obs = detect_order_blocks(rs, tf)
        brk = detect_breakers(rs, obs, tf)
        tf_zones = fvg + obs + brk
        if tf == "15m":
            tf_zones = _filter_major_15m(tf_zones, rs)
        zones.extend(tf_zones)
    return zones


def summarize_zone_confluence(zones: List[Zone], last_price: Optional[float]) -> ZoneConfluence:
    if not zones or last_price is None:
        return ZoneConfluence(
            score=0.0,
            bias="Neutral",
            bullish_hits=0,
            bearish_hits=0,
            notes="No zone confluence detected.",
        )

    weights = {"4H": 1.5, "1H": 1.2, "30m": 1.0, "15m": 0.8}
    score = 0.0
    bullish_hits = 0
    bearish_hits = 0
    for z in zones:
        weight = weights.get(z.timeframe, 1.0)
        buffer = max(abs(z.high - z.low) * 0.25, last_price * 0.0015)
        if (z.low - buffer) <= last_price <= (z.high + buffer):
            if z.side == "bullish":
                score += weight
                bullish_hits += 1
            else:
                score -= weight
                bearish_hits += 1

    if score > 0.5:
        bias = "Bullish"
    elif score < -0.5:
        bias = "Bearish"
    else:
        bias = "Neutral"

    notes = f"Bullish zones: {bullish_hits}, Bearish zones: {bearish_hits}."

    return ZoneConfluence(
        score=score,
        bias=bias,
        bullish_hits=bullish_hits,
        bearish_hits=bearish_hits,
        notes=notes,
    )


def zone_size_points(zone: Zone) -> float:
    return float(max(zone.high - zone.low, 0.0))


def _timeframe_rule(timeframe: str) -> str:
    mapping = {"4H": "4H", "1H": "1H", "30m": "30min", "15m": "15min"}
    return mapping.get(timeframe, "1H")


def zone_liquidity_scores(df: pd.DataFrame, zone: Zone) -> tuple:
    if df is None or df.empty:
        return (0.0, 0.0)
    rs = resample_ohlcv(df, _timeframe_rule(zone.timeframe))
    if rs.empty:
        return (0.0, 0.0)

    lines = detect_liquidity_lines(rs, window=3)
    mid = (zone.low + zone.high) / 2.0
    tol = max(mid * 0.0015, 1e-6)
    density = len([line for line in lines if (zone.low - tol) <= line <= (zone.high + tol)])

    overlap = rs[(rs["low"] <= zone.high) & (rs["high"] >= zone.low)]
    avg_vol = float(rs["volume"].mean()) if "volume" in rs.columns else 0.0
    if avg_vol <= 0 or overlap.empty:
        volume_score = 0.0
    else:
        volume_score = float(overlap["volume"].sum()) / (avg_vol * max(1, len(overlap)))

    return (float(density), float(volume_score))


def score_zone_setup(
    zone: Zone,
    last_price: Optional[float],
    touched: bool,
    liquidity_density: float,
    volume_score: float,
) -> float:
    if last_price is None or last_price <= 0:
        return 0.0
    weights = {"4H": 1.5, "1H": 1.2, "30m": 1.0, "15m": 0.8}
    weight = weights.get(zone.timeframe, 1.0)

    size = zone_size_points(zone)
    size_scale = max(last_price * 0.0015, 1e-6)
    size_score = min(1.5, size / size_scale)

    mid = (zone.low + zone.high) / 2.0
    distance = abs(mid - last_price)
    dist_scale = max(last_price * 0.003, 1e-6)
    proximity_score = 1.0 / (1.0 + (distance / dist_scale))

    liquidity_score = min(2.0, liquidity_density * 0.15 + volume_score * 0.3)
    freshness = 0.3 if not touched else 0.0

    return float(weight * (0.4 * proximity_score + 0.3 * size_score + 0.3 * liquidity_score) + freshness)


def is_zone_touched(df: pd.DataFrame, zone: Zone) -> bool:
    if df is None or df.empty:
        return False
    after = df[df["timestamp"] > zone.start]
    if after.empty:
        return False
    touched = after[(after["low"] <= zone.high) & (after["high"] >= zone.low)]
    return not touched.empty


def filter_untouched_zones(df: pd.DataFrame, zones: List[Zone]) -> List[Zone]:
    untouched: List[Zone] = []
    for zone in zones:
        if not is_zone_touched(df, zone):
            untouched.append(zone)
    return untouched


def find_rejection_candles(df: pd.DataFrame, zone: Zone) -> List[pd.Timestamp]:
    if df is None or df.empty:
        return []
    after = df[df["timestamp"] > zone.start]
    if after.empty:
        return []
    hits: List[pd.Timestamp] = []
    for _, row in after.iterrows():
        if zone.side == "bullish":
            if row["low"] <= zone.high and row["close"] > zone.high:
                hits.append(row["timestamp"])
        else:
            if row["high"] >= zone.low and row["close"] < zone.low:
                hits.append(row["timestamp"])
    return hits
