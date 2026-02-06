from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from indicators.moving_averages import compute_daily_vwap, compute_weekly_vwap
from indicators.volatility import rolling_volatility, classify_volatility, atr_like
from indicators.momentum import roc, trend_strength
from indicators.volume import rvol
from indicators.trend import classify_trend
from data.news_fetcher import combine_news_signals
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
        comment = "Price is above both daily and weekly VWAP (strong bullish posture)."
    elif not daily_above and not weekly_above:
        comment = "Price is below both daily and weekly VWAP (strong bearish posture)."
    elif daily_above and not weekly_above:
        comment = "Price is above daily but below weekly VWAP (short-term strength, long-term weakness)."
    else:
        comment = "Price is below daily but above weekly VWAP (short-term weakness, long-term strength)."

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

    # 4. Early US (US Open Bias) – first part of US session
    us_open_bias = "Neutral"
    if us:
        us_dir = _session_direction(us)
        us_open_bias = _direction_label(us_dir, threshold=atr_threshold)

    # 5. VWAP posture
    vwap_posture = analyze_vwap_posture(df_today)

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

    # Volume
    if last_rvol > 1.2:
        daily_conf += 0.05
    elif last_rvol < 0.8:
        daily_conf -= 0.05

    # News tone (modifier only)
    if news_signal.tone == "risk_on" and daily_bias == "Bullish":
        daily_conf += 0.05
    elif news_signal.tone == "risk_off" and daily_bias == "Bearish":
        daily_conf += 0.05
    elif news_signal.tone in ("risk_on", "risk_off"):
        daily_conf -= 0.05

    # Clamp
    daily_conf = max(0.1, min(0.95, daily_conf))
    us_conf = daily_conf  # for now, mirror daily confidence

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

    explanation = (
        f"Daily Bias is {daily_bias} based on previous day ({prev_label}), "
        f"Asia ({asia_label}), and London ({london_label}) with {daily_bias_source}. "
        f"US Open Bias is {us_open_bias}. Overnight range position is {overnight_label}. "
        f"Gap is {gap_label}. HTF slope bias is {htf_bias}. "
        f"Volatility regime is '{vol_regime}', momentum and ROC aligned at {last_roc:.3f}, "
        f"RVOL at {last_rvol:.2f}. News tone is {news_signal.tone} and only adjusts confidence, not direction. "
        f"Day impact: {' '.join(impact_notes)}"
    )

    if zone_confluence is not None and zone_confluence.bias != "Neutral":
        score = abs(zone_confluence.score)
        bias_boost = min(0.15, 0.04 * score)
        daily_conf += bias_boost
        if score >= 2 and daily_bias != zone_confluence.bias:
            daily_bias = zone_confluence.bias
        explanation += (
            f" HTF zones suggest {zone_confluence.bias} bias "
            f"(score {zone_confluence.score:.2f}; {zone_confluence.notes})."
        )

    daily_conf = max(0.1, min(0.95, daily_conf))
    us_conf = daily_conf

    return BiasSummary(
        daily_bias=daily_bias,
        daily_confidence=daily_conf,
        us_open_bias=us_open_bias,
        us_open_confidence=us_conf,
        explanation=explanation,
        vwap_comment=vwap_posture.comment,
        news_comment=news_signal.explanation,
    )