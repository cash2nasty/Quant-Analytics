from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from indicators.moving_averages import compute_daily_vwap, compute_weekly_vwap
from indicators.volatility import rolling_volatility, classify_volatility
from indicators.momentum import roc, trend_strength
from indicators.volume import rvol
from indicators.trend import classify_trend
from data.news_fetcher import combine_news_signals
from storage.history_manager import SessionStats, BiasSummary
from engines.zones import ZoneConfluence


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

    # 1. Previous day direction
    prev_dir = 0.0
    if not df_prev.empty:
        prev_open = df_prev["open"].iloc[0]
        prev_close = df_prev["close"].iloc[-1]
        prev_dir = prev_close - prev_open
    prev_label = _direction_label(prev_dir)

    # 2. Asia
    asia_label = _direction_label(_session_direction(asia)) if asia else "Neutral"

    # 3. London
    london_label = _direction_label(_session_direction(london)) if london else "Neutral"

    # Combine into Daily Bias (majority vote style)
    votes = [prev_label, asia_label, london_label]
    daily_bias = max(set(votes), key=votes.count)

    # 4. Early US (US Open Bias) – first part of US session
    us_open_bias = "Neutral"
    if us:
        us_dir = _session_direction(us)
        us_open_bias = _direction_label(us_dir)

    # 5. VWAP posture
    vwap_posture = analyze_vwap_posture(df_today)

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

    explanation = (
        f"Daily Bias is {daily_bias} based on previous day ({prev_label}), "
        f"Asia ({asia_label}), and London ({london_label}). "
        f"US Open Bias is {us_open_bias}, with volatility regime '{vol_regime}', "
        f"momentum and ROC aligned at {last_roc:.3f}, RVOL at {last_rvol:.2f}. "
        f"News tone is {news_signal.tone} and only adjusts confidence, not direction."
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