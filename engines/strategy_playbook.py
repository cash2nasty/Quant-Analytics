import datetime as dt
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.session_reference import get_session_windows_for_date
from engines.patterns import PatternSummary
from engines.sessions import SessionStats
from engines.zones import (
    Zone,
    find_rejection_candles,
    is_fvg_inversed,
    is_zone_failed,
    is_zone_touched,
    zone_formed_timestamp,
    zone_liquidity_scores,
)
from indicators.moving_averages import compute_anchored_vwap, compute_daily_vwap, compute_weekly_vwap


OPEN_PATTERN_CANDLE3_STRICTNESS = "balanced"
OPEN_PATTERN_RECLAIM_SPEED_WINDOW_MINUTES = 30.0


def _fmt_ts(value: Optional[pd.Timestamp]) -> Optional[str]:
    if value is None:
        return None
    ts = pd.to_datetime(value)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d %H:%M")


def _first_after(df: pd.DataFrame, mask: pd.Series) -> Optional[Tuple[pd.Timestamp, float]]:
    filtered = df[mask]
    if filtered.empty:
        return None
    row = filtered.iloc[0]
    return pd.to_datetime(row["timestamp"]), float(row["close"])


def _detect_orb_trigger(df: pd.DataFrame, minutes: int) -> Optional[Dict[str, object]]:
    if df is None or df.empty:
        return None
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    date = pd.to_datetime(work["timestamp"].iloc[-1]).date()
    work = work[work["timestamp"].dt.date == date]
    if work.empty:
        return None
    start = pd.Timestamp.combine(date, pd.Timestamp("09:30").time())
    end = start + pd.Timedelta(minutes=minutes)
    opening = work[(work["timestamp"] >= start) & (work["timestamp"] <= end)]
    if opening.empty:
        return None
    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())
    or_size = max(or_high - or_low, 1e-6)

    after = work[work["timestamp"] > end]
    if after.empty:
        return None

    bullish = _first_after(after, after["close"] > (or_high + 0.1 * or_size))
    bearish = _first_after(after, after["close"] < (or_low - 0.1 * or_size))

    if bullish and (not bearish or bullish[0] <= bearish[0]):
        return {
            "name": f"ORB {minutes}m Breakout",
            "direction": "Bullish",
            "time": _fmt_ts(bullish[0]),
            "price": bullish[1],
            "timeframe": "5m",
            "details": f"Close broke above OR high + 10% OR range ({or_high:.2f}).",
        }
    if bearish:
        return {
            "name": f"ORB {minutes}m Breakout",
            "direction": "Bearish",
            "time": _fmt_ts(bearish[0]),
            "price": bearish[1],
            "timeframe": "5m",
            "details": f"Close broke below OR low - 10% OR range ({or_low:.2f}).",
        }
    return None


def _detect_failed_orb_trigger(df: pd.DataFrame, minutes: int) -> Optional[Dict[str, object]]:
    if df is None or df.empty:
        return None
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    date = pd.to_datetime(work["timestamp"].iloc[-1]).date()
    work = work[work["timestamp"].dt.date == date]
    if work.empty:
        return None
    start = pd.Timestamp.combine(date, pd.Timestamp("09:30").time())
    end = start + pd.Timedelta(minutes=minutes)
    opening = work[(work["timestamp"] >= start) & (work["timestamp"] <= end)]
    if opening.empty:
        return None

    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())
    after = work[work["timestamp"] > end]
    if after.empty:
        return None

    break_up = after[after["high"] > or_high]
    if not break_up.empty:
        t0 = pd.to_datetime(break_up.iloc[0]["timestamp"])
        window = after[(after["timestamp"] > t0) & (after["timestamp"] <= t0 + pd.Timedelta(minutes=30))]
        recover = _first_after(window, window["close"] < or_high)
        if recover:
            return {
                "name": f"Failed ORB {minutes}m",
                "direction": "Bearish",
                "time": _fmt_ts(recover[0]),
                "price": recover[1],
                "timeframe": "5m",
                "details": "Break above OR failed and closed back inside within 30 minutes.",
            }

    break_down = after[after["low"] < or_low]
    if not break_down.empty:
        t0 = pd.to_datetime(break_down.iloc[0]["timestamp"])
        window = after[(after["timestamp"] > t0) & (after["timestamp"] <= t0 + pd.Timedelta(minutes=30))]
        recover = _first_after(window, window["close"] > or_low)
        if recover:
            return {
                "name": f"Failed ORB {minutes}m",
                "direction": "Bullish",
                "time": _fmt_ts(recover[0]),
                "price": recover[1],
                "timeframe": "5m",
                "details": "Break below OR failed and closed back inside within 30 minutes.",
            }

    return None


def _detect_vwap_trigger(df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if df is None or df.empty or len(df) < 6:
        return None

    vwap = compute_daily_vwap(df)
    if vwap.empty:
        return None

    closes = df["close"].values
    vwap_vals = vwap.values
    ts_vals = pd.to_datetime(df["timestamp"]).values

    for i in range(1, len(closes) - 4):
        prev = closes[i - 1] - vwap_vals[i - 1]
        curr = closes[i] - vwap_vals[i]

        if prev <= 0 and curr > 0:
            if all(closes[i : i + 4] > vwap_vals[i : i + 4]):
                return {
                    "name": "VWAP Reclaim Hold",
                    "direction": "Bullish",
                    "time": _fmt_ts(pd.Timestamp(ts_vals[i + 3])),
                    "price": float(closes[i + 3]),
                    "timeframe": "1m-5m",
                    "details": "Crossed above VWAP and held 4 consecutive closes above.",
                }

        if prev >= 0 and curr < 0:
            if all(closes[i : i + 4] < vwap_vals[i : i + 4]):
                return {
                    "name": "VWAP Reject Hold",
                    "direction": "Bearish",
                    "time": _fmt_ts(pd.Timestamp(ts_vals[i + 3])),
                    "price": float(closes[i + 3]),
                    "timeframe": "1m-5m",
                    "details": "Crossed below VWAP and held 4 consecutive closes below.",
                }

    return None


def _rolling_atr_like(df: pd.DataFrame, length: int = 20) -> Optional[float]:
    if df is None or df.empty or len(df) < length:
        return None
    try:
        atr_series = (df["high"].astype(float) - df["low"].astype(float)).rolling(length).mean()
        val = atr_series.iloc[-1]
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _volatility_pack(df: pd.DataFrame, length: int = 20) -> Dict[str, object]:
    if df is None or df.empty or len(df) < max(6, length):
        return {
            "std": None,
            "atr_like": None,
            "zscore": None,
            "regime": "normal",
            "regime_note": "Insufficient bars for robust volatility estimate.",
        }

    closes = df["close"].astype(float)
    returns = closes.pct_change().dropna()
    std = float(returns.rolling(length).std().iloc[-1]) if len(returns) >= length else float(returns.std())
    mean = float(closes.rolling(length).mean().iloc[-1])
    sd_px = float(closes.rolling(length).std().iloc[-1])
    atr_like = _rolling_atr_like(df, length=length)

    zscore = None
    if sd_px > 0:
        zscore = float((closes.iloc[-1] - mean) / sd_px)

    hist = returns.rolling(length).std().dropna()
    regime = "normal"
    note = "Volatility is near recent baseline."
    if not hist.empty:
        p30 = float(hist.quantile(0.30))
        p70 = float(hist.quantile(0.70))
        if std > p70:
            regime = "expanded"
            note = "Volatility is expanded versus recent history."
        elif std < p30:
            regime = "compressed"
            note = "Volatility is compressed versus recent history."

    return {
        "std": std,
        "atr_like": atr_like,
        "zscore": zscore,
        "regime": regime,
        "regime_note": note,
        "good_bad_guide": "STDV lower than p30 is compressed, between p30-p70 is normal, above p70 is expanded.",
        "regime_outcomes": {
            "compressed": "Tighter ranges and breakout potential; confirm before chasing.",
            "normal": "Balanced conditions; standard risk rules apply.",
            "expanded": "Large swings and faster invalidation risk; reduce size and widen stops.",
        },
        "rest_of_day_assumption": (
            "Expect larger intraday swings and faster rotations."
            if regime == "expanded"
            else "Expect tighter ranges until a breakout confirms."
            if regime == "compressed"
            else "Expect moderate two-way movement around key levels."
        ),
    }


def _volume_detector(df: pd.DataFrame, length: int = 20) -> Dict[str, object]:
    if df is None or df.empty or "volume" not in df.columns or len(df) < max(5, length):
        return {
            "rvol": None,
            "participation_score": 50.0,
            "spike": False,
            "state": "normal",
            "note": "Insufficient bars for relative volume profile.",
        }

    vol = df["volume"].astype(float)
    avg = vol.rolling(length).mean().iloc[-1]
    if pd.isna(avg) or avg <= 0:
        return {
            "rvol": None,
            "participation_score": 50.0,
            "spike": False,
            "state": "normal",
            "note": "Average volume unavailable for RVOL.",
        }

    rvol = float(vol.iloc[-1] / avg)
    spike = bool(rvol >= 1.6)
    state = "low" if rvol < 0.85 else "high" if rvol > 1.25 else "normal"
    score = max(5.0, min(100.0, 40.0 + 30.0 * min(rvol, 2.0)))
    note = "Volume participation is average."
    if state == "low":
        note = "Below-normal participation; fakeout risk is higher."
    elif state == "high":
        note = "Above-normal participation; move quality is stronger."

    return {
        "rvol": rvol,
        "participation_score": round(score, 1),
        "spike": spike,
        "state": state,
        "note": note,
        "good_bad_guide": "RVOL < 0.85 is weak participation, 0.85-1.25 is normal, > 1.25 is strong.",
        "rest_of_day_assumption": (
            "Moves can follow through if structure aligns."
            if state == "high"
            else "Breakout failure risk is elevated without fresh participation."
            if state == "low"
            else "Need structure confirmation because participation is average."
        ),
    }


def _momentum_prediction(
    df: pd.DataFrame,
    ny_direction: str,
    volatility: Dict[str, object],
    volume: Dict[str, object],
) -> Dict[str, object]:
    if df is None or df.empty or len(df) < 25:
        return {
            "prob_up": 50.0,
            "prob_down": 50.0,
            "predicted": "Neutral",
            "confidence": 0.0,
            "note": "Insufficient bars for momentum model.",
        }

    closes = df["close"].astype(float)
    ret5 = float((closes.iloc[-1] / closes.iloc[-6]) - 1.0) if len(closes) >= 6 else 0.0
    ret15 = float((closes.iloc[-1] / closes.iloc[-16]) - 1.0) if len(closes) >= 16 else 0.0
    y = closes.tail(20).values
    x = np.arange(len(y))
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 5 else 0.0

    slope_norm = slope / max(abs(closes.iloc[-1]) * 0.0005, 1e-6)
    score = 0.0
    score += max(-2.0, min(2.0, ret5 * 300.0))
    score += max(-2.0, min(2.0, ret15 * 150.0))
    score += max(-2.0, min(2.0, slope_norm))

    if str(volume.get("state", "normal")) == "high":
        score += 0.5
    elif str(volume.get("state", "normal")) == "low":
        score -= 0.5

    if str(volatility.get("regime", "normal")) == "expanded":
        score *= 0.85

    prob_up = 50.0 + max(-40.0, min(40.0, score * 10.0))
    prob_up = max(1.0, min(99.0, prob_up))
    prob_down = 100.0 - prob_up

    predicted = "Bullish" if prob_up > 55 else "Bearish" if prob_up < 45 else "Neutral"
    confidence = abs(prob_up - 50.0) * 2.0
    note = "Momentum is balanced."
    if predicted == "Bullish":
        note = "Short-horizon momentum leans bullish."
    elif predicted == "Bearish":
        note = "Short-horizon momentum leans bearish."

    if ny_direction in {"Bullish", "Bearish"} and predicted == ny_direction:
        note += " Momentum agrees with session bias."
    elif ny_direction in {"Bullish", "Bearish"} and predicted in {"Bullish", "Bearish"}:
        note += " Momentum conflicts with session bias."

    return {
        "prob_up": round(prob_up, 1),
        "prob_down": round(prob_down, 1),
        "predicted": predicted,
        "confidence": round(confidence, 1),
        "note": note,
        "good_bad_guide": "Confidence < 15 is weak, 15-35 is moderate, > 35 is strong momentum signal.",
        "rest_of_day_assumption": (
            "Continuation odds are stronger while trigger structure remains valid."
            if confidence >= 35.0
            else "Momentum edge is modest; rely more on confluence and risk gates."
        ),
    }


def _vwap_mean_reversion_vs_expansion(
    df: pd.DataFrame,
    daily_vwap: Optional[float],
    windows: Dict[str, Dict[str, dt.datetime]],
    volume: Dict[str, object],
    momentum: Dict[str, object],
    volatility: Dict[str, object],
) -> Dict[str, object]:
    if df is None or df.empty or daily_vwap is None:
        return {
            "session": "NY",
            "price_vs_vwap": "n/a",
            "mean_reversion_prob": 50.0,
            "expansion_prob": 50.0,
            "directional_expansion": "Neutral",
            "note": "VWAP unavailable for probability model.",
        }

    us_start = windows.get("US", {}).get("start")
    us_end = windows.get("US", {}).get("end")
    work = df.copy()
    if us_start is not None:
        work = work[work["timestamp"] >= us_start]
    if us_end is not None:
        work = work[work["timestamp"] <= us_end]
    if work.empty:
        work = df.tail(60)

    closes = work["close"].astype(float)
    above_ratio = float((closes > daily_vwap).mean()) if len(closes) else 0.5
    last = float(closes.iloc[-1])

    stretch = float(volatility.get("zscore", 0.0) or 0.0)
    rvol = float(volume.get("rvol", 1.0) or 1.0)
    mom_up = float(momentum.get("prob_up", 50.0) or 50.0)

    expansion = 50.0
    if last >= daily_vwap:
        expansion += (above_ratio - 0.5) * 40.0
        expansion += (mom_up - 50.0) * 0.35
    else:
        expansion += ((1.0 - above_ratio) - 0.5) * 40.0
        expansion += ((100.0 - mom_up) - 50.0) * 0.35

    expansion += (min(max(rvol, 0.5), 2.0) - 1.0) * 12.0
    expansion -= max(0.0, abs(stretch) - 1.8) * 10.0

    expansion = max(5.0, min(95.0, expansion))
    mean_rev = 100.0 - expansion
    vs = "Above" if last > daily_vwap else "Below" if last < daily_vwap else "At"
    directional = "Bullish" if last >= daily_vwap and expansion >= mean_rev else "Bearish" if last < daily_vwap and expansion >= mean_rev else "Neutral"

    note = (
        "NY VWAP model favors expansion from current side."
        if expansion >= mean_rev
        else "NY VWAP model favors mean reversion back to VWAP."
    )

    return {
        "session": "NY",
        "price_vs_vwap": vs,
        "mean_reversion_prob": round(mean_rev, 1),
        "expansion_prob": round(expansion, 1),
        "directional_expansion": directional,
        "note": note,
    }


def _vwap_strength_after_dip(df: pd.DataFrame, daily_vwap: Optional[float]) -> Dict[str, object]:
    base = {
        "score": 0.0,
        "label": "Not Active",
        "reclaim_time": "n/a",
        "entry_timing": "Wait for reclaim sequence.",
        "note": "No active dip-and-reclaim pattern.",
        "rest_of_day_assumption": "No actionable VWAP reclaim expectation yet.",
    }
    if df is None or df.empty or daily_vwap is None or len(df) < 10:
        return base

    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    rel = work["close"].astype(float) - float(daily_vwap)

    reclaim_idx = None
    for i in range(2, len(rel)):
        if rel.iloc[i - 1] <= 0 and rel.iloc[i] > 0:
            reclaim_idx = i
    if reclaim_idx is None:
        return base

    segment = work.iloc[max(0, reclaim_idx - 6) : min(len(work), reclaim_idx + 10)]
    closes = segment["close"].astype(float).values
    above = np.mean(closes > daily_vwap) if len(closes) else 0.0
    reclaim_bar = work.iloc[reclaim_idx]
    reclaim_time = _fmt_ts(pd.to_datetime(reclaim_bar["timestamp"])) or "n/a"

    forward = work.iloc[reclaim_idx : min(len(work), reclaim_idx + 8)]
    advance = max(0.0, float(forward["high"].max()) - float(reclaim_bar["close"])) if not forward.empty else 0.0
    pullback = max(0.0, float(reclaim_bar["close"]) - float(forward["low"].min())) if not forward.empty else 0.0

    score = 35.0 + above * 35.0 + min(20.0, advance * 4.0) - min(15.0, pullback * 4.0)
    score = max(0.0, min(100.0, score))
    label = "Weak" if score < 45 else "Moderate" if score < 70 else "Strong"
    timing = "First reclaim hold is acceptable." if score >= 70 else "Prefer second retest confirmation." if score >= 45 else "Wait for better reclaim quality."
    note = (
        "Reclaim held with follow-through above VWAP."
        if score >= 70
        else "Reclaim is present but needs cleaner confirmation."
        if score >= 45
        else "Reclaim quality is weak and vulnerable to failure."
    )

    return {
        "score": round(score, 1),
        "score_out_of": f"{round(score, 1)}/100",
        "label": label,
        "reclaim_time": reclaim_time,
        "entry_timing": timing,
        "note": note,
        "good_bad_guide": "<45 weak, 45-69 moderate, >=70 strong reclaim quality.",
        "rest_of_day_assumption": (
            "Expect continuation attempts above VWAP while reclaim structure remains intact."
            if score >= 70
            else "Expect choppy retests around VWAP until cleaner confirmation appears."
            if score >= 45
            else "Expect higher risk of reclaim failure and rotation back toward VWAP/under it."
        ),
    }


def _build_expectation_summaries(
    vwap_dip: Dict[str, object],
    performance: Dict[str, object],
    volatility: Dict[str, object],
    volume: Dict[str, object],
    momentum: Dict[str, object],
) -> Dict[str, str]:
    def _expand(msg: object) -> str:
        text = str(msg or "n/a").strip()
        if not text or text.lower() == "n/a":
            return "No clear expectation is available yet because the underlying signals are incomplete."
        return f"In plain terms: {text} Keep position size disciplined until this expectation is confirmed by price action."

    return {
        "vwap_dip": _expand(vwap_dip.get("rest_of_day_assumption", "n/a")),
        "performance": _expand(performance.get("rest_of_day_assumption", "n/a")),
        "volatility": _expand(volatility.get("rest_of_day_assumption", "n/a")),
        "volume": _expand(volume.get("rest_of_day_assumption", "n/a")),
        "momentum": _expand(momentum.get("rest_of_day_assumption", "n/a")),
    }


def _risk_engine(
    last_price: float,
    direction: str,
    reference_rows: List[Dict[str, object]],
    target_ladder: List[Dict[str, object]],
    volatility: Dict[str, object],
    volume: Dict[str, object],
    momentum: Dict[str, object],
    vwap_prob: Dict[str, object],
) -> Dict[str, object]:
    def _pct(value: float) -> float:
        return round(max(5.0, min(95.0, value)), 1)

    atr_like = float(volatility.get("atr_like", 1.0) or 1.0)
    side = "Long" if direction == "Bullish" else "Short" if direction == "Bearish" else "Wait"
    mom_up = float(momentum.get("prob_up", 50.0) or 50.0)
    mom_down = float(momentum.get("prob_down", 50.0) or 50.0)
    mom_conf = float(momentum.get("confidence", 0.0) or 0.0)
    rvol = float(volume.get("rvol", 0.0) or 0.0)
    participation = float(volume.get("participation_score", 50.0) or 50.0)
    exp_prob = float(vwap_prob.get("expansion_prob", 50.0) or 50.0)
    mean_prob = float(vwap_prob.get("mean_reversion_prob", 50.0) or 50.0)

    directional_prob = mom_up if side == "Long" else mom_down if side == "Short" else max(mom_up, mom_down)
    momentum_align_conf = _pct(directional_prob * 0.7 + mom_conf * 0.6)
    volume_align_conf = _pct(30.0 + (rvol * 35.0) + (participation - 50.0) * 0.4)
    vwap_edge = exp_prob - mean_prob
    vwap_align_conf = _pct(45.0 + abs(vwap_edge) * 1.1) if side == "Wait" else _pct(
        50.0 + vwap_edge * 0.9 + (directional_prob - 50.0) * 0.3
    )

    def _probability_label(pct: float) -> str:
        if pct >= 75.0:
            return "Highly probable (Good)"
        if pct >= 60.0:
            return "Probable (Good)"
        if pct >= 45.0:
            return "Moderately probable"
        if pct >= 30.0:
            return "Unlikely (Bad)"
        return "Highly unlikely (Bad)"

    direction_word = "upside" if side == "Long" else "downside" if side == "Short" else "either direction"
    momentum_threshold = 57.0 if side in {"Long", "Short"} else 55.0
    spread_target = 8.0
    strengthening_alignments = [
        (
            f"Momentum alignment: we want directional probability above {momentum_threshold:.0f}% with at least "
            f"a {spread_target:.0f}% spread versus the opposite side, and confidence above 30. "
            f"Current state is up {mom_up:.1f}% vs down {mom_down:.1f}% with confidence {mom_conf:.1f}. "
            f"Chance of this improving soon: {momentum_align_conf:.1f}% ({_probability_label(momentum_align_conf)})."
        ),
        (
            f"Participation alignment: we want RVOL >= 1.10 and participation score >= 58 so follow-through is less likely to stall. "
            f"Current state is RVOL {rvol:.2f} and participation {participation:.1f}/100. "
            f"Chance of this strengthening: {volume_align_conf:.1f}% ({_probability_label(volume_align_conf)})."
        ),
        (
            f"VWAP scenario alignment: we want expansion probability to separate from reversion by at least 8 points in the intended direction. "
            f"Current state is expansion {exp_prob:.1f}% vs reversion {mean_prob:.1f}% (spread {vwap_edge:.1f}). "
            f"Chance of cleaner VWAP alignment: {vwap_align_conf:.1f}% ({_probability_label(vwap_align_conf)})."
        ),
    ]

    if side == "Wait":
        directional_hint = "Bullish" if mom_up > mom_down + 3.0 else "Bearish" if mom_down > mom_up + 3.0 else "Neutral"
        confirmations = [
            "Wait for a directional trigger (OR break/retest, reclaim, or rejection setup) before sizing.",
            (
                f"Wait for momentum separation to widen (now up {mom_up:.1f}% vs down {mom_down:.1f}%, hint: {directional_hint})."
            ),
            (
                f"Wait for participation to improve (RVOL now {rvol:.2f}; preferred >= 1.00)."
                if rvol < 1.0
                else f"Participation is acceptable (RVOL {rvol:.2f}); wait for price-structure confirmation."
            ),
            (
                f"Wait for VWAP probabilities to tilt more clearly (expansion {exp_prob:.1f}% vs reversion {mean_prob:.1f}%)."
            ),
        ]
        return {
            "side": "Wait",
            "entry": round(last_price, 2),
            "stop": None,
            "target": None,
            "suggested_target": None,
            "minimum_target": None,
            "maximum_target": None,
            "risk_points": None,
            "reward_points": None,
            "rr": None,
            "size_contracts_10k": 0,
            "quality_score": 35.0,
            "status": "Caution",
            "notes": "No executable setup yet. Risk engine is waiting for confirmation before publishing stop/target.",
            "alignment_factors": [
                f"Momentum split is not decisive enough yet (up {mom_up:.1f}% vs down {mom_down:.1f}%).",
                f"RVOL is {rvol:.2f}, so participation is not strong enough for confident sizing.",
                f"VWAP tilt is mixed (expansion {exp_prob:.1f}% vs reversion {mean_prob:.1f}%).",
            ],
            "trade_summary": "No trade is suggested yet because direction and confirmation quality are not aligned.",
            "expected_behavior": "Expect additional back-and-forth until one side clearly holds structure and VWAP context.",
            "post_trade_assumption": "After confirmation appears, the engine will publish stop and target levels with RR >= 0.90.",
            "strengthening_alignments": strengthening_alignments,
            "waiting_for_confirmations": confirmations,
        }

    ref = reference_rows[0] if reference_rows else None
    if ref:
        zone_low = float(ref.get("Price Low", last_price))
        zone_high = float(ref.get("Price High", last_price))
    else:
        zone_low = last_price - atr_like
        zone_high = last_price + atr_like

    stop = (zone_low - 0.25 * atr_like) if side == "Long" else (zone_high + 0.25 * atr_like)

    candidate_targets: List[float] = []
    for t in target_ladder:
        px = float(t.get("Price", last_price))
        if side == "Long" and px > last_price:
            candidate_targets.append(px)
        if side == "Short" and px < last_price:
            candidate_targets.append(px)

    if not candidate_targets:
        fallback_min = last_price + (1.0 * atr_like if side == "Long" else -1.0 * atr_like)
        fallback_max = last_price + (2.4 * atr_like if side == "Long" else -2.4 * atr_like)
        candidate_targets = [fallback_min, fallback_max]

    if side == "Long":
        candidate_targets = sorted(set(candidate_targets))
    else:
        candidate_targets = sorted(set(candidate_targets), reverse=True)

    suggested_target = candidate_targets[0]
    minimum_target = min(candidate_targets)
    maximum_target = max(candidate_targets)
    target = suggested_target

    risk_points = max(abs(last_price - stop), 0.25)
    reward_points = max(abs(target - last_price), 0.0)
    rr = reward_points / risk_points if risk_points > 0 else 0.0

    # Enforce hard RR floor: do not suggest a trade if best available target is < 0.90R.
    if rr < 0.9:
        rr_candidates = []
        for px in candidate_targets:
            rew = abs(px - last_price)
            rr_candidates.append((px, rew / risk_points if risk_points > 0 else 0.0))
        valid_rr = [pair for pair in rr_candidates if pair[1] >= 0.9]
        if valid_rr:
            target = valid_rr[0][0]
            reward_points = abs(target - last_price)
            rr = reward_points / risk_points if risk_points > 0 else 0.0
            suggested_target = target
        else:
            confirmations = [
                f"Current stop distance is {risk_points:.2f} points, but available target path does not yet provide 0.90R.",
                "Wait for either a tighter invalidation level (smaller stop) or a cleaner expansion target.",
                "Require fresh trigger confirmation before accepting a lower-quality payoff profile.",
            ]
            return {
                "side": "Wait",
                "entry": round(last_price, 2),
                "stop": None,
                "target": None,
                "suggested_target": round(suggested_target, 2),
                "minimum_target": round(minimum_target, 2),
                "maximum_target": round(maximum_target, 2),
                "risk_points": round(risk_points, 2),
                "reward_points": round(reward_points, 2),
                "rr": round(rr, 2),
                "size_contracts_10k": 0,
                "quality_score": 42.0,
                "status": "Caution",
                "notes": "Trade blocked: expected reward is below 0.90R, so the setup is not worth taking yet.",
                "alignment_factors": [
                    f"Directional bias is {direction}, but reward path is too short for current stop distance.",
                    f"Momentum confidence is {mom_conf:.1f} and participation score is {participation:.1f}/100.",
                    f"VWAP context is expansion {exp_prob:.1f}% vs reversion {mean_prob:.1f}%.",
                ],
                "trade_summary": "The setup has directional context, but the payoff profile is currently unfavorable.",
                "expected_behavior": "Unless range expands or stop can be tightened, price may not provide enough reward relative to risk.",
                "post_trade_assumption": "If a larger directional move opens up, the engine will reactivate this setup once RR reaches at least 0.90.",
                "strengthening_alignments": strengthening_alignments,
                "waiting_for_confirmations": confirmations,
            }

    # NQ-like point value for sizing guidance.
    point_value = 20.0
    risk_budget = 50.0
    contracts = int(max(0.0, math.floor(risk_budget / max(risk_points * point_value, 1e-6))))

    quality = 45.0
    quality += max(-10.0, min(15.0, (rr - 1.2) * 18.0))
    quality += (participation - 50.0) * 0.20
    quality += (mom_conf - 20.0) * 0.10
    if direction == "Bullish":
        quality += (mom_up - 50.0) * 0.15
    else:
        quality += (mom_down - 50.0) * 0.15

    quality += (exp_prob - mean_prob) * 0.10

    quality = max(0.0, min(100.0, quality))
    status = "High" if quality >= 70 else "Moderate" if quality >= 55 else "Caution"
    notes = (
        "Risk model supports execution with current parameters."
        if quality >= 70
        else "Execution is possible but requires strict trigger confirmation."
        if quality >= 55
        else "Risk quality is weak; wait for better alignment."
    )

    alignment_factors = [
        f"Direction and momentum alignment: side {side}, up {mom_up:.1f}% vs down {mom_down:.1f}%, confidence {mom_conf:.1f}.",
        f"Participation context: RVOL {rvol:.2f}, participation score {participation:.1f}/100.",
        f"VWAP scenario tilt: expansion {exp_prob:.1f}% vs reversion {mean_prob:.1f}%.",
        f"Payoff profile: risk {risk_points:.2f} pts vs reward {reward_points:.2f} pts (R:R {rr:.2f}).",
    ]

    trade_summary = (
        f"Risk engine suggests a {side.lower()} setup because directional signals and payoff are aligned at {rr:.2f}R."
    )
    expected_behavior = (
        "If this setup is valid, price should hold away from the stop zone and progress toward the suggested target in staged pushes."
    )
    post_trade_assumption = (
        "After entry, assume continuation remains valid only while trigger-side structure is respected; violation of that structure invalidates the idea."
    )

    return {
        "side": side,
        "entry": round(last_price, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "suggested_target": round(suggested_target, 2),
        "minimum_target": round(minimum_target, 2),
        "maximum_target": round(maximum_target, 2),
        "risk_points": round(risk_points, 2),
        "reward_points": round(reward_points, 2),
        "rr": round(rr, 2),
        "size_contracts_10k": contracts,
        "quality_score": round(quality, 1),
        "status": status,
        "notes": notes,
        "alignment_factors": alignment_factors,
        "trade_summary": trade_summary,
        "expected_behavior": expected_behavior,
        "post_trade_assumption": post_trade_assumption,
        "strengthening_alignments": strengthening_alignments,
        "waiting_for_confirmations": [],
    }


def _performance_metrics(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty or len(df) < 12:
        return {
            "max_drawdown_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "drawdown_duration_bars": 0,
            "expected_value_bps": 0.0,
            "expected_value_raw": 0.0,
            "raw_sharpe": 0.0,
            "annualized_sharpe": 0.0,
            "notes": "Insufficient bars for drawdown/EV/Sharpe metrics.",
        }

    closes = df["close"].astype(float)
    rets = closes.pct_change().dropna()
    if rets.empty:
        return {
            "max_drawdown_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "drawdown_duration_bars": 0,
            "expected_value_bps": 0.0,
            "expected_value_raw": 0.0,
            "raw_sharpe": 0.0,
            "annualized_sharpe": 0.0,
            "notes": "Returns unavailable for performance metrics.",
        }

    equity = (1.0 + rets).cumprod()
    peaks = equity.cummax()
    dd = equity / peaks - 1.0
    max_dd = float(dd.min()) * 100.0
    curr_dd = float(dd.iloc[-1]) * 100.0

    dd_duration = 0
    cur = 0
    for val in dd.values:
        if val < 0:
            cur += 1
            dd_duration = max(dd_duration, cur)
        else:
            cur = 0

    mean_r = float(rets.mean())
    std_r = float(rets.std())
    raw_sharpe = (mean_r / std_r) if std_r > 1e-12 else 0.0

    # Approximate annualization based on inferred bars/day and 252 trading days.
    periods = max(len(rets), 1)
    ann_factor = math.sqrt(min(252.0 * max(periods / 1.0, 1.0), 252.0 * 390.0))
    ann_sharpe = raw_sharpe * ann_factor

    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    p_win = float(len(wins) / len(rets))
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = abs(float(losses.mean())) if not losses.empty else 0.0
    ev_raw = p_win * avg_win - (1.0 - p_win) * avg_loss

    notes = "Metrics are intraday-return based and should be validated against trade-level backtests."
    return {
        "max_drawdown_pct": round(max_dd, 2),
        "current_drawdown_pct": round(curr_dd, 2),
        "drawdown_duration_bars": int(dd_duration),
        "expected_value_bps": round(ev_raw * 10000.0, 2),
        "expected_value_raw": round(ev_raw, 6),
        "raw_sharpe": round(raw_sharpe, 4),
        "annualized_sharpe": round(ann_sharpe, 3),
        "notes": notes,
        "good_bad_guide": {
            "max_drawdown_pct": "Closer to 0 is better. Deep negatives indicate fragile behavior.",
            "expected_value_bps": "Positive is good edge, near zero is weak edge, negative is bad edge.",
            "raw_sharpe": "~0.5 modest, ~1.0 good, >1.5 strong, below 0 weak.",
        },
        "rest_of_day_assumption": (
            "Recent profile is unstable; prioritize defense and selective entries."
            if max_dd <= -3.0 or ev_raw <= 0
            else "Recent profile is constructive; maintain disciplined execution."
        ),
    }


def _entry_confidence_tier(score: float) -> str:
    if score >= 75.0:
        return "High"
    if score >= 55.0:
        return "Moderate"
    return "Low"


def _trade_decision_bucket(trade_today: str) -> str:
    v = str(trade_today or "").strip().lower()
    if v in {"yes", "tradeable whipsaw"}:
        return "Yes"
    if v in {"no", "untradeable whipsaw"}:
        return "No"
    return "Wait"


def _attach_entry_confidence(
    entry_styles: List[Dict[str, object]],
    risk_engine: Dict[str, object],
    momentum: Dict[str, object],
    volume: Dict[str, object],
    vwap_prob: Dict[str, object],
    ny_direction: str,
    trade_today: str,
    primary_trigger: Optional[Dict[str, object]],
    whipsaw_risk: bool,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    trade_bucket = _trade_decision_bucket(trade_today)
    for row in entry_styles:
        score = 40.0
        reasons: List[str] = []
        action = str(row.get("Action", "Wait"))
        reaction_score = float(row.get("Reaction Score", 0.0) or 0.0)
        status = str(row.get("Status", "Fresh"))

        score += (float(risk_engine.get("quality_score", 50.0)) - 50.0) * 0.35
        reasons.append(f"Risk quality {float(risk_engine.get('quality_score', 50.0)):.1f}/100")

        if reaction_score > 0:
            score += (reaction_score - 50.0) * 0.30
            reasons.append(f"Zone reaction score {reaction_score:.1f}")

        mom_conf = float(momentum.get("confidence", 0.0) or 0.0)
        mom_pred = str(momentum.get("predicted", "Neutral"))
        if action == "Long":
            score += (float(momentum.get("prob_up", 50.0)) - 50.0) * 0.25
            if mom_pred == "Bullish":
                reasons.append("Momentum agrees with long direction")
        elif action == "Short":
            score += (float(momentum.get("prob_down", 50.0)) - 50.0) * 0.25
            if mom_pred == "Bearish":
                reasons.append("Momentum agrees with short direction")
        score += (mom_conf - 20.0) * 0.08

        score += (float(volume.get("participation_score", 50.0)) - 50.0) * 0.20
        reasons.append(f"Volume participation {float(volume.get('participation_score', 50.0)):.1f}/100")

        exp = float(vwap_prob.get("expansion_prob", 50.0) or 50.0)
        rev = float(vwap_prob.get("mean_reversion_prob", 50.0) or 50.0)
        if action == "Long" and ny_direction == "Bullish":
            score += (exp - rev) * 0.10
        elif action == "Short" and ny_direction == "Bearish":
            score += (exp - rev) * 0.10
        else:
            score -= 4.0

        if status == "Invalidated":
            score -= 35.0
            reasons.append("Confluence invalidated")
        elif status == "Retested":
            score -= 6.0
            reasons.append("Confluence already retested")

        if whipsaw_risk:
            score -= 8.0
            reasons.append("Whipsaw risk penalty")

        if trade_bucket == "No":
            score -= 25.0
            reasons.append("Trade decision is No")
        elif trade_bucket == "Wait":
            score -= 10.0
            reasons.append("Trade decision is Wait")

        trigger_aligned = False
        if primary_trigger is not None:
            trigger_dir = str(primary_trigger.get("direction", "Neutral"))
            if (action == "Long" and trigger_dir == "Bullish") or (action == "Short" and trigger_dir == "Bearish"):
                trigger_aligned = True
                score += 10.0
                reasons.append(f"Primary trigger alignment: {primary_trigger.get('name', 'n/a')}")
            elif trigger_dir in {"Bullish", "Bearish"}:
                score -= 6.0
                reasons.append("Primary trigger conflicts with entry side")

        # Option 2 tuning: reward strong trigger + reaction confluence more aggressively.
        if trigger_aligned:
            if reaction_score >= 70.0:
                score += 14.0
                reasons.append("Strong trigger+reaction alignment boost")
            elif reaction_score >= 55.0:
                score += 10.0
                reasons.append("Moderate trigger+reaction alignment boost")
            elif reaction_score >= 40.0:
                score += 6.0
                reasons.append("Early trigger+reaction alignment boost")

        score = max(0.0, min(100.0, score))
        tier = _entry_confidence_tier(score)
        row_out = dict(row)
        row_out["Entry Confidence"] = round(score, 1)
        row_out["Confidence Tier"] = tier
        row_out["Confidence Reasoning"] = "; ".join(reasons[:6])
        out.append(row_out)
    return out


def _to_ts(value: object) -> Optional[pd.Timestamp]:
    if value in (None, "", "n/a"):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _session_marker_for_time(
    ts: Optional[pd.Timestamp],
    windows: Optional[Dict[str, Dict[str, dt.datetime]]],
) -> Tuple[str, str]:
    if ts is None or windows is None:
        return ("⚪", "Unknown")

    ts_val = pd.Timestamp(ts)
    asia = windows.get("Asia", {})
    london = windows.get("London", {})
    us = windows.get("US", {})

    asia_start = pd.to_datetime(asia.get("start")) if asia.get("start") else None
    asia_end = pd.to_datetime(asia.get("end")) if asia.get("end") else None
    london_start = pd.to_datetime(london.get("start")) if london.get("start") else None
    london_end = pd.to_datetime(london.get("end")) if london.get("end") else None
    us_start = pd.to_datetime(us.get("start")) if us.get("start") else None
    us_end = pd.to_datetime(us.get("end")) if us.get("end") else None

    power_hour_start = pd.Timestamp.combine(ts_val.date(), dt.time(14, 0))
    power_hour_end = pd.Timestamp.combine(ts_val.date(), dt.time(16, 59, 59))

    pre_asia_start = None
    if asia_start is not None:
        pre_asia_start = pd.Timestamp.combine(asia_start.date(), dt.time(18, 0))

    # Priority order matters: power-hour should override generic NY tagging.
    if power_hour_start <= ts_val <= power_hour_end:
        return ("⚫", "Power Hour")
    if pre_asia_start is not None and asia_start is not None and pre_asia_start <= ts_val < asia_start:
        return ("🔴", "Pre Asia")
    if asia_end is not None and london_start is not None and asia_end <= ts_val < london_start:
        return ("⚪", "Pre London")
    if london_end is not None and us_start is not None and london_end <= ts_val < us_start:
        return ("🟤", "Pre NY")

    if asia_start is not None and asia_end is not None and asia_start <= ts_val <= asia_end:
        return ("🟡", "Asia")
    if london_start is not None and london_end is not None and london_start <= ts_val <= london_end:
        return ("🟣", "London")
    if us_start is not None and us_end is not None and us_start <= ts_val <= us_end:
        return ("🔵", "NY")
    return ("⚪", "Off Session")


def _evaluate_entry_executions(
    df: pd.DataFrame,
    entry_styles: List[Dict[str, object]],
    risk_engine: Dict[str, object],
    windows: Optional[Dict[str, Dict[str, dt.datetime]]] = None,
) -> List[Dict[str, object]]:
    if df is None or df.empty:
        return []

    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    atr_like = _rolling_atr_like(work, length=20) or 2.0
    trade_date = pd.to_datetime(work["timestamp"].iloc[-1]).date()
    use_windows = windows or get_session_windows_for_date(trade_date)
    us_end = use_windows.get("US", {}).get("end") if use_windows else None
    trading_day_ended = bool(us_end is not None and pd.to_datetime(work["timestamp"].max()) >= pd.to_datetime(us_end))

    results: List[Dict[str, object]] = []

    for row in entry_styles:
        confluence = str(row.get("Confluence", "n/a"))
        action = str(row.get("Action", "Wait"))
        suggested = str(row.get("Suggested Entry", "First Tap"))
        suggested_key = suggested.lower()
        preview_use_mid = "midline" in suggested_key or "split" in suggested_key
        preview_entry_price = (
            float(row.get("Midline Price", 0.0)) if preview_use_mid else float(row.get("First Tap Price", 0.0))
        )
        suggested_ts = (
            _to_ts(row.get("Suggested Time"))
            or _to_ts(row.get("Reaction Time"))
            or _to_ts(row.get("Tap Time"))
            or _to_ts(row.get("Midline Time"))
            or _to_ts(row.get("Formed Time"))
        )
        session_marker, suggested_session = _session_marker_for_time(suggested_ts, use_windows)
        suggested_time_str = _fmt_ts(suggested_ts) or "n/a"
        confluence_range = str(row.get("Confluence Range", "n/a"))
        formed_time = str(row.get("Formed Time", "n/a"))
        min_target = row.get("Min Target", row.get("Minimum Target Price", "n/a"))
        max_target = row.get("Max Target", row.get("Maximum Target Price", "n/a"))
        meta = {
            "Session Marker": f"{session_marker} {suggested_session}",
            "Suggested Time": suggested_time_str,
            "Entry ETA (HH:MM)": row.get("Entry ETA (HH:MM)", "n/a"),
            "Entry Window": row.get("Entry Window", "n/a"),
            "Entry ETA Detail": row.get("Entry ETA Detail", "n/a"),
            "Confluence Range": confluence_range,
            "Confluence Formed Time": formed_time,
            "Entry Confidence": row.get("Entry Confidence", "n/a"),
            "Confidence Tier": row.get("Confidence Tier", "n/a"),
            "Confidence Reasoning": row.get("Confidence Reasoning", row.get("Reason", "n/a")),
            "Min Target": min_target,
            "Max Target": max_target,
        }

        if action not in {"Long", "Short"}:
            results.append(
                {
                    **meta,
                    "Confluence": confluence,
                    "Action": action,
                    "Suggested Entry": suggested,
                    "Entry Price": round(preview_entry_price, 2) if preview_entry_price > 0 else "n/a",
                    "Risk Pts(Ticks)": "n/a",
                    "Target Pts(Ticks)": "n/a",
                    "RR": "n/a",
                    "Risk $ (1 contract)": "n/a",
                    "Executed": "No",
                    "Execution Time": "n/a",
                    "Outcome": "Skipped",
                    "Exit Time": "n/a",
                    "Exit Price": "n/a",
                    "Why": "No directional action for this entry suggestion.",
                }
            )
            continue

        is_long = action == "Long"
        use_mid = "midline" in suggested.lower() or "split" in suggested.lower()
        entry_price = float(row.get("Midline Price", 0.0)) if use_mid else float(row.get("First Tap Price", 0.0))

        # Build stop/target from suggestion first, then fall back to risk engine.
        if is_long:
            stop = min(entry_price - atr_like * 0.6, float(row.get("First Tap Price", entry_price)) - atr_like * 0.25)
            tgt = float(row.get("Preferred Target Price") or risk_engine.get("target") or (entry_price + atr_like * 1.5))
        else:
            stop = max(entry_price + atr_like * 0.6, float(row.get("First Tap Price", entry_price)) + atr_like * 0.25)
            tgt = float(row.get("Preferred Target Price") or risk_engine.get("target") or (entry_price - atr_like * 1.5))

        exec_time = _to_ts(row.get("Midline Time")) if use_mid else _to_ts(row.get("Tap Time"))
        if exec_time is None:
            # Fallback: if style already marked as hit, use formed time as proxy.
            hit = str(row.get("Midline Hit" if use_mid else "Tap Hit", "No")) == "Yes"
            if hit:
                exec_time = _to_ts(row.get("Formed Time"))

        if exec_time is None:
            pending_risk_points = max(0.25, abs(entry_price - stop))
            pending_ticks = int(round(pending_risk_points / 0.25))
            pending_target_points = max(0.0, abs(tgt - entry_price))
            pending_target_ticks = int(round(pending_target_points / 0.25))
            pending_risk_dollars = pending_risk_points * 20.0
            pending_outcome = "Unfilled" if trading_day_ended else "Pending"
            pending_why = (
                "Entry level was not touched before the trading day ended."
                if trading_day_ended
                else "Entry level has not been touched yet."
            )
            results.append(
                {
                    **meta,
                    "Confluence": confluence,
                    "Action": action,
                    "Suggested Entry": suggested,
                    "Entry Price": round(entry_price, 2),
                    "Risk (Ticks)": pending_ticks,
                    "Target (Ticks)": pending_target_ticks,
                    "Risk Pts(Ticks)": f"{pending_risk_points:.2f} ({pending_ticks})",
                    "Target Pts(Ticks)": f"{pending_target_points:.2f} ({pending_target_ticks})",
                    "RR": round((pending_target_points / pending_risk_points) if pending_risk_points > 0 else 0.0, 2),
                    "Risk $ (1 contract)": round(pending_risk_dollars, 2),
                    "Executed": "No",
                    "Execution Time": "n/a",
                    "Outcome": pending_outcome,
                    "Exit Time": "n/a",
                    "Exit Price": "n/a",
                    "Why": pending_why,
                }
            )
            continue

        future = work[work["timestamp"] >= exec_time]
        if future.empty:
            open_risk_points = max(0.25, abs(entry_price - stop))
            open_ticks = int(round(open_risk_points / 0.25))
            open_target_points = max(0.0, abs(tgt - entry_price))
            open_target_ticks = int(round(open_target_points / 0.25))
            open_risk_dollars = open_risk_points * 20.0
            results.append(
                {
                    **meta,
                    "Confluence": confluence,
                    "Action": action,
                    "Suggested Entry": suggested,
                    "Entry Price": round(entry_price, 2),
                    "Risk (Ticks)": open_ticks,
                    "Target (Ticks)": open_target_ticks,
                    "Risk Pts(Ticks)": f"{open_risk_points:.2f} ({open_ticks})",
                    "Target Pts(Ticks)": f"{open_target_points:.2f} ({open_target_ticks})",
                    "RR": round((open_target_points / open_risk_points) if open_risk_points > 0 else 0.0, 2),
                    "Risk $ (1 contract)": round(open_risk_dollars, 2),
                    "Executed": "Yes",
                    "Execution Time": _fmt_ts(exec_time),
                    "Outcome": "Open",
                    "Exit Time": "n/a",
                    "Exit Price": "n/a",
                    "Why": "No bars yet after execution timestamp.",
                }
            )
            continue

        outcome = "Open"
        exit_time = "n/a"
        exit_price: object = "n/a"
        why = "Target/stop not reached yet."
        for _, bar in future.iterrows():
            high = float(bar["high"])
            low = float(bar["low"])
            bar_ts = pd.to_datetime(bar["timestamp"])

            if is_long:
                hit_stop = low <= stop
                hit_target = high >= tgt
            else:
                hit_stop = high >= stop
                hit_target = low <= tgt

            if hit_stop and hit_target:
                # Conservative tie-breaker: assume stop first.
                outcome = "Failed"
                exit_time = _fmt_ts(bar_ts) or "n/a"
                exit_price = round(stop, 2)
                why = "Both stop and target touched in same bar; conservative stop-first assumption applied."
                break
            if hit_stop:
                outcome = "Failed"
                exit_time = _fmt_ts(bar_ts) or "n/a"
                exit_price = round(stop, 2)
                why = "Stop level was touched after execution."
                break
            if hit_target:
                outcome = "Successful"
                exit_time = _fmt_ts(bar_ts) or "n/a"
                exit_price = round(tgt, 2)
                why = "Target level was reached after execution."
                break

        final_risk_points = max(0.25, abs(entry_price - stop))
        final_ticks = int(round(final_risk_points / 0.25))
        final_target_points = max(0.0, abs(tgt - entry_price))
        final_target_ticks = int(round(final_target_points / 0.25))
        results.append(
            {
                **meta,
                "Confluence": confluence,
                "Action": action,
                "Suggested Entry": suggested,
                "Entry Price": round(entry_price, 2),
            "Risk (Ticks)": final_ticks,
            "Target (Ticks)": final_target_ticks,
                "Risk Pts(Ticks)": f"{final_risk_points:.2f} ({final_ticks})",
            "Target Pts(Ticks)": f"{final_target_points:.2f} ({final_target_ticks})",
                "RR": round((final_target_points / final_risk_points) if final_risk_points > 0 else 0.0, 2),
                "Risk $ (1 contract)": round(final_risk_points * 20.0, 2),
                "Executed": "Yes",
                "Execution Time": _fmt_ts(exec_time) or "n/a",
                "Outcome": outcome,
                "Exit Time": exit_time,
                "Exit Price": exit_price,
                "Why": why,
            }
        )

    return sorted(
        results,
        key=lambda r: (_to_ts(r.get("Suggested Time")) or pd.Timestamp.min),
        reverse=True,
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _infer_bar_minutes(df: pd.DataFrame, default_minutes: int = 5) -> int:
    if df is None or df.empty or "timestamp" not in df.columns or len(df) < 3:
        return default_minutes
    ts = pd.to_datetime(df["timestamp"]).sort_values()
    deltas = ts.diff().dropna().dt.total_seconds() / 60.0
    deltas = deltas[(deltas >= 1) & (deltas <= 60)]
    if deltas.empty:
        return default_minutes
    mode = deltas.mode()
    minutes = int(round(float(mode.iloc[0]))) if not mode.empty else default_minutes
    return max(1, minutes)


def _us_open_reclaim_watch(
    df: pd.DataFrame,
    trade_date: dt.date,
    now: dt.datetime,
    confluences: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    start_0930 = pd.Timestamp.combine(trade_date, dt.time(9, 30))
    end_1030 = pd.Timestamp.combine(trade_date, dt.time(10, 30))
    confluences = confluences or []

    base = {
        "name": "US Open Reclaim Pattern",
        "status": "Not Active",
        "direction": "Bullish",
        "confidence": 0.0,
        "reason": "Waiting for US open structure.",
        "checklist": [
            "9:30 candle shows two-sided rejection (long upper and lower wick).",
            "9:31 candle closes green.",
            "Price reclaims daily VWAP by 10:30 ET.",
        ],
        "entry": None,
        "stop": None,
        "targets": [],
        "invalidation": "No VWAP reclaim by 10:30 ET or a break below setup low after trigger.",
        "anticipation_factors": [],
        "provisional_plan": {
            "entry_rule": "After 9:31, first close back above Daily VWAP.",
            "stop_rule": "Lowest low from 9:30 through trigger candle.",
            "target_rule": "Target 1 = pre-open high (09:00-09:29), Target 2 = 2R extension.",
        },
        "score_breakdown": [],
        "penalty_points": 0.0,
        "model_settings": {
            "candle3_strictness": OPEN_PATTERN_CANDLE3_STRICTNESS,
            "reclaim_speed_window_minutes": OPEN_PATTERN_RECLAIM_SPEED_WINDOW_MINUTES,
        },
    }

    if df is None or df.empty:
        base["reason"] = "No intraday bars available."
        return base

    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    bar_minutes = _infer_bar_minutes(work, default_minutes=5)

    second_candle_ts = start_0930 + pd.Timedelta(minutes=bar_minutes)
    third_candle_ts = second_candle_ts + pd.Timedelta(minutes=bar_minutes)
    if bar_minutes <= 1:
        base["checklist"] = [
            "9:30 candle shows two-sided rejection (long upper and lower wick).",
            "9:31 candle closes green.",
            "Price reclaims daily VWAP by 10:30 ET.",
        ]
    else:
        base["checklist"] = [
            "9:30 candle shows sweep/rejection behavior.",
            "Next 5m candle is bullish or followed by a strong candle-3 close above candle-2 body/high.",
            "Price reclaims/accepts above daily VWAP during the open window.",
        ]

    pre_open_window = work[
        (work["timestamp"] >= pd.Timestamp.combine(trade_date, dt.time(8, 0)))
        & (work["timestamp"] < start_0930)
    ]

    vwap = compute_daily_vwap(work)
    if not vwap.empty:
        work["daily_vwap"] = vwap.values
    else:
        work["daily_vwap"] = pd.NA

    component_max = {
        "Sweep Quality": 20.0,
        "VWAP Reclaim Quality": 20.0,
        "Confirmation Candle": 15.0,
        "Retest Behavior": 15.0,
        "Structure Alignment": 15.0,
        "Liquidity Path Clarity": 10.0,
    }

    def _build_score(components: Dict[str, float], notes: Dict[str, str], penalty_points: float) -> Tuple[float, List[Dict[str, object]]]:
        rows: List[Dict[str, object]] = []
        total = 0.0
        for name, max_score in component_max.items():
            val = max(0.0, min(float(components.get(name, 0.0)), max_score))
            total += val
            rows.append(
                {
                    "component": name,
                    "score": round(val, 1),
                    "max": max_score,
                    "note": notes.get(name, ""),
                }
            )
        total -= max(0.0, min(float(penalty_points), 15.0))
        return max(0.0, min(total, 100.0)), rows

    if now < start_0930:
        factors: List[str] = []
        notes: Dict[str, str] = {}
        comp = {name: 0.0 for name in component_max.keys()}
        penalty_points = 0.0

        if not pre_open_window.empty:
            pre_high = _safe_float(pre_open_window["high"].max())
            pre_low = _safe_float(pre_open_window["low"].min())
            pre_range = max(pre_high - pre_low, 1e-6)
            last_close = _safe_float(pre_open_window["close"].iloc[-1])
            last_open = _safe_float(pre_open_window["open"].iloc[-1])

            early = pre_open_window[pre_open_window["timestamp"] < pd.Timestamp.combine(trade_date, dt.time(9, 0))]
            late = pre_open_window[pre_open_window["timestamp"] >= pd.Timestamp.combine(trade_date, dt.time(9, 0))]

            if not early.empty and not late.empty:
                early_low = _safe_float(early["low"].min())
                late_low = _safe_float(late["low"].min())
                recovery_ratio = (last_close - late_low) / pre_range
                if late_low < early_low:
                    comp["Sweep Quality"] = min(12.0 + max(recovery_ratio, 0.0) * 16.0, 20.0)
                    notes["Sweep Quality"] = "Pre-open low sweep and recovery detected."
                    factors.append("Pre-open swept earlier lows and recovered (bullish sweep behavior).")
                else:
                    comp["Sweep Quality"] = 6.0
                    notes["Sweep Quality"] = "No clear pre-open sweep; partial credit only."
            else:
                comp["Sweep Quality"] = 5.0
                notes["Sweep Quality"] = "Limited early/late pre-open segmentation available."

            if pd.notna(work["daily_vwap"].iloc[-1]):
                vwap_last = _safe_float(work["daily_vwap"].iloc[-1])
                dist = abs(last_close - vwap_last)
                if last_close <= vwap_last:
                    comp["VWAP Reclaim Quality"] = min(16.0 + max(0.0, 1.0 - dist / max(pre_range, 1e-6)) * 4.0, 20.0)
                    notes["VWAP Reclaim Quality"] = "Below/near VWAP pre-open with reclaim potential."
                    factors.append("Price is at/below Daily VWAP pre-open (reclaim potential).")
                else:
                    comp["VWAP Reclaim Quality"] = max(6.0, 12.0 - min(dist / max(pre_range, 1e-6), 1.0) * 6.0)
                    notes["VWAP Reclaim Quality"] = "Already above VWAP; less reclaim edge but still actionable."
                    factors.append("Price already above Daily VWAP pre-open (less reclaim distance).")
            else:
                comp["VWAP Reclaim Quality"] = 8.0
                notes["VWAP Reclaim Quality"] = "VWAP unavailable; using reduced default score."

            body_ratio = abs(last_close - last_open) / pre_range
            if body_ratio > 0.18:
                comp["Confirmation Candle"] = min(4.0 + body_ratio * 30.0, 9.0)
                notes["Confirmation Candle"] = "Premarket impulse candle present; open confirmation pending."
            else:
                comp["Confirmation Candle"] = 3.0
                notes["Confirmation Candle"] = "No strong pre-open impulse candle yet."

            late_window = pre_open_window[pre_open_window["timestamp"] >= pd.Timestamp.combine(trade_date, dt.time(9, 15))]
            if not late_window.empty:
                lows = late_window["low"].astype(float).values
                higher_low = lows[-1] >= min(lows)
                comp["Retest Behavior"] = 8.0 if higher_low else 4.0
                notes["Retest Behavior"] = "Pre-open pullbacks are stabilizing." if higher_low else "Pre-open retests still unstable."
            else:
                comp["Retest Behavior"] = 4.0
                notes["Retest Behavior"] = "Insufficient late pre-open bars for retest assessment."

            if not pre_open_window.empty and len(pre_open_window) >= 8:
                recent = pre_open_window.tail(8)
                first_mid = (_safe_float(recent.iloc[0]["open"]) + _safe_float(recent.iloc[0]["close"])) / 2.0
                last_mid = (_safe_float(recent.iloc[-1]["open"]) + _safe_float(recent.iloc[-1]["close"])) / 2.0
                slope_up = last_mid > first_mid
                comp["Structure Alignment"] = 10.0 if slope_up else 5.0
                notes["Structure Alignment"] = "Short-term pre-open slope is rising." if slope_up else "Pre-open slope not yet aligned bullish."
            else:
                comp["Structure Alignment"] = 5.0
                notes["Structure Alignment"] = "Not enough bars for structure slope estimate."

            bullish_confluences = [
                c
                for c in confluences
                if str(c.get("Status", "")).lower() != "invalidated" and str(c.get("Side", "")).lower() == "bullish"
            ]
            near = False
            if bullish_confluences:
                for c in bullish_confluences[:7]:
                    low_px = _safe_float(c.get("Price Low"))
                    high_px = _safe_float(c.get("Price High"))
                    if low_px - 10.0 <= last_close <= high_px + 10.0:
                        near = True
                        break
            if near:
                comp["Liquidity Path Clarity"] = 8.0
                notes["Liquidity Path Clarity"] = "Price is near bullish confluence with upside path to pre-open high."
                factors.append("Nearby bullish liquidity confluence detected.")
            else:
                comp["Liquidity Path Clarity"] = 4.0
                notes["Liquidity Path Clarity"] = "Liquidity path less clear pre-open."

            if abs(last_close - (pre_low + pre_range * 0.5)) < pre_range * 0.10:
                penalty_points += 4.0
                factors.append("Penalty: pre-open close near range midpoint (less directional edge).")

            if pd.notna(work["daily_vwap"].iloc[-1]) and len(pre_open_window) >= 20:
                recent = pre_open_window.tail(20).copy()
                rv = work[(work["timestamp"].isin(recent["timestamp"]))].copy()
                if not rv.empty and "daily_vwap" in rv.columns:
                    side = (rv["close"].astype(float) > rv["daily_vwap"].astype(float)).astype(int)
                    flips = int((side.diff().abs().fillna(0) > 0).sum())
                    if flips >= 6:
                        penalty_points += 5.0
                        factors.append("Penalty: frequent VWAP side flips pre-open (chop risk).")

            base["provisional_plan"] = {
                "entry_rule": "If 9:30/9:31 structure confirms, enter on first close above Daily VWAP.",
                "stop_rule": f"Use setup low; pre-open reference low is {pre_low:.2f}.",
                "target_rule": f"Target 1 near pre-open high {pre_high:.2f}; Target 2 at 2R extension.",
            }
        else:
            factors.append("Limited pre-open bars; use reduced confidence.")
            comp["Sweep Quality"] = 4.0
            comp["VWAP Reclaim Quality"] = 6.0
            comp["Confirmation Candle"] = 2.0
            comp["Retest Behavior"] = 2.0
            comp["Structure Alignment"] = 3.0
            comp["Liquidity Path Clarity"] = 2.0
            notes["Sweep Quality"] = "No pre-open structure data."
            notes["VWAP Reclaim Quality"] = "No pre-open structure data."
            notes["Confirmation Candle"] = "No pre-open structure data."
            notes["Retest Behavior"] = "No pre-open structure data."
            notes["Structure Alignment"] = "No pre-open structure data."
            notes["Liquidity Path Clarity"] = "No pre-open structure data."

        total_score, breakdown = _build_score(comp, notes, penalty_points)

        base["status"] = "Watch"
        base["confidence"] = round(min(total_score, 70.0), 1)
        base["reason"] = "Pre-open anticipation mode: monitor open for sweep/reclaim confirmation."
        base["anticipation_factors"] = factors
        base["score_breakdown"] = breakdown
        base["penalty_points"] = round(min(max(penalty_points, 0.0), 15.0), 1)
        return base

    c1_df = work[work["timestamp"] == start_0930]
    c2_df = work[work["timestamp"] == second_candle_ts]
    c3_df = work[work["timestamp"] == third_candle_ts]

    if c1_df.empty:
        base["status"] = "Invalidated"
        base["reason"] = "Missing 9:30 candle data for this date."
        return base

    c1 = c1_df.iloc[0]
    c1_high = _safe_float(c1["high"])
    c1_low = _safe_float(c1["low"])
    c1_open = _safe_float(c1["open"])
    c1_close = _safe_float(c1["close"])
    c1_range = max(c1_high - c1_low, 1e-6)
    c1_upper = c1_high - max(c1_open, c1_close)
    c1_lower = min(c1_open, c1_close) - c1_low
    c1_body = abs(c1_close - c1_open)
    wick_ok = c1_upper >= 0.20 * c1_range and c1_lower >= 0.20 * c1_range and c1_body <= 0.60 * c1_range

    notes: Dict[str, str] = {}
    comp = {name: 0.0 for name in component_max.keys()}
    penalty_points = 0.0

    pre_open_low = _safe_float(pre_open_window["low"].min()) if not pre_open_window.empty else c1_low
    pre_open_high = _safe_float(pre_open_window["high"].max()) if not pre_open_window.empty else c1_high

    lower_ratio = c1_lower / c1_range
    upper_ratio = c1_upper / c1_range
    sweep_bonus = 5.0 if c1_low < pre_open_low else 0.0
    comp["Sweep Quality"] = min(max(lower_ratio * 10.0 + upper_ratio * 5.0 + sweep_bonus, 0.0), 20.0)
    notes["Sweep Quality"] = "9:30 two-sided rejection quality plus low sweep context."

    if c2_df.empty:
        base["status"] = "Watch"
        comp["Confirmation Candle"] = 4.0 if wick_ok else 1.5
        notes["Confirmation Candle"] = "Waiting for second open-window candle confirmation."
        comp["Retest Behavior"] = 3.0
        notes["Retest Behavior"] = "Retest sequence not formed yet."
        comp["Structure Alignment"] = 6.0 if wick_ok else 3.0
        notes["Structure Alignment"] = "Provisional structure read after 9:30 only."
        comp["Liquidity Path Clarity"] = 5.0 if pre_open_high > c1_close else 3.0
        notes["Liquidity Path Clarity"] = "Path estimated from pre-open highs."
        comp["VWAP Reclaim Quality"] = 5.0
        notes["VWAP Reclaim Quality"] = "VWAP reclaim not evaluated yet."
        total_score, breakdown = _build_score(comp, notes, penalty_points)
        base["confidence"] = round(min(total_score, 74.0), 1)
        base["reason"] = "Open candle printed; waiting for second-candle confirmation."
        base["score_breakdown"] = breakdown
        base["penalty_points"] = 0.0
        return base

    c2 = c2_df.iloc[0]
    c2_open = _safe_float(c2["open"])
    c2_close = _safe_float(c2["close"])
    c2_high = _safe_float(c2["high"])
    c2_green = c2_close > c2_open

    c3_close = _safe_float(c3_df.iloc[0]["close"]) if not c3_df.empty else c2_close
    c2_body_top = max(c2_open, c2_close)
    c2_body_mid = (c2_open + c2_close) / 2.0
    c3_break_mid = c3_close > c2_body_mid
    c3_break_body = c3_close > c2_body_top
    c3_break_high = c3_close > c2_high

    strictness = str(OPEN_PATTERN_CANDLE3_STRICTNESS).strip().lower()
    if strictness not in {"conservative", "balanced", "aggressive"}:
        strictness = "balanced"

    if strictness == "conservative":
        c3_confirm = c3_break_high
    elif strictness == "aggressive":
        c3_confirm = c3_break_mid
    else:
        c3_confirm = c3_break_body

    if bar_minutes <= 1:
        setup_confirmed = c2_green
    else:
        setup_confirmed = c2_green or c3_confirm

    if not wick_ok or not setup_confirmed:
        base["status"] = "Invalidated"
        why = []
        if not wick_ok:
            why.append("9:30 candle did not meet two-sided wick quality")
        if not setup_confirmed:
            if bar_minutes <= 1:
                why.append("9:31 did not close green")
            else:
                why.append("second/third open candles did not confirm bullish continuation")
        base["reason"] = "; ".join(why) + "."
        base["confidence"] = 15.0
        comp["Confirmation Candle"] = 0.0
        notes["Confirmation Candle"] = "Open-window candle confirmation failed."
        comp["VWAP Reclaim Quality"] = 2.0
        notes["VWAP Reclaim Quality"] = "Setup invalid before reclaim."
        comp["Retest Behavior"] = 2.0
        notes["Retest Behavior"] = "No valid continuation retest sequence."
        comp["Structure Alignment"] = 2.0
        notes["Structure Alignment"] = "Structure did not confirm."
        comp["Liquidity Path Clarity"] = 2.0
        notes["Liquidity Path Clarity"] = "Path is unclear after invalidation."
        _, breakdown = _build_score(comp, notes, 8.0)
        base["score_breakdown"] = breakdown
        base["penalty_points"] = 8.0
        return base

    body_strength = (c2_close - c2_open) / c1_range
    engulfing = c2_close >= c1_open
    c3_bonus = 0.0
    if bar_minutes > 1:
        if c3_break_body:
            c3_bonus += 2.0
        if c3_break_high:
            c3_bonus += 2.0
    comp["Confirmation Candle"] = min(8.0 + max(body_strength, 0.0) * 10.0 + (4.0 if engulfing else 0.0) + c3_bonus, 15.0)
    notes["Confirmation Candle"] = f"Open-window confirmation strength (candle-3 strictness: {strictness})."

    vwap = compute_daily_vwap(work)
    if vwap.empty:
        base["status"] = "Armed"
        comp["VWAP Reclaim Quality"] = 7.0
        notes["VWAP Reclaim Quality"] = "VWAP unavailable; reduced confidence until reclaim proxy."
        comp["Retest Behavior"] = 5.0
        notes["Retest Behavior"] = "Retest pending."
        comp["Structure Alignment"] = 8.0
        notes["Structure Alignment"] = "Early structure favorable; needs reclaim confirmation."
        comp["Liquidity Path Clarity"] = 6.0
        notes["Liquidity Path Clarity"] = "Upside path based on pre-open high reference."
        total_score, breakdown = _build_score(comp, notes, penalty_points)
        base["confidence"] = round(min(total_score, 78.0), 1)
        base["reason"] = "Setup formed; VWAP unavailable, waiting for reclaim confirmation."
        base["score_breakdown"] = breakdown
        base["penalty_points"] = round(penalty_points, 1)
        return base

    work["daily_vwap"] = vwap.values
    reclaim_window = work[
        (work["timestamp"] >= second_candle_ts)
        & (work["timestamp"] <= end_1030)
        & (work["close"] > work["daily_vwap"])
    ]

    if reclaim_window.empty:
        latest_ts = pd.to_datetime(work["timestamp"].iloc[-1])
        comp["VWAP Reclaim Quality"] = 6.0
        notes["VWAP Reclaim Quality"] = "Reclaim not printed yet; still waiting."
        comp["Retest Behavior"] = 6.0
        notes["Retest Behavior"] = "Post-open retest developing; no reclaim confirmation yet."
        comp["Structure Alignment"] = 8.0
        notes["Structure Alignment"] = "Structure favorable but unconfirmed by VWAP acceptance."
        comp["Liquidity Path Clarity"] = 6.0
        notes["Liquidity Path Clarity"] = "Path remains provisional until reclaim."
        total_score, breakdown = _build_score(comp, notes, penalty_points)
        if latest_ts <= end_1030:
            base["status"] = "Armed"
            base["confidence"] = round(min(total_score, 82.0), 1)
            base["reason"] = "Setup confirmed; waiting for first close above VWAP."
        else:
            base["status"] = "Invalidated"
            base["confidence"] = 22.0
            base["reason"] = "No VWAP reclaim by 10:30 ET."
            penalty_points += 9.0
        base["score_breakdown"] = breakdown
        base["penalty_points"] = round(min(max(penalty_points, 0.0), 15.0), 1)
        return base

    trigger = reclaim_window.iloc[0]
    trigger_ts = pd.to_datetime(trigger["timestamp"])
    entry = _safe_float(trigger["close"])

    stop_window = work[(work["timestamp"] >= start_0930) & (work["timestamp"] <= trigger_ts)]
    stop_price = _safe_float(stop_window["low"].min()) if not stop_window.empty else _safe_float(c1_low)
    risk = max(entry - stop_price, 0.25)

    pre_open_high = _safe_float(pre_open_window["high"].max()) if not pre_open_window.empty else entry + risk
    if pre_open_high <= entry:
        pre_open_high = entry + risk

    target_1 = pre_open_high
    target_2 = entry + 2.0 * risk
    rr_1 = (target_1 - entry) / risk

    reclaim_delay = max((trigger_ts - second_candle_ts).total_seconds() / 60.0, 0.0)
    configured_speed = max(float(OPEN_PATTERN_RECLAIM_SPEED_WINDOW_MINUTES), 10.0)
    speed_norm = max(12.0, configured_speed * 0.6) if bar_minutes <= 1 else configured_speed
    reclaim_speed_score = max(0.0, 1.0 - min(reclaim_delay / speed_norm, 1.0))
    reclaim_distance = max(entry - _safe_float(trigger["daily_vwap"]), 0.0)
    comp["VWAP Reclaim Quality"] = min(10.0 + reclaim_speed_score * 6.0 + min(reclaim_distance / max(risk, 1e-6), 1.0) * 4.0, 20.0)
    notes["VWAP Reclaim Quality"] = f"Scored from reclaim speed (window {speed_norm:.0f}m) and close strength above VWAP."

    post_trigger_minutes = 8 if bar_minutes <= 1 else 20
    post_trigger = work[(work["timestamp"] > trigger_ts) & (work["timestamp"] <= trigger_ts + pd.Timedelta(minutes=post_trigger_minutes))].copy()
    retest_score = 6.0
    if not post_trigger.empty:
        touched_vwap = (post_trigger["low"] <= post_trigger["daily_vwap"]).any()
        held_after_touch = False
        if touched_vwap:
            touch_idx = post_trigger[post_trigger["low"] <= post_trigger["daily_vwap"]].index[0]
            after_touch = post_trigger.loc[touch_idx:]
            held_after_touch = bool((after_touch["close"] >= after_touch["daily_vwap"]).all())
        if touched_vwap and held_after_touch:
            retest_score = 14.0
        elif touched_vwap:
            retest_score = 8.0
            penalty_points += 3.0
        else:
            retest_score = 9.0
    comp["Retest Behavior"] = min(retest_score, 15.0)
    notes["Retest Behavior"] = "Scored from post-reclaim VWAP retest and hold behavior."

    pre_trigger = work[(work["timestamp"] >= start_0930) & (work["timestamp"] <= trigger_ts)]
    recent_high = _safe_float(pre_trigger["high"].tail(6).max()) if not pre_trigger.empty else entry
    structure_break = entry >= recent_high
    comp["Structure Alignment"] = 12.0 if structure_break else 8.0
    notes["Structure Alignment"] = "Scored from local structure break strength into trigger."

    bearish_blocks_between = 0
    for c in confluences[:10]:
        side = str(c.get("Side", "")).lower()
        status = str(c.get("Status", "")).lower()
        if side != "bearish" or status == "invalidated":
            continue
        low_px = _safe_float(c.get("Price Low"))
        if entry < low_px < target_1:
            bearish_blocks_between += 1
    path_score = 10.0 - min(bearish_blocks_between * 2.5, 7.5)
    path_score += min(max(rr_1 - 1.0, 0.0) * 1.5, 2.5)
    comp["Liquidity Path Clarity"] = max(2.0, min(path_score, 10.0))
    notes["Liquidity Path Clarity"] = "Scored by target path friction and R multiple quality."

    if rr_1 < 1.0:
        penalty_points += 5.0
    if not post_trigger.empty:
        adverse = float((entry - post_trigger["low"]).max()) if len(post_trigger) else 0.0
        if adverse > 0.8 * risk:
            penalty_points += 4.0

    confidence, breakdown = _build_score(comp, notes, penalty_points)

    base["status"] = "Triggered"
    base["confidence"] = round(confidence, 1)
    if bar_minutes <= 1:
        base["reason"] = "9:30/9:31 setup formed and first close above daily VWAP confirmed the trigger."
    else:
        base["reason"] = "Open-window setup formed and first close above daily VWAP confirmed the trigger."
    base["entry"] = {
        "time": _fmt_ts(trigger_ts),
        "price": round(entry, 2),
    }
    base["stop"] = {
        "price": round(stop_price, 2),
        "rule": "Lowest low from 9:30 through trigger candle.",
    }
    base["targets"] = [
        {
            "name": "Target 1 (Pre-open High)",
            "price": round(target_1, 2),
            "rr": round(rr_1, 2),
        },
        {
            "name": "Target 2 (2R Extension)",
            "price": round(target_2, 2),
            "rr": 2.0,
        },
    ]
    base["anticipation_factors"] = [
        "Post-open confirmation complete: 9:30/9:31 structure + VWAP reclaim.",
        f"Pre-open high reference used for Target 1 ({target_1:.2f}).",
    ]
    base["score_breakdown"] = breakdown
    base["penalty_points"] = round(min(max(penalty_points, 0.0), 15.0), 1)
    return base


def _zone_touch_times(df: pd.DataFrame, zone: Zone) -> Tuple[Optional[str], Optional[str], int]:
    if df is None or df.empty:
        return None, None, 0
    formed_ts = zone_formed_timestamp(zone)
    after = df[df["timestamp"] > formed_ts]
    if after.empty:
        return None, None, 0
    hits = after[(after["low"] <= zone.high) & (after["high"] >= zone.low)]
    if hits.empty:
        return None, None, 0
    first = _fmt_ts(pd.to_datetime(hits.iloc[0]["timestamp"]))
    retest = _fmt_ts(pd.to_datetime(hits.iloc[1]["timestamp"])) if len(hits) > 1 else None
    return first, retest, max(len(hits) - 1, 0)


def _stdv_levels(df: pd.DataFrame, center: float, length: int = 20) -> List[Dict[str, object]]:
    if df is None or df.empty:
        return []
    std = df["close"].rolling(length).std()
    if std.empty or pd.isna(std.iloc[-1]):
        return []
    sigma = float(std.iloc[-1])
    if sigma <= 0:
        return []

    day_high = float(df["high"].max())
    day_low = float(df["low"].min())

    levels: List[Dict[str, object]] = []
    for n in [1, 2, 3]:
        up = center + n * sigma
        down = center - n * sigma
        levels.append(
            {
                "Level": f"+{n}σ",
                "Price": float(up),
                "From Center": float(n * sigma),
                "Status": "Reached" if day_high >= up else "Untouched",
            }
        )
        levels.append(
            {
                "Level": f"-{n}σ",
                "Price": float(down),
                "From Center": float(n * sigma),
                "Status": "Reached" if day_low <= down else "Untouched",
            }
        )
    return levels


def _liquidity_level(score: float) -> str:
    if score < 0.25:
        return "Very Low"
    if score < 0.60:
        return "Low"
    if score < 1.00:
        return "Medium"
    if score < 1.50:
        return "High"
    if score < 2.20:
        return "Very High"
    return "Major"


def _enrich_trigger(trigger: Dict[str, object]) -> Dict[str, object]:
    name = str(trigger.get("name", ""))
    if "VWAP Reject Hold" in name:
        trigger["what_happened"] = "Price crossed below VWAP and held below it for 4 consecutive closes."
        trigger["must_happen"] = "On a pullback, price should fail to reclaim VWAP."
        trigger["execution_look_for"] = "Bearish retest rejection candle at/under VWAP plus momentum expansion lower."
        trigger["invalidation"] = "A sustained close back above VWAP."
        return trigger
    if "VWAP Reclaim Hold" in name:
        trigger["what_happened"] = "Price crossed above VWAP and held above it for 4 consecutive closes."
        trigger["must_happen"] = "On a pullback, price should hold above VWAP."
        trigger["execution_look_for"] = "Bullish hold/rejection candle above VWAP plus momentum expansion higher."
        trigger["invalidation"] = "A sustained close back below VWAP."
        return trigger
    if "Failed ORB" in name:
        trigger["what_happened"] = "An OR breakout failed and price closed back inside the OR window."
        trigger["must_happen"] = "Price should continue away from the failed breakout side."
        trigger["execution_look_for"] = "Retest failure of OR edge and continuation in failure direction."
        trigger["invalidation"] = "Re-break and hold beyond original OR breakout side."
        return trigger
    if "ORB" in name:
        trigger["what_happened"] = "Price closed decisively beyond the opening range threshold."
        trigger["must_happen"] = "Retest should hold on the breakout side."
        trigger["execution_look_for"] = "Break-and-retest continuation with no immediate rejection back into OR."
        trigger["invalidation"] = "Close back inside OR with failed follow-through."
        return trigger

    trigger["what_happened"] = trigger.get("details", "Trigger condition was detected.")
    trigger["must_happen"] = "Price must continue to respect the trigger side."
    trigger["execution_look_for"] = "A confirmed retest and continuation candle sequence."
    trigger["invalidation"] = "Clear rejection of trigger direction."
    return trigger


def _pick_primary_trigger(triggers: List[Dict[str, object]], direction: str) -> Optional[Dict[str, object]]:
    if not triggers:
        return None

    def _priority(name: str) -> int:
        if "ORB 30m" in name:
            return 1
        if "ORB 60m" in name:
            return 2
        if "Failed ORB 30m" in name:
            return 3
        if "Failed ORB 60m" in name:
            return 4
        if "VWAP" in name:
            return 5
        return 9

    candidates = triggers
    if direction in {"Bullish", "Bearish"}:
        aligned = [t for t in triggers if t.get("direction") == direction]
        if aligned:
            candidates = aligned

    candidates = sorted(candidates, key=lambda t: (_priority(str(t.get("name", ""))), str(t.get("time", ""))))
    return candidates[0] if candidates else None


def _select_reference_confluences(
    confluences: List[Dict[str, object]],
    direction: str,
    max_items: int = 2,
) -> List[str]:
    if not confluences:
        return []

    filtered = [c for c in confluences if c.get("Status") != "Invalidated"]
    if direction == "Bullish":
        directional = [c for c in filtered if str(c.get("Side", "")).lower() == "bullish"]
    elif direction == "Bearish":
        directional = [c for c in filtered if str(c.get("Side", "")).lower() == "bearish"]
    else:
        directional = filtered

    candidates = directional if directional else filtered
    candidates = sorted(
        candidates,
        key=lambda x: (not bool(x.get("Top Liquidity", False)), -float(x.get("Liquidity Score", 0.0))),
    )

    refs: List[str] = []
    for row in candidates[:max_items]:
        refs.append(
            f"{row.get('Confluence')} ({float(row.get('Price Low', 0.0)):.2f}-{float(row.get('Price High', 0.0)):.2f})"
        )
    return refs


def _select_reference_confluence_rows(
    confluences: List[Dict[str, object]],
    direction: str,
    max_items: int = 2,
) -> List[Dict[str, object]]:
    if not confluences:
        return []

    filtered = [c for c in confluences if c.get("Status") != "Invalidated"]
    if direction == "Bullish":
        directional = [c for c in filtered if str(c.get("Side", "")).lower() == "bullish"]
    elif direction == "Bearish":
        directional = [c for c in filtered if str(c.get("Side", "")).lower() == "bearish"]
    else:
        directional = filtered

    candidates = directional if directional else filtered
    return sorted(
        candidates,
        key=lambda x: (not bool(x.get("Top Liquidity", False)), -float(x.get("Liquidity Score", 0.0))),
    )[:max_items]


def _first_zone_tap_time(
    df: pd.DataFrame,
    zone_low: float,
    zone_high: float,
    formed_time: Optional[str] = None,
) -> Optional[str]:
    if df is None or df.empty:
        return None
    work = df
    if formed_time and formed_time != "n/a":
        formed_ts = pd.to_datetime(formed_time, errors="coerce")
        if pd.notna(formed_ts):
            work = work[pd.to_datetime(work["timestamp"]) > formed_ts]
    hits = work[(work["low"] <= zone_high) & (work["high"] >= zone_low)]
    if hits.empty:
        return None
    return _fmt_ts(pd.to_datetime(hits.iloc[0]["timestamp"]))


def _first_midline_hit_time(
    df: pd.DataFrame,
    midline: float,
    formed_time: Optional[str] = None,
) -> Optional[str]:
    if df is None or df.empty:
        return None
    work = df
    if formed_time and formed_time != "n/a":
        formed_ts = pd.to_datetime(formed_time, errors="coerce")
        if pd.notna(formed_ts):
            work = work[pd.to_datetime(work["timestamp"]) > formed_ts]
    hits = work[(work["low"] <= midline) & (work["high"] >= midline)]
    if hits.empty:
        return None
    return _fmt_ts(pd.to_datetime(hits.iloc[0]["timestamp"]))


def _entry_style_target_prices(
    target_ladder: List[Dict[str, object]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    prices: List[float] = []
    for t in target_ladder[:8]:
        try:
            prices.append(float(t.get("Price")))
        except Exception:
            continue

    if not prices:
        return None, None, None

    preferred = prices[0]
    minimum = min(prices)
    maximum = max(prices)
    return preferred, minimum, maximum


def _estimate_entry_eta(
    action: str,
    style: str,
    suggested_ts: Optional[pd.Timestamp],
    tap_time: Optional[str],
    midline_time: Optional[str],
    bar_minutes: int,
) -> Tuple[str, str, str]:
    if action not in {"Long", "Short"}:
        return ("n/a", "n/a", "Waiting for confirmation before entry timing can be estimated.")

    tap_ts = _to_ts(tap_time)
    mid_ts = _to_ts(midline_time)
    style_l = str(style or "").lower()

    if "midline" in style_l and mid_ts is not None:
        hhmm = mid_ts.strftime("%H:%M")
        return (hhmm, "0-1 min", f"Midline was already touched around {hhmm} ET.")
    if "first tap" in style_l and tap_ts is not None:
        hhmm = tap_ts.strftime("%H:%M")
        return (hhmm, "0-1 min", f"First-tap level was already touched around {hhmm} ET.")

    if suggested_ts is None:
        return ("n/a", "n/a", "No valid timestamp anchor is available yet to project an entry ETA.")

    bm = max(1, int(bar_minutes))
    if "split" in style_l:
        start = suggested_ts + pd.Timedelta(minutes=max(1 * bm, 3))
        end = suggested_ts + pd.Timedelta(minutes=max(3 * bm, 15))
    elif "midline" in style_l:
        start = suggested_ts + pd.Timedelta(minutes=max(2 * bm, 5))
        end = suggested_ts + pd.Timedelta(minutes=max(4 * bm, 15))
    else:
        start = suggested_ts + pd.Timedelta(minutes=max(1 * bm, 2))
        end = suggested_ts + pd.Timedelta(minutes=max(3 * bm, 10))

    eta_hhmm = start.strftime("%H:%M")
    min_from = max(1, int((start - suggested_ts).total_seconds() // 60))
    min_to = max(1, int((end - suggested_ts).total_seconds() // 60))
    window_label = f"{min_from}-{min_to} min"
    detail = f"Expect entry around {eta_hhmm} ET, typically within {window_label} after the signal anchor."
    return (eta_hhmm, window_label, detail)


def _analyze_zone_reaction_candles(
    df: pd.DataFrame,
    zone_low: float,
    zone_high: float,
    side: str,
    lookahead_bars: int = 3,
) -> Dict[str, object]:
    if df is None or df.empty:
        return {
            "signal": "None",
            "score": 0.0,
            "reason": "No intraday bars available for reaction analysis.",
            "event_time": "n/a",
        }

    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    touches = work[(work["low"] <= zone_high) & (work["high"] >= zone_low)]
    if touches.empty:
        return {
            "signal": "None",
            "score": 0.0,
            "reason": "Zone has not been touched yet.",
            "event_time": "n/a",
        }

    bullish = str(side).lower() == "bullish"
    best_score = -1.0
    best = {
        "signal": "None",
        "score": 0.0,
        "reason": "No qualifying reaction candle found yet.",
        "event_time": "n/a",
    }

    for idx in touches.index[:5]:
        row = work.iloc[idx]
        o = _safe_float(row.get("open"))
        h = _safe_float(row.get("high"))
        l = _safe_float(row.get("low"))
        c = _safe_float(row.get("close"))
        rng = max(h - l, 1e-6)
        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)

        engulfing = False
        rejection = False
        rejection_score = 0.0
        engulfing_score = 0.0

        if idx > 0:
            prev = work.iloc[idx - 1]
            po = _safe_float(prev.get("open"))
            pc = _safe_float(prev.get("close"))
            if bullish:
                engulfing = (c > o) and (pc < po) and (o <= pc) and (c >= po)
            else:
                engulfing = (c < o) and (pc > po) and (o >= pc) and (c <= po)
            if engulfing:
                engulfing_score = 35.0

        if bullish:
            wick_ok = lower_wick >= max(0.8 * body, 0.20 * rng)
            close_away = c >= (zone_low + zone_high) / 2.0
            color_ok = c >= o
            rejection = wick_ok and close_away and color_ok
            if rejection:
                rejection_score = 30.0
        else:
            wick_ok = upper_wick >= max(0.8 * body, 0.20 * rng)
            close_away = c <= (zone_low + zone_high) / 2.0
            color_ok = c <= o
            rejection = wick_ok and close_away and color_ok
            if rejection:
                rejection_score = 30.0

        end_idx = min(idx + max(1, lookahead_bars), len(work) - 1)
        forward = work.iloc[idx : end_idx + 1]
        if bullish:
            departure_points = max(_safe_float(forward["high"].max()) - c, 0.0)
        else:
            departure_points = max(c - _safe_float(forward["low"].min()), 0.0)
        departure_score = min(35.0, departure_points * 4.0)

        score = min(100.0, engulfing_score + rejection_score + departure_score)
        if score > best_score:
            best_score = score
            signal = "None"
            if engulfing and rejection:
                signal = "Engulfing + Rejection"
            elif engulfing:
                signal = "Engulfing"
            elif rejection:
                signal = "Rejection"

            best = {
                "signal": signal,
                "score": round(score, 1),
                "reason": (
                    f"{signal} reaction with departure strength {departure_points:.2f} points "
                    f"over next {max(1, lookahead_bars)} bars."
                    if signal != "None"
                    else f"Touch occurred but no clear engulfing/rejection; departure {departure_points:.2f} points."
                ),
                "event_time": _fmt_ts(pd.to_datetime(row["timestamp"])) or "n/a",
            }

    return best


def _suggest_confluence_entry_styles(
    reference_rows: List[Dict[str, object]],
    df: pd.DataFrame,
    target_ladder: List[Dict[str, object]],
    mode: str,
    whipsaw_risk: bool,
    windows: Optional[Dict[str, Dict[str, dt.datetime]]] = None,
) -> List[Dict[str, object]]:
    suggestions: List[Dict[str, object]] = []
    bar_minutes = _infer_bar_minutes(df, default_minutes=5)
    preferred_tgt, min_tgt, max_tgt = _entry_style_target_prices(target_ladder)
    for row in reference_rows:
        name = str(row.get("Confluence", "n/a"))
        name_l = name.lower()
        side = str(row.get("Side", "")).capitalize() or "Neutral"
        low = float(row.get("Price Low", 0.0))
        high = float(row.get("Price High", 0.0))
        mid = (low + high) / 2.0
        width = abs(high - low)
        status = str(row.get("Status", "Fresh"))
        formed_time = str(row.get("Formed Time", "n/a"))

        tap_time = _first_zone_tap_time(df, low, high, formed_time=formed_time)
        midline_time = _first_midline_hit_time(df, mid, formed_time=formed_time)
        tap_hit = tap_time is not None
        midline_hit = midline_time is not None
        invalidated = status == "Invalidated"

        zone_kind = "Other"
        if "fvg" in name_l:
            zone_kind = "FVG"
        elif "ob" in name_l or "order block" in name_l:
            zone_kind = "OB"
        elif "bb" in name_l or "breaker" in name_l:
            zone_kind = "BB"

        style = "First Tap"
        reason = "Momentum context favors immediate reaction entries."

        reaction = _analyze_zone_reaction_candles(
            df=df,
            zone_low=low,
            zone_high=high,
            side=str(row.get("Side", "")),
            lookahead_bars=3,
        )
        reaction_signal = str(reaction.get("signal", "None"))
        reaction_score = float(reaction.get("score", 0.0) or 0.0)
        reaction_reason = str(reaction.get("reason", "n/a"))
        reaction_time = str(reaction.get("event_time", "n/a"))
        suggested_ts = (
            _to_ts(reaction_time)
            or _to_ts(tap_time)
            or _to_ts(midline_time)
            or _to_ts(formed_time)
        )
        session_marker, suggested_session = _session_marker_for_time(suggested_ts, windows)
        suggested_time_str = _fmt_ts(suggested_ts) or "n/a"
        confluence_range = f"{low:.2f}-{high:.2f}"

        if zone_kind == "FVG":
            style = "Midline"
            reason = "FVG setups typically improve fill quality near the 50% balance point."
        elif zone_kind in {"OB", "BB"}:
            style = "First Tap"
            reason = "OB/BB zones often react on first contact during directional conditions."

        if status in {"Retested", "Tested"}:
            style = "Midline"
            reason = "Zone has already been interacted with; waiting for midline can reduce weak edge taps."

        if whipsaw_risk or mode == "Whipsaw Risk":
            style = "Midline"
            reason = "Whipsaw conditions favor more selective pricing over immediate touch entries."

        if width < 1.5 and style == "Midline":
            style = "First Tap"
            reason = "Zone is narrow enough that midline and tap fills are effectively equivalent."

        # Adaptive FVG mode: strong reaction at touch prioritizes participation over perfect fill.
        if zone_kind == "FVG":
            if reaction_score >= 70.0 and reaction_signal in {"Engulfing", "Rejection", "Engulfing + Rejection"}:
                style = "First Tap"
                reason = f"Strong {reaction_signal.lower()} reaction detected at the zone ({reaction_score:.1f} score)."
            elif 45.0 <= reaction_score < 70.0 and reaction_signal in {"Engulfing", "Rejection", "Engulfing + Rejection"}:
                style = "Split (Tap + Midline)"
                reason = f"Moderate {reaction_signal.lower()} reaction suggests partial fill at tap and add near midline."

        if preferred_tgt is not None:
            exit_plan = f"Scale at preferred target {preferred_tgt:.2f}; extend toward {max_tgt:.2f}."
        else:
            exit_plan = "Use nearest qualified ladder target; exit early on invalidation."

        action = "Long" if side == "Bullish" else "Short" if side == "Bearish" else "Wait"
        if invalidated:
            action = "Wait"

        if reaction_score >= 70.0:
            expected_retests = "0-1 likely before continuation"
            entry_timing = "First touch/reclaim is acceptable if reaction confirms."
        elif reaction_score >= 45.0:
            expected_retests = "1-2 likely before cleaner continuation"
            entry_timing = "Prefer second retest confirmation before full size."
        else:
            expected_retests = "2+ likely or chop risk elevated"
            entry_timing = "Wait for stronger rejection/engulfing evidence."

        if status in {"Retested", "Tested"} and reaction_score < 70.0:
            entry_timing = "Zone already tested; wait for fresh confirmation candle."

        entry_style_key = style.lower()
        entry_price_for_metrics = mid if ("midline" in entry_style_key or "split" in entry_style_key) else (
            low if side == "Bullish" else high
        )
        risk_points_style = max(0.25, abs(entry_price_for_metrics - (low if side == "Bullish" else high)))
        target_price_style = preferred_tgt
        if target_price_style is None:
            if side == "Bullish":
                target_price_style = max_tgt if max_tgt is not None else entry_price_for_metrics + max(width, 0.5)
            elif side == "Bearish":
                target_price_style = min_tgt if min_tgt is not None else entry_price_for_metrics - max(width, 0.5)
            else:
                target_price_style = entry_price_for_metrics
        target_points_style = max(0.0, abs(float(target_price_style) - entry_price_for_metrics))
        risk_ticks_style = int(round(risk_points_style / 0.25))
        target_ticks_style = int(round(target_points_style / 0.25))
        rr_style = (target_points_style / risk_points_style) if risk_points_style > 0 else 0.0

        if action in {"Long", "Short"} and rr_style < 0.9:
            action = "Wait"
            style = "Wait"
            reason = (
                f"Setup blocked because projected RR is {rr_style:.2f}, below the minimum required 0.90. "
                "Wait for either tighter risk or wider target extension."
            )
            entry_timing = "Do not execute until projected RR improves to at least 0.90."

        eta_hhmm, eta_window, eta_detail = _estimate_entry_eta(
            action=action,
            style=style,
            suggested_ts=suggested_ts,
            tap_time=tap_time,
            midline_time=midline_time,
            bar_minutes=bar_minutes,
        )

        suggestions.append(
            {
                "Session Marker": f"{session_marker} {suggested_session}",
                "Suggested Time": suggested_time_str,
                "Entry ETA (HH:MM)": eta_hhmm,
                "Entry Window": eta_window,
                "Entry ETA Detail": eta_detail,
                "Confluence Range": confluence_range,
                "Confluence": name,
                "Side": side,
                "Action": action,
                "Kind": zone_kind,
                "Status": status,
                "Formed Time": formed_time,
                "Suggested Entry": style,
                "First Tap Price": round(low if "bullish" in name_l else high, 2),
                "Midline Price": round(mid, 2),
                "Entry Price": round(entry_price_for_metrics, 2),
                "Risk (Ticks)": risk_ticks_style,
                "Target (Ticks)": target_ticks_style,
                "Risk Pts(Ticks)": f"{risk_points_style:.2f} ({risk_ticks_style})",
                "Target Pts(Ticks)": f"{target_points_style:.2f} ({target_ticks_style})",
                "RR": round(rr_style, 2),
                "Tap Hit": "Yes" if tap_hit else "No",
                "Tap Time": tap_time or "n/a",
                "Midline Hit": "Yes" if midline_hit else "No",
                "Midline Time": midline_time or "n/a",
                "Zone Invalidated": "Yes" if invalidated else "No",
                "Exit": exit_plan,
                "Reaction Signal": reaction_signal,
                "Reaction Score": round(reaction_score, 1),
                "Reaction Time": reaction_time,
                "Reaction Why": reaction_reason,
                "Preferred Target Price": round(preferred_tgt, 2) if preferred_tgt is not None else None,
                "Minimum Target Price": round(min_tgt, 2) if min_tgt is not None else None,
                "Maximum Target Price": round(max_tgt, 2) if max_tgt is not None else None,
                "Min Target": round(min_tgt, 2) if min_tgt is not None else "n/a",
                "Max Target": round(max_tgt, 2) if max_tgt is not None else "n/a",
                "Expected Retests": expected_retests,
                "Entry Timing": entry_timing,
                "Entry Style Reason": reason,
                "Reason": reason,
            }
        )
    return sorted(
        suggestions,
        key=lambda r: (_to_ts(r.get("Suggested Time")) or pd.Timestamp.min),
        reverse=True,
    )


def _estimate_daily_entry_capacity(
    mode: str,
    direction: str,
    trade_today: str,
    confidence: float,
    reference_rows: List[Dict[str, object]],
    primary_trigger: Optional[Dict[str, object]],
    now: dt.datetime,
    windows: Dict[str, Dict[str, dt.datetime]],
) -> Dict[str, object]:
    us_start = windows.get("US", {}).get("start")
    us_end = windows.get("US", {}).get("end")
    trade_bucket = _trade_decision_bucket(trade_today)

    if trade_bucket == "No":
        return {
            "expected_min": 0,
            "expected_max": 0,
            "likely": 0,
            "remaining_estimate": 0,
            "planning_note": "Trade-today decision is No, so no entries are planned.",
            "eod_execution_status": "Not Executed",
            "eod_execution_note": "Trade-today decision prevented execution.",
        }

    min_e, max_e = 0, 2
    if mode in {"AMD", "NY Continuation"} and direction in {"Bullish", "Bearish"}:
        min_e, max_e = 1, 3
    elif mode == "Whipsaw Risk":
        min_e, max_e = 0, 1

    if direction not in {"Bullish", "Bearish"}:
        min_e = 0
        max_e = min(max_e, 2)

    top_liq_count = sum(1 for r in reference_rows if bool(r.get("Top Liquidity", False)))
    if top_liq_count >= 2 and mode != "Whipsaw Risk":
        max_e = min(max_e + 1, 4)

    if primary_trigger is not None and trade_bucket in {"Yes", "Wait"}:
        min_e = max(min_e, 1)

    if us_start and now < us_start:
        note = "Pre-US open estimate based on mode, directional alignment, and confluence quality."
        eod_status = "Pending"
        eod_note = "Session is not complete yet."
    elif us_end and now > us_end:
        if primary_trigger is None:
            min_e, max_e = 0, 0
            note = "US session has ended without a trigger; no additional entries expected."
            eod_status = "Not Executed"
            eod_note = "No qualifying trigger was confirmed by session close."
        else:
            min_e, max_e = min(min_e, 1), min(max_e, 1)
            note = "US session is largely complete; at most one late opportunity remains."
            eod_status = "Executed"
            eod_note = "A qualifying trigger fired during session hours."
    else:
        note = "In-session estimate adjusts as trigger state and confluence validity evolve."
        eod_status = "Pending"
        eod_note = "Waiting for end-of-day session close to finalize execution status."

    if confidence >= 0.70:
        likely = min(max(min_e + 1, min_e), max_e)
    elif confidence <= 0.55:
        likely = min_e
    else:
        likely = min(min_e + 1, max_e)

    return {
        "expected_min": int(min_e),
        "expected_max": int(max_e),
        "likely": int(likely),
        "remaining_estimate": int(max_e),
        "planning_note": note,
        "eod_execution_status": eod_status,
        "eod_execution_note": eod_note,
    }


def _entry_blueprints(
    mode: str,
    direction: str,
    windows: Dict[str, Dict[str, dt.datetime]],
    reference_confluences: List[str],
    vwap_levels: List[Dict[str, object]],
    entry_style_suggestions: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, str]]:
    us_start = windows.get("US", {}).get("start")
    us_end = windows.get("US", {}).get("end")
    us_window = f"{us_start:%H:%M}-{us_end:%H:%M} ET" if us_start and us_end else "US session"

    primary_confluence = reference_confluences[0] if reference_confluences else "nearest active confluence"
    secondary_confluence = reference_confluences[1] if len(reference_confluences) > 1 else primary_confluence
    daily_vwap_label = "Daily VWAP"
    for v in vwap_levels:
        if str(v.get("Name", "")).lower().startswith("daily vwap"):
            daily_vwap_label = str(v.get("Name"))
            break

    primary_style = "First Tap"
    primary_style_reason = "Use default confluence tap execution."
    if entry_style_suggestions:
        primary_style = str(entry_style_suggestions[0].get("Suggested Entry", primary_style))
        primary_style_reason = str(entry_style_suggestions[0].get("Reason", primary_style_reason))

    if mode == "Whipsaw Risk":
        return [
            {
                "Setup": "Wait For OR Confirmation",
                "Action": "Long/Short (post-OR only)",
                "IfThen": "If whipsaw is active before OR forms, wait until OR is defined and only trade confirmed OR break/retest direction.",
                "Trigger": "ORB 30m/60m or Failed ORB aligned with daily bias.",
                "Confluence Entry": "Confirmation Retest",
                "Entry Reason": "Prevents missing valid post-OR trades while avoiding early chop.",
                "Execution TF": "1m-5m",
                "Session": "09:30-11:30 ET",
            },
            {
                "Setup": "Risk-Off / Two-Sided",
                "Action": "Wait",
                "IfThen": f"If price keeps crossing around {daily_vwap_label} and cannot hold either side of OR, then avoid directional entries.",
                "Trigger": "Use only exceptional reclaim/reject + structure alignment at high-liquidity confluence.",
                "Confluence Entry": primary_style,
                "Entry Reason": primary_style_reason,
                "Execution TF": "5m",
                "Session": us_window,
            }
        ]

    direction_word = "long" if direction == "Bullish" else "short" if direction == "Bearish" else "directional"
    opposite = "below" if direction == "Bullish" else "above"

    base = [
        {
            "Setup": "VWAP Acceptance",
            "Action": "Long" if direction == "Bullish" else "Short" if direction == "Bearish" else "Wait",
            "IfThen": f"If price retests {daily_vwap_label} and holds the {direction.lower()} side near {primary_confluence}, then look for {direction_word} continuation.",
            "Trigger": f"VWAP reclaim/reject hold + 1m structure confirmation while staying {opposite} invalidation side.",
            "Confluence Entry": primary_style,
            "Entry Reason": primary_style_reason,
            "Execution TF": "1m",
            "Session": us_window,
        },
        {
            "Setup": "OR Retest",
            "Action": "Long" if direction == "Bullish" else "Short" if direction == "Bearish" else "Wait",
            "IfThen": f"If OR break aligns {direction.lower()} and retest holds into {secondary_confluence}, then execute continuation.",
            "Trigger": "ORB or Failed ORB confirmation.",
            "Confluence Entry": primary_style,
            "Entry Reason": primary_style_reason,
            "Execution TF": "1m-5m",
            "Session": "09:30-11:30 ET",
        },
    ]

    if mode == "AMD":
        base.insert(
            0,
            {
                "Setup": "AMD Reversal",
                "Action": "Long" if direction == "Bullish" else "Short" if direction == "Bearish" else "Wait",
                "IfThen": f"If London sweep is rejected and US reclaims through {primary_confluence}, then look for {direction_word} toward opposite liquidity.",
                "Trigger": "Sweep rejection + reclaim + micro BOS/CHOCH.",
                "Confluence Entry": primary_style,
                "Entry Reason": primary_style_reason,
                "Execution TF": "1m",
                "Session": "09:30-12:00 ET",
            },
        )

    if mode == "NY Continuation":
        base.insert(
            0,
            {
                "Setup": "NY Continuation Pullback",
                "Action": "Long" if direction == "Bullish" else "Short" if direction == "Bearish" else "Wait",
                "IfThen": f"If price pulls back to {primary_confluence} and holds with {daily_vwap_label} support/resistance, then execute {direction_word}.",
                "Trigger": "Retest hold + momentum re-expansion.",
                "Confluence Entry": primary_style,
                "Entry Reason": primary_style_reason,
                "Execution TF": "1m-5m",
                "Session": us_window,
            },
        )

    return base


def _timeframe_playbook(mode: str) -> List[Dict[str, str]]:
    if mode == "Whipsaw Risk":
        return [
            {
                "Context TF": "15m",
                "Setup TF": "5m",
                "Trigger TF": "1m",
                "Execution TF": "5m",
                "Management TF": "1m-5m",
                "Use Case": "Noise control / defensive mode",
            }
        ]

    return [
        {
            "Context TF": "15m-1H",
            "Setup TF": "5m-15m",
            "Trigger TF": "1m-5m",
            "Execution TF": "1m",
            "Management TF": "1m-5m",
            "Use Case": "Primary intraday execution",
        },
        {
            "Context TF": "15m",
            "Setup TF": "5m",
            "Trigger TF": "1m",
            "Execution TF": "5m",
            "Management TF": "5m",
            "Use Case": "High-volatility fallback execution",
        },
    ]


def build_strategy_playbook(
    df_today: pd.DataFrame,
    df_prev: Optional[pd.DataFrame],
    sessions: Dict[str, SessionStats],
    patterns: PatternSummary,
    zones: List[Zone],
    now_et: Optional[dt.datetime] = None,
    whipsaw_threshold: float = 3.0,
) -> Dict[str, object]:
    if df_today is None or df_today.empty:
        return {
            "decision": {
                "trade_today": "Wait",
                "ny_mode": "n/a",
                "ny_direction": "Neutral",
                "confidence": 0.0,
                "primary_reason": "No intraday data available.",
                "supporting_factors": [],
                "blocking_factors": ["No today data"],
            },
            "triggers": [],
            "confluences": [],
            "targets": [],
            "vwap_levels": [],
            "stdv_levels": [],
            "power_hour": {
                "focus": "No",
                "bias": "Neutral",
                "reason": "No data.",
                "entries": [],
            },
            "entry_blueprints": [],
            "timeframe_playbook": [],
            "open_pattern_watch": {
                "name": "US Open Reclaim Pattern",
                "status": "Not Active",
                "direction": "Bullish",
                "confidence": 0.0,
                "reason": "No intraday data available.",
                "checklist": [],
                "entry": None,
                "stop": None,
                "targets": [],
                "invalidation": "n/a",
            },
        }

    df = df_today.copy().sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    trade_date = pd.to_datetime(df["timestamp"].iloc[-1]).date()
    windows = get_session_windows_for_date(trade_date)
    now = now_et or pd.to_datetime(df["timestamp"].iloc[-1]).to_pydatetime()

    asia = sessions.get("Asia")
    london = sessions.get("London")

    asia_range = (asia.high - asia.low) if asia else None
    london_range = (london.high - london.low) if london else None
    whipsaw_ratio = None
    if asia_range and london_range and asia_range > 0:
        whipsaw_ratio = london_range / asia_range
    whipsaw_risk = bool(whipsaw_ratio is not None and whipsaw_ratio > whipsaw_threshold)

    ny_mode = "Neutral"
    if whipsaw_risk or bool(getattr(patterns, "whipsaw", False)):
        ny_mode = "Whipsaw Risk"
    elif bool(getattr(patterns, "asia_range_sweep", False)):
        ny_mode = "AMD"
    elif bool(getattr(patterns, "london_continuation", False)):
        ny_mode = "NY Continuation"
    elif asia and london:
        if london.close > asia.high or london.close < asia.low:
            ny_mode = "NY Continuation"

    ny_direction = "Neutral"
    if getattr(patterns, "asia_range_sweep_bias", "Neutral") in {"Bullish", "Bearish"}:
        ny_direction = getattr(patterns, "asia_range_sweep_bias")
    elif getattr(patterns, "london_continuation_bias", "Neutral") in {"Bullish", "Bearish"}:
        ny_direction = getattr(patterns, "london_continuation_bias")
    elif getattr(patterns, "power_hour_bias", "Neutral") in {"Bullish", "Bearish"}:
        ny_direction = getattr(patterns, "power_hour_bias")
    elif london:
        move = london.close - london.open
        ny_direction = "Bullish" if move > 0 else "Bearish" if move < 0 else "Neutral"

    triggers: List[Dict[str, object]] = []
    for detected in [
        _detect_orb_trigger(df, 30),
        _detect_orb_trigger(df, 60),
        _detect_failed_orb_trigger(df, 30),
        _detect_failed_orb_trigger(df, 60),
        _detect_vwap_trigger(df),
    ]:
        if detected:
            triggers.append(detected)

    confirmation_triggers: List[Dict[str, object]] = []
    if getattr(patterns, "asia_range_sweep", False):
        confirmation_triggers.append(
            {
                "name": "Asia Range Sweep Bias",
                "direction": getattr(patterns, "asia_range_sweep_bias", "Neutral"),
                "time": _fmt_ts(pd.to_datetime(df["timestamp"].iloc[0])),
                "price": float(df["open"].iloc[0]),
                "timeframe": "Session",
                "details": "Overnight sweep pattern supports directional bias.",
                "category": "Overnight Confirmation",
            }
        )
    if getattr(patterns, "london_continuation", False):
        confirmation_triggers.append(
            {
                "name": "London Continuation Bias",
                "direction": getattr(patterns, "london_continuation_bias", "Neutral"),
                "time": _fmt_ts(pd.to_datetime(df["timestamp"].iloc[0])),
                "price": float(df["open"].iloc[0]),
                "timeframe": "Session",
                "details": "London continuation structure supports NY bias.",
                "category": "Overnight Confirmation",
            }
        )

    triggers = sorted(triggers, key=lambda x: x.get("time") or "")
    triggers = [_enrich_trigger(t) for t in triggers]
    primary_trigger = _pick_primary_trigger(triggers, ny_direction)
    active_trigger_names = [str(t.get("name", "")) for t in triggers]
    orb_trigger_seen = any("ORB" in name for name in active_trigger_names)

    supporting_factors: List[str] = []
    blocking_factors: List[str] = []

    if ny_mode != "Neutral":
        supporting_factors.append(f"Mode classified as {ny_mode}")
    if ny_direction in {"Bullish", "Bearish"}:
        supporting_factors.append(f"Directional bias: {ny_direction}")
    if primary_trigger:
        supporting_factors.append(f"Trigger fired: {primary_trigger['name']}")
    if confirmation_triggers:
        supporting_factors.append(
            "Bias confirmations: " + ", ".join([str(t.get("name", "n/a")) for t in confirmation_triggers[:2]])
        )

    if whipsaw_risk:
        blocking_factors.append(f"Whipsaw ratio elevated ({whipsaw_ratio:.2f})")
    if ny_direction == "Neutral":
        blocking_factors.append("No clear directional bias")

    us_start = windows.get("US", {}).get("start")
    us_confirm = us_start + dt.timedelta(minutes=60) if us_start else None
    us_end = windows.get("US", {}).get("end")
    or30_ready = us_start is not None and now >= (us_start + dt.timedelta(minutes=30))
    or60_ready = us_start is not None and now >= (us_start + dt.timedelta(minutes=60))

    trade_today = "Wait"
    primary_reason = "Waiting for session confirmation."
    confidence = 0.45
    wait_for_confirmations: List[str] = []

    if us_start and now < us_start:
        trade_today = "Wait"
        primary_reason = "US session not started yet."
        confidence = 0.55
        wait_for_confirmations = [
            "Wait for US open and OR 30m formation.",
            "Require first directional trigger aligned with daily bias.",
        ]
    elif ny_mode == "Whipsaw Risk" and not or30_ready:
        trade_today = "Wait"
        primary_reason = "Whipsaw risk detected pre-OR. Wait for OR formation before committing to an entry."
        confidence = 0.62
        wait_for_confirmations = [
            "Wait for OR 30m/60m to complete (10:00/10:30 ET).",
            "Require ORB/Failed ORB trigger aligned with daily bias.",
            "Require reaction confirmation at top-liquidity confluence.",
        ]
    elif primary_trigger is not None:
        trade_today = "Yes"
        primary_reason = f"Qualified trigger detected: {primary_trigger['name']}."
        confidence = 0.70 if ny_mode != "Whipsaw Risk" else 0.64
        wait_for_confirmations = []
    elif ny_mode == "Whipsaw Risk" and (or30_ready or or60_ready) and not orb_trigger_seen:
        trade_today = "Wait"
        primary_reason = "Whipsaw remains elevated. OR is formed; wait for OR trigger or stronger daily-bias confirmation."
        confidence = 0.60
        wait_for_confirmations = [
            "Wait for ORB 30m/60m or Failed ORB confirmation.",
            "Require trigger direction to match daily/session bias.",
            "Require volume participation and momentum agreement before entry.",
        ]
    elif us_confirm and now < us_confirm:
        trade_today = "Wait"
        primary_reason = "Waiting for opening-hour confirmation window."
        confidence = 0.60
        wait_for_confirmations = [
            "Wait for opening-hour trigger confirmation.",
            "Require confluence reaction score to improve.",
        ]
    elif us_end and now > us_end and primary_trigger is None:
        trade_today = "No"
        primary_reason = "No valid trigger occurred during the US session."
        confidence = 0.72
        wait_for_confirmations = []
    else:
        trade_today = "Wait"
        primary_reason = "Bias exists but no execution trigger has fired yet."
        confidence = 0.58
        wait_for_confirmations = [
            "Wait for primary trigger to fire.",
            "Require aligned momentum and volume confirmation.",
        ]

    last_price = float(df["close"].iloc[-1])
    day_high = float(df["high"].max())
    day_low = float(df["low"].min())

    dvwap = compute_daily_vwap(df)
    wvwap = compute_weekly_vwap(df)
    daily_vwap = float(dvwap.iloc[-1]) if not dvwap.empty else None
    weekly_vwap = float(wvwap.iloc[-1]) if not wvwap.empty else None

    vwap_levels: List[Dict[str, object]] = []
    if daily_vwap is not None:
        vwap_levels.append(
            {
                "Name": "Daily VWAP",
                "Price": daily_vwap,
                "Distance": last_price - daily_vwap,
                "Status": "Above" if last_price > daily_vwap else "Below" if last_price < daily_vwap else "At",
                "Time": _fmt_ts(pd.to_datetime(df["timestamp"].iloc[-1])),
            }
        )
    if weekly_vwap is not None:
        vwap_levels.append(
            {
                "Name": "Weekly VWAP",
                "Price": weekly_vwap,
                "Distance": last_price - weekly_vwap,
                "Status": "Above" if last_price > weekly_vwap else "Below" if last_price < weekly_vwap else "At",
                "Time": _fmt_ts(pd.to_datetime(df["timestamp"].iloc[-1])),
            }
        )

    if df_prev is not None and not df_prev.empty:
        prev_high_idx = df_prev["high"].idxmax()
        prev_low_idx = df_prev["low"].idxmin()
        anchors = [
            ("Anchored VWAP (PDH)", pd.to_datetime(df_prev.loc[prev_high_idx, "timestamp"])),
            ("Anchored VWAP (PDL)", pd.to_datetime(df_prev.loc[prev_low_idx, "timestamp"])),
        ]
        for name, anchor_ts in anchors:
            av = compute_anchored_vwap(pd.concat([df_prev, df], ignore_index=True), anchor_ts)
            if av is None or av.empty:
                continue
            px = float(av.iloc[-1])
            vwap_levels.append(
                {
                    "Name": name,
                    "Price": px,
                    "Distance": last_price - px,
                    "Status": "Above" if last_price > px else "Below" if last_price < px else "At",
                    "Time": _fmt_ts(anchor_ts),
                }
            )

    stdv_center = daily_vwap if daily_vwap is not None else last_price
    stdv_levels = _stdv_levels(df, center=stdv_center, length=20)
    volatility_metrics = _volatility_pack(df, length=20)
    volume_metrics = _volume_detector(df, length=20)
    momentum_prediction = _momentum_prediction(
        df,
        ny_direction=ny_direction,
        volatility=volatility_metrics,
        volume=volume_metrics,
    )
    vwap_probabilities = _vwap_mean_reversion_vs_expansion(
        df,
        daily_vwap=daily_vwap,
        windows=windows,
        volume=volume_metrics,
        momentum=momentum_prediction,
        volatility=volatility_metrics,
    )
    vwap_dip_strength = _vwap_strength_after_dip(df, daily_vwap=daily_vwap)
    performance_metrics = _performance_metrics(df)
    expectation_summaries = _build_expectation_summaries(
        vwap_dip=vwap_dip_strength,
        performance=performance_metrics,
        volatility=volatility_metrics,
        volume=volume_metrics,
        momentum=momentum_prediction,
    )

    confluences: List[Dict[str, object]] = []
    for z in zones:
        first_test, retest_time, retest_count = _zone_touch_times(df, z)
        touched = is_zone_touched(df, z)
        inversed = is_fvg_inversed(df, z) if z.kind == "fvg" else False
        failed = is_zone_failed(df, z) or inversed
        density, volume_score = zone_liquidity_scores(df, z)
        liquidity_score = float(density * 0.2 + volume_score)
        liquidity_level = _liquidity_level(liquidity_score)
        rejection_times = [t.strftime("%Y-%m-%d %H:%M") for t in find_rejection_candles(df, z)]

        if failed:
            status = "Invalidated"
        elif retest_count > 0:
            status = "Retested"
        elif touched:
            status = "Tested"
        else:
            status = "Fresh"

        confluences.append(
            {
                "Confluence": f"{z.timeframe} {z.kind} {z.side}",
                "Price Low": float(z.low),
                "Price High": float(z.high),
                "Formed Time": _fmt_ts(zone_formed_timestamp(z)),
                "First Test": first_test,
                "Retest Time": retest_time,
                "Retest Count": int(retest_count),
                "Rejection Times": ", ".join(rejection_times[:3]) if rejection_times else "",
                "Status": status,
                "Side": z.side,
                "Liquidity Score": liquidity_score,
                "Liquidity Level": liquidity_level,
                "Top Liquidity": False,
            }
        )

    if confluences:
        ranked = sorted(range(len(confluences)), key=lambda i: confluences[i]["Liquidity Score"], reverse=True)
        for idx in ranked[:7]:
            confluences[idx]["Top Liquidity"] = True
        confluences = sorted(
            confluences,
            key=lambda row: (not bool(row.get("Top Liquidity", False)), -float(row.get("Liquidity Score", 0.0))),
        )

    direction_sign = 1 if ny_direction == "Bullish" else -1 if ny_direction == "Bearish" else 0
    targets: List[Dict[str, object]] = []

    def _add_target(name: str, price: float, source: str):
        distance = price - last_price
        if direction_sign == 1 and distance <= 0:
            return
        if direction_sign == -1 and distance >= 0:
            return
        targets.append(
            {
                "Target": name,
                "Price": float(price),
                "Distance": float(distance),
                "Source": source,
            }
        )

    for z in zones:
        candidate = z.high if ny_direction == "Bullish" else z.low if ny_direction == "Bearish" else (z.high + z.low) / 2
        _add_target(f"{z.timeframe} {z.kind} {z.side}", float(candidate), "Zone")

    for level in vwap_levels:
        _add_target(level["Name"], float(level["Price"]), "VWAP")

    for level in stdv_levels:
        _add_target(f"STDV {level['Level']}", float(level["Price"]), "STDV")

    targets_sorted = sorted(targets, key=lambda x: abs(float(x["Distance"])))
    target_ladder = targets_sorted[:12]

    power_hour_focus = "Conditional"
    power_hour_bias = getattr(patterns, "power_hour_bias", "Neutral") or "Neutral"
    power_hour_reason = "Awaiting 14:00-16:00 development."
    power_hour_entries: List[str] = []

    if getattr(patterns, "power_hour_trend", False):
        power_hour_focus = "Yes"
        power_hour_reason = "Power-hour trend condition is active."
    elif ny_mode in {"AMD", "NY Continuation"} and ny_direction in {"Bullish", "Bearish"}:
        power_hour_focus = "Conditional"
        power_hour_bias = ny_direction
        power_hour_reason = "Directional context supports a potential late-session continuation if momentum persists."
    else:
        power_hour_focus = "No"
        power_hour_reason = "No directional alignment for late-session focus."

    if power_hour_focus in {"Yes", "Conditional"}:
        direction_word = "long" if power_hour_bias == "Bullish" else "short" if power_hour_bias == "Bearish" else "directional"
        power_hour_entries = [
            f"Look for {direction_word} pullback entries at VWAP or nearest confluence after 14:00 ET hold.",
            "Look for break-retest continuation on 1m-5m structure before entering.",
            "Avoid first impulsive candle chase; prefer retest confirmation.",
        ]

    reference_rows = _select_reference_confluence_rows(confluences, ny_direction, max_items=2)
    tracker_reference_rows = _select_reference_confluence_rows(
        confluences,
        ny_direction,
        max_items=max(1, len(confluences)),
    )
    reference_confluences = _select_reference_confluences(confluences, ny_direction, max_items=2)
    entry_style_suggestions = _suggest_confluence_entry_styles(
        tracker_reference_rows,
        df=df,
        target_ladder=target_ladder,
        mode=ny_mode,
        whipsaw_risk=whipsaw_risk,
        windows=windows,
    )
    entry_blueprints = _entry_blueprints(
        ny_mode,
        ny_direction,
        windows,
        reference_confluences=reference_confluences,
        vwap_levels=vwap_levels,
        entry_style_suggestions=entry_style_suggestions,
    )
    daily_entry_capacity = _estimate_daily_entry_capacity(
        mode=ny_mode,
        direction=ny_direction,
        trade_today=trade_today,
        confidence=float(confidence),
        reference_rows=reference_rows,
        primary_trigger=primary_trigger,
        now=now,
        windows=windows,
    )
    risk_engine = _risk_engine(
        last_price=last_price,
        direction=ny_direction,
        reference_rows=reference_rows,
        target_ladder=target_ladder,
        volatility=volatility_metrics,
        volume=volume_metrics,
        momentum=momentum_prediction,
        vwap_prob=vwap_probabilities,
    )
    entry_style_suggestions = _attach_entry_confidence(
        entry_styles=entry_style_suggestions,
        risk_engine=risk_engine,
        momentum=momentum_prediction,
        volume=volume_metrics,
        vwap_prob=vwap_probabilities,
        ny_direction=ny_direction,
        trade_today=trade_today,
        primary_trigger=primary_trigger,
        whipsaw_risk=whipsaw_risk,
    )
    entry_execution_tracker = _evaluate_entry_executions(
        df=df,
        entry_styles=entry_style_suggestions,
        risk_engine=risk_engine,
        windows=windows,
    )

    # Confidence blend: retain existing logic but adjust using confluence analytics.
    confidence += (float(risk_engine.get("quality_score", 50.0)) - 50.0) / 200.0
    confidence += (float(momentum_prediction.get("confidence", 0.0)) - 20.0) / 400.0
    if float(vwap_probabilities.get("expansion_prob", 50.0)) > float(vwap_probabilities.get("mean_reversion_prob", 50.0)):
        confidence += 0.03
    else:
        confidence -= 0.02
    confidence = max(0.20, min(0.92, confidence))

    if trade_today == "Yes" and str(risk_engine.get("status", "Caution")) == "Caution":
        trade_today = "Wait"
        primary_reason = "Initial trigger detected, but risk-quality is weak. Wait for better confirmation before entry."
        wait_for_confirmations = [
            "Wait for trigger retest hold on the trigger side.",
            "Require confluence reaction score >= 55.",
            "Require RVOL to improve toward >= 1.0.",
            "Require momentum to align with entry side.",
        ]
    if trade_today == "Wait" and str(risk_engine.get("status", "Caution")) == "High" and primary_trigger is not None:
        trade_today = "Yes"
        primary_reason = "Trigger + risk-quality alignment upgraded execution readiness."
        wait_for_confirmations = []

    # User-facing whipsaw labeling.
    if ny_mode == "Whipsaw Risk":
        trade_today = "Tradeable Whipsaw" if trade_today == "Yes" else "Untradeable Whipsaw"

    if _trade_decision_bucket(trade_today) in {"Wait", "Yes"} and ny_direction in {"Bullish", "Bearish"}:
        supporting_factors.append(
            f"Risk engine: {risk_engine.get('status', 'Caution')} ({risk_engine.get('quality_score', 0.0):.1f})"
        )
    if str(volume_metrics.get("state", "normal")) == "low":
        blocking_factors.append("Volume participation is below normal")
    if ny_direction in {"Bullish", "Bearish"}:
        m_pred = str(momentum_prediction.get("predicted", "Neutral"))
        if m_pred in {"Bullish", "Bearish"} and m_pred != ny_direction:
            blocking_factors.append("Momentum prediction conflicts with directional bias")

    trigger_watchlist: List[Dict[str, object]] = []
    if us_start:
        trigger_watchlist.append(
            {
                "watch": "ORB 30m Confirmation",
                "status": "Triggered" if any("ORB 30m" in n for n in active_trigger_names) else "Monitoring",
                "ready_time": _fmt_ts(pd.to_datetime(us_start + dt.timedelta(minutes=30))),
                "required_for": "Intraday confirmation",
            }
        )
        trigger_watchlist.append(
            {
                "watch": "ORB 60m Confirmation",
                "status": "Triggered" if any("ORB 60m" in n for n in active_trigger_names) else "Monitoring",
                "ready_time": _fmt_ts(pd.to_datetime(us_start + dt.timedelta(minutes=60))),
                "required_for": "Stronger continuation confirmation",
            }
        )

    for c in confirmation_triggers:
        trigger_watchlist.append(
            {
                "watch": c.get("name", "Bias Confirmation"),
                "status": "Active",
                "ready_time": c.get("time", "n/a"),
                "required_for": "Daily bias alignment",
            }
        )
    open_pattern_watch = _us_open_reclaim_watch(df=df, trade_date=trade_date, now=now, confluences=confluences)

    return {
        "decision": {
            "trade_today": trade_today,
            "ny_mode": ny_mode,
            "ny_direction": ny_direction,
            "confidence": float(confidence),
            "primary_reason": primary_reason,
            "wait_for_confirmations": wait_for_confirmations,
            "supporting_factors": supporting_factors,
            "blocking_factors": blocking_factors,
            "whipsaw_ratio": float(whipsaw_ratio) if whipsaw_ratio is not None else None,
        },
        "triggers": triggers,
        "confirmation_triggers": confirmation_triggers,
        "trigger_watchlist": trigger_watchlist,
        "primary_trigger": primary_trigger,
        "confluences": confluences,
        "targets": target_ladder,
        "vwap_levels": vwap_levels,
        "stdv_levels": stdv_levels,
        "volatility_metrics": volatility_metrics,
        "volume_detector": volume_metrics,
        "momentum_prediction": momentum_prediction,
        "vwap_probabilities": vwap_probabilities,
        "vwap_strength_after_dip": vwap_dip_strength,
        "risk_engine": risk_engine,
        "entry_execution_tracker": entry_execution_tracker,
        "performance_metrics": performance_metrics,
        "expectation_summaries": expectation_summaries,
        "power_hour": {
            "focus": power_hour_focus,
            "bias": power_hour_bias,
            "reason": power_hour_reason,
            "entries": power_hour_entries,
        },
        "entry_blueprints": entry_blueprints,
        "confluence_entry_styles": entry_style_suggestions,
        "daily_entry_capacity": daily_entry_capacity,
        "timeframe_playbook": _timeframe_playbook(ny_mode),
        "open_pattern_watch": open_pattern_watch,
        "market_snapshot": {
            "last_price": last_price,
            "day_high": day_high,
            "day_low": day_low,
            "trade_date": str(trade_date),
        },
    }
