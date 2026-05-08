import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd

from engines.bias import build_bias
from engines.patterns import detect_patterns
from engines.sessions import compute_session_stats
from engines.strategy_playbook import build_strategy_playbook
from engines.volume_profile import build_session_profiles, trading_day_bounds
from engines.zones import build_htf_zones
from indicators.moving_averages import compute_daily_vwap
from indicators.momentum import roc
from indicators.statistics import zscore
from indicators.volatility import atr_like


def _prepare_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = df.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in out.columns]
    out = out[keep].sort_values("timestamp").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _blend(components: List[Dict[str, object]]) -> Dict[str, object]:
    weighted = 0.0
    total = 0.0
    for row in components:
        bias = str(row.get("bias", "Neutral"))
        if bias == "Bullish":
            sign = 1.0
        elif bias == "Bearish":
            sign = -1.0
        else:
            sign = 0.0
        conf = max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0)))
        if sign == 0.0 or conf <= 0:
            continue
        weighted += sign * conf
        total += conf

    if total <= 0:
        return {"bias": "Neutral", "confidence": 0.0, "score": 0.0, "tone": "Balanced"}

    score = weighted / total
    confidence = abs(score)
    if score >= 0.60:
        return {"bias": "Bullish", "confidence": confidence, "score": score, "tone": "Strong bullish"}
    if score >= 0.25:
        return {"bias": "Bullish", "confidence": confidence, "score": score, "tone": "Constructive bullish"}
    if score >= 0.08:
        return {"bias": "Bullish", "confidence": confidence, "score": score, "tone": "Slight bullish tilt"}
    if score <= -0.60:
        return {"bias": "Bearish", "confidence": confidence, "score": score, "tone": "Strong bearish"}
    if score <= -0.25:
        return {"bias": "Bearish", "confidence": confidence, "score": score, "tone": "Defensive bearish"}
    if score <= -0.08:
        return {"bias": "Bearish", "confidence": confidence, "score": score, "tone": "Slight bearish tilt"}
    return {"bias": "Neutral", "confidence": confidence, "score": score, "tone": "Balanced"}


def _market_component_snapshot(
    day_df: pd.DataFrame,
    prev_day_df: pd.DataFrame,
    trading_date: dt.date,
) -> Dict[str, object]:
    td_start, _ = trading_day_bounds(trading_date)
    us_open = pd.Timestamp.combine(trading_date, dt.time(9, 30))
    ny_open_end = pd.Timestamp.combine(trading_date, dt.time(9, 45))

    overnight = day_df[(day_df["timestamp"] >= td_start) & (day_df["timestamp"] < us_open)].copy()
    ny_open_df = day_df[(day_df["timestamp"] >= us_open) & (day_df["timestamp"] <= ny_open_end)].copy()

    last_price = float(day_df["close"].iloc[-1]) if not day_df.empty else 0.0
    dvwap = compute_daily_vwap(day_df)
    vwap = float(dvwap.iloc[-1]) if len(dvwap) else last_price
    vwap_bias = "Bullish" if last_price > vwap else "Bearish" if last_price < vwap else "Neutral"

    z = zscore(day_df["close"], length=20)
    z_last = float(z.iloc[-1]) if len(z) and pd.notna(z.iloc[-1]) else 0.0
    atr = atr_like(day_df, length=14)
    atr_last = float(atr.iloc[-1]) if len(atr) and pd.notna(atr.iloc[-1]) else 0.0
    roc_series = roc(day_df["close"], length=10)
    roc_last = float(roc_series.iloc[-1]) if len(roc_series) and pd.notna(roc_series.iloc[-1]) else 0.0

    momentum_bias = "Bullish" if roc_last > 0 else "Bearish" if roc_last < 0 else "Neutral"

    overnight_high = float(overnight["high"].max()) if not overnight.empty else float("nan")
    overnight_low = float(overnight["low"].min()) if not overnight.empty else float("nan")
    ny_open_price = float(ny_open_df["open"].iloc[0]) if not ny_open_df.empty else last_price
    if pd.notna(overnight_high) and ny_open_price > overnight_high:
        ny_open_bias = "Bullish"
    elif pd.notna(overnight_low) and ny_open_price < overnight_low:
        ny_open_bias = "Bearish"
    else:
        ny_open_bias = "Neutral"

    profiles = build_session_profiles(day_df, trading_date, tick_size=0.25, value_area_pct=0.70)
    full = profiles.get("Full Day")
    profile_bias = "Neutral"
    if full is not None and full.vah is not None and full.val is not None:
        if last_price > float(full.vah):
            profile_bias = "Bullish"
        elif last_price < float(full.val):
            profile_bias = "Bearish"

    daily_votes = [vwap_bias, momentum_bias, profile_bias]
    ny_open_votes = [ny_open_bias, vwap_bias, momentum_bias]

    daily_blend = _blend([{"bias": b, "confidence": 0.60} for b in daily_votes])
    ny_open_blend = _blend([{"bias": b, "confidence": 0.58} for b in ny_open_votes])

    prev_close_location = "n/a"
    if prev_day_df is not None and not prev_day_df.empty:
        prev_profiles = build_session_profiles(prev_day_df, trading_date - dt.timedelta(days=1), tick_size=0.25, value_area_pct=0.70)
        prev_full = prev_profiles.get("Full Day")
        if prev_full is not None:
            prev_close_location = str(prev_full.close_location)

    return {
        "daily_bias": daily_blend["bias"],
        "daily_conf": float(daily_blend["confidence"]),
        "ny_open_bias": ny_open_blend["bias"],
        "ny_open_conf": float(ny_open_blend["confidence"]),
        "summary": (
            f"Market statistics snapshot: VWAP posture {vwap_bias.lower()}, momentum {momentum_bias.lower()}, "
            f"profile location {profile_bias.lower()}, z-score {z_last:+.2f}, ATR {atr_last:.2f}, "
            f"previous close location {prev_close_location}."
        ),
    }


def _expected_behavior(bias: str, horizon: str) -> str:
    if bias == "Bullish":
        return (
            f"{horizon} expectation: price should seek higher acceptance above value references, "
            "show buy-side defense on pullbacks, and expand if opening structure holds."
        )
    if bias == "Bearish":
        return (
            f"{horizon} expectation: price should accept lower against value references, "
            "show sell-side defense on bounces, and extend if opening structure holds."
        )
    return (
        f"{horizon} expectation: rotational behavior is favored, with two-way trade around value until one side gains acceptance."
    )


def build_unified_bias(
    df_today: Optional[pd.DataFrame],
    df_prev: Optional[pd.DataFrame],
    trading_date: dt.date,
    now_et: dt.datetime,
) -> Dict[str, object]:
    today_df = _prepare_df(df_today)
    prev_df = _prepare_df(df_prev)

    td_start, td_end = trading_day_bounds(trading_date)
    day_df = today_df[(today_df["timestamp"] >= td_start) & (today_df["timestamp"] <= td_end)].copy()
    prev_td_start, prev_td_end = trading_day_bounds(trading_date - dt.timedelta(days=1))
    prev_day_df = prev_df[(prev_df["timestamp"] >= prev_td_start) & (prev_df["timestamp"] <= prev_td_end)].copy()

    if day_df.empty:
        return {
            "daily": {
                "bias": "Neutral",
                "confidence": 0.0,
                "score": 0.0,
                "tone": "Balanced",
                "expected": "No intraday data available.",
                "reasoning": "Unified reasoning is unavailable because current trading-day bars are missing.",
                "finalized": False,
                "finalized_at": "10:45 ET",
            },
            "ny_open": {
                "bias": "Neutral",
                "confidence": 0.0,
                "score": 0.0,
                "tone": "Balanced",
                "expected": "No intraday data available.",
                "reasoning": "Unified reasoning is unavailable because current trading-day bars are missing.",
                "finalized": False,
                "finalized_at": "09:15 ET",
            },
            "components_daily": [],
            "components_ny_open": [],
            "updated_at": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
        }

    session_source = (
        pd.concat([prev_day_df, day_df], ignore_index=True).sort_values("timestamp")
        if not prev_day_df.empty
        else day_df.copy()
    )
    sessions = compute_session_stats(session_source, trading_date)
    patterns = detect_patterns(sessions, day_df, prev_day_df if not prev_day_df.empty else None)
    zones = build_htf_zones(session_source)

    live_bias = build_bias(
        day_df,
        prev_day_df,
        sessions,
    )
    playbook = build_strategy_playbook(
        df_today=day_df,
        df_prev=prev_day_df if not prev_day_df.empty else None,
        sessions=sessions,
        patterns=patterns,
        zones=zones,
        now_et=now_et,
        trading_day=trading_date,
    )
    market = _market_component_snapshot(day_df, prev_day_df, trading_date)

    pb_direction = str(playbook.get("decision", {}).get("ny_direction", "Neutral"))
    pb_conf = float(playbook.get("decision", {}).get("confidence", 0.0) or 0.0)
    pb_reason = str(playbook.get("decision", {}).get("primary_reason", "n/a"))

    daily_components = [
        {
            "name": "Live Analysis Daily Bias",
            "bias": getattr(live_bias, "daily_bias", "Neutral"),
            "confidence": float(getattr(live_bias, "daily_confidence", 0.0) or 0.0),
            "finalized": bool(getattr(live_bias, "daily_finalized", False)),
            "finalized_at": "10:45 ET",
        },
        {
            "name": "Market Statistics Composite",
            "bias": str(market.get("daily_bias", "Neutral")),
            "confidence": float(market.get("daily_conf", 0.0) or 0.0),
            "finalized": now_et.date() > trading_date or (now_et.date() == trading_date and now_et.time() >= dt.time(10, 45)),
            "finalized_at": "10:45 ET",
        },
        {
            "name": "Strategy Playbook Direction",
            "bias": pb_direction,
            "confidence": pb_conf,
            "finalized": now_et.date() > trading_date or (now_et.date() == trading_date and now_et.time() >= dt.time(10, 45)),
            "finalized_at": "10:45 ET",
        },
    ]

    ny_open_label = getattr(live_bias, "us_open_bias_30", None) or getattr(live_bias, "us_open_bias", "Neutral")
    ny_open_conf = float(getattr(live_bias, "us_open_confidence_30", 0.0) or 0.0)
    ny_open_components = [
        {
            "name": "Live Analysis NY Open Bias",
            "bias": ny_open_label,
            "confidence": ny_open_conf,
            "finalized": now_et.date() > trading_date or (now_et.date() == trading_date and now_et.time() >= dt.time(9, 15)),
            "finalized_at": "09:15 ET",
        },
        {
            "name": "Market Statistics NY Open Composite",
            "bias": str(market.get("ny_open_bias", "Neutral")),
            "confidence": float(market.get("ny_open_conf", 0.0) or 0.0),
            "finalized": now_et.date() > trading_date or (now_et.date() == trading_date and now_et.time() >= dt.time(9, 15)),
            "finalized_at": "09:15 ET",
        },
        {
            "name": "Strategy Playbook Opening Lean",
            "bias": pb_direction,
            "confidence": max(0.0, min(1.0, pb_conf * 0.85)),
            "finalized": now_et.date() > trading_date or (now_et.date() == trading_date and now_et.time() >= dt.time(9, 15)),
            "finalized_at": "09:15 ET",
        },
    ]

    daily_blend = _blend(daily_components)
    ny_open_blend = _blend(ny_open_components)

    daily_reasoning = (
        f"Unified daily reasoning blends Live Analysis (daily bias {getattr(live_bias, 'daily_bias', 'Neutral')} at "
        f"{float(getattr(live_bias, 'daily_confidence', 0.0) or 0.0):.0%}), Market Statistics ({market.get('summary', 'n/a')}), "
        f"and Strategy Playbook direction ({pb_direction} at {pb_conf:.0%}; reason: {pb_reason}). "
        f"This confidence-blended synthesis reduces single-model dominance and reflects cross-tab agreement/disagreement."
    )
    ny_open_reasoning = (
        f"Unified NY Open reasoning blends Live pre-open read ({ny_open_label} at {ny_open_conf:.0%}), "
        f"Market Statistics opening context ({str(market.get('ny_open_bias', 'Neutral')).lower()}), and Strategy Playbook opening lean "
        f"({pb_direction} at {max(0.0, min(1.0, pb_conf * 0.85)):.0%}). "
        "This creates one synchronized opening posture used consistently across all tabs."
    )

    daily_finalized = now_et.date() > trading_date or (now_et.date() == trading_date and now_et.time() >= dt.time(10, 45))
    ny_open_finalized = now_et.date() > trading_date or (now_et.date() == trading_date and now_et.time() >= dt.time(9, 15))

    return {
        "daily": {
            "bias": str(daily_blend.get("bias", "Neutral")),
            "confidence": float(daily_blend.get("confidence", 0.0) or 0.0),
            "score": float(daily_blend.get("score", 0.0) or 0.0),
            "tone": str(daily_blend.get("tone", "Balanced")),
            "expected": _expected_behavior(str(daily_blend.get("bias", "Neutral")), "Daily"),
            "reasoning": daily_reasoning,
            "finalized": daily_finalized,
            "finalized_at": "10:45 ET",
        },
        "ny_open": {
            "bias": str(ny_open_blend.get("bias", "Neutral")),
            "confidence": float(ny_open_blend.get("confidence", 0.0) or 0.0),
            "score": float(ny_open_blend.get("score", 0.0) or 0.0),
            "tone": str(ny_open_blend.get("tone", "Balanced")),
            "expected": _expected_behavior(str(ny_open_blend.get("bias", "Neutral")), "NY Open"),
            "reasoning": ny_open_reasoning,
            "finalized": ny_open_finalized,
            "finalized_at": "09:15 ET",
        },
        "components_daily": daily_components,
        "components_ny_open": ny_open_components,
        "updated_at": now_et.strftime("%Y-%m-%d %H:%M:%S ET"),
    }
