from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from engines.zones import ZoneConfluence
from storage.history_manager import BiasSummary, PatternSummary, SessionStats, TradeSuggestion


def build_trade_suggestion(bias: BiasSummary) -> TradeSuggestion:
    """
    Suggests whether to trade, trade small, or stand aside based on
    Daily Bias, US Open Bias, and confidence.
    """
    daily = bias.daily_bias
    us = getattr(bias, "us_open_bias_60", None) or bias.us_open_bias
    conf = bias.daily_confidence
    explanation = getattr(bias, "explanation", "") or ""
    vwap_comment = getattr(bias, "vwap_comment", "") or ""

    if "not finalized until 10:45 ET" in explanation:
        action = "Wait"
        rationale = (
            "Daily bias is not finalized yet (60m OR pending). "
            "Wait for the 10:45 ET confirmation before taking directional trades."
        )
        if us in ("Bullish", "Bearish"):
            rationale += f" Potential early move: US Open bias is {us.lower()}—look for confirmation."
        if "Suggestion:" in vwap_comment:
            vwap_hint = vwap_comment.split("Suggestion:", 1)[-1].strip()
            rationale += f" VWAP cue: {vwap_hint}"
        return TradeSuggestion(action=action, rationale=rationale)

    if daily == "Neutral" or us == "Neutral":
        action = "Don't Trade"
        rationale = (
            "Bias is Neutral or conflicting; structure does not provide a clear edge. "
            "Stand aside until a clearer directional picture emerges."
        )
        if us in ("Bullish", "Bearish"):
            rationale += f" Potential early move: US Open bias is {us.lower()}—wait for confirmation."
        if "Suggestion:" in vwap_comment:
            vwap_hint = vwap_comment.split("Suggestion:", 1)[-1].strip()
            rationale += f" VWAP cue: {vwap_hint}"
    elif daily == us and conf >= 0.7:
        action = "Trade"
        rationale = (
            f"Daily Bias and US Open Bias are both {daily} with high confidence ({conf:.2f}). "
            "Sessions, VWAP posture, and supporting indicators align—this is a day to trade "
            "in the direction of the bias."
        )
    elif daily == us and 0.5 <= conf < 0.7:
        action = "Trade Small"
        rationale = (
            f"Daily Bias and US Open Bias are both {daily}, but confidence is moderate ({conf:.2f}). "
            "There is some edge, but mixed conditions suggest reduced size."
        )
    else:
        action = "Don't Trade"
        rationale = (
            f"Daily Bias is {daily} but US Open Bias is {us}, indicating conflict between sessions. "
            "When higher-timeframe and US session structure disagree, the safest choice is not to trade."
        )
        if "Suggestion:" in vwap_comment:
            vwap_hint = vwap_comment.split("Suggestion:", 1)[-1].strip()
            rationale += f" VWAP cue: {vwap_hint}"

    return TradeSuggestion(action=action, rationale=rationale)


@dataclass
class PostBiasGuidance:
    summary: str
    bullets: List[str]


def _bias_finalized(explanation: str) -> bool:
    return "not finalized until 10:45 ET" not in (explanation or "")


def _format_level(name: str, value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{name} {value:.2f}"


def _direction_focus(daily: str, us: str) -> str:
    if daily in ("Bullish", "Bearish"):
        return daily
    if us in ("Bullish", "Bearish"):
        return us
    return "Neutral"


def _session_context(sessions: dict) -> Optional[str]:
    if not sessions:
        return None
    labels = []
    for name in ("Asia", "London", "US"):
        sess = sessions.get(name)
        if not isinstance(sess, SessionStats):
            continue
        move = sess.close - sess.open
        threshold = 0.1 * sess.range if sess.range else 0.0
        if abs(move) < threshold:
            labels.append(f"{name} range")
        elif move > 0:
            labels.append(f"{name} up")
        else:
            labels.append(f"{name} down")
    if not labels:
        return None
    return "; ".join(labels)


def build_post_bias_guidance(
    bias: BiasSummary,
    sessions: dict,
    patterns: PatternSummary,
    zone_confluence: Optional[ZoneConfluence],
    trend_bias: str,
    df_today: Optional[pd.DataFrame] = None,
    df_prev: Optional[pd.DataFrame] = None,
) -> PostBiasGuidance:
    explanation = getattr(bias, "explanation", "") or ""
    daily = getattr(bias, "daily_bias", "Neutral")
    us = getattr(bias, "us_open_bias_60", None) or getattr(bias, "us_open_bias", "Neutral")
    conf = float(getattr(bias, "daily_confidence", 0.0) or 0.0)

    if not _bias_finalized(explanation):
        summary = (
            "Daily bias is not finalized yet. Entry guidance will lock in after 10:45 ET "
            "once the 60m opening range confirms direction."
        )
        bullets = [
            f"US Open bias is {us.lower()} if you need an early lean, but require confirmation.",
        ]
        return PostBiasGuidance(summary=summary, bullets=bullets)

    focus = _direction_focus(daily, us)
    if daily in ("Bullish", "Bearish") and us in ("Bullish", "Bearish") and daily != us:
        alignment_note = "Daily and US Open biases conflict, so be selective and wait for clean reactions."
    else:
        alignment_note = "Daily and US Open biases align or are neutral, so continuation setups have priority."

    if focus == "Bullish":
        focus_text = "longs"
        reaction_text = "support holds and pullbacks into value"
    elif focus == "Bearish":
        focus_text = "shorts"
        reaction_text = "resistance holds and pullbacks into value"
    else:
        focus_text = "balanced or mean-revert setups"
        reaction_text = "clear range edges and failed breaks"

    summary = (
        f"Bias is finalized with a {daily.lower()} daily bias ({conf:.0%}) and "
        f"{us.lower()} US Open bias. For the rest of the day, prioritize {focus_text} "
        f"and look for {reaction_text}. {alignment_note}"
    )

    bullets: List[str] = []

    levels: List[str] = []
    prev_high = None
    prev_low = None
    day_high = None
    day_low = None
    if df_prev is not None and not df_prev.empty:
        prev_high = float(df_prev["high"].max())
        prev_low = float(df_prev["low"].min())
    if df_today is not None and not df_today.empty:
        day_high = float(df_today["high"].max())
        day_low = float(df_today["low"].min())

    for level in (
        _format_level("Prior day high", prev_high),
        _format_level("Prior day low", prev_low),
        _format_level("Current day high", day_high),
        _format_level("Current day low", day_low),
    ):
        if level:
            levels.append(level)
    if levels:
        bullets.append(f"Key levels: {', '.join(levels)}.")

    vwap_comment = getattr(bias, "vwap_comment", "") or ""
    if "Suggestion:" in vwap_comment:
        vwap_hint = vwap_comment.split("Suggestion:", 1)[-1].strip()
        if vwap_hint:
            bullets.append(f"VWAP cue: {vwap_hint}")

    pattern_cues: List[str] = []
    if getattr(patterns, "orb_60", False):
        pattern_cues.append(f"ORB 60m {getattr(patterns, 'orb_60_bias', 'Neutral').lower()} break")
    if getattr(patterns, "failed_orb_60", False):
        pattern_cues.append(f"Failed ORB 60m {getattr(patterns, 'failed_orb_60_bias', 'Neutral').lower()} fade")
    if getattr(patterns, "vwap_reclaim_reject", False):
        pattern_cues.append(
            f"VWAP {getattr(patterns, 'vwap_reclaim_reject_bias', 'Neutral').lower()} reclaim/reject"
        )
    if getattr(patterns, "london_continuation", False):
        pattern_cues.append(
            f"London continuation {getattr(patterns, 'london_continuation_bias', 'Neutral').lower()}"
        )
    if getattr(patterns, "asia_range_sweep", False):
        pattern_cues.append(
            f"Asia range sweep {getattr(patterns, 'asia_range_sweep_bias', 'Neutral').lower()}"
        )
    if getattr(patterns, "us_open_gap_fill", False):
        pattern_cues.append(
            f"US open gap fill {getattr(patterns, 'us_open_gap_fill_bias', 'Neutral').lower()}"
        )
    if pattern_cues:
        bullets.append(f"Pattern cues: {', '.join(pattern_cues)}.")

    or30_failed = getattr(patterns, "failed_orb_30", False)
    or30_break = getattr(patterns, "orb_30", False)
    or30_bias = getattr(patterns, "failed_orb_30_bias", None) or getattr(patterns, "orb_30_bias", None)
    or60_break = getattr(patterns, "orb_60", False)
    or60_failed = getattr(patterns, "failed_orb_60", False)
    if or30_failed:
        bullets.append(
            f"30m OR failed ({getattr(patterns, 'failed_orb_30_bias', 'Neutral').lower()}); "
            "higher odds the 60m OR fails unless price reclaims and holds the range."
        )
    elif or30_break:
        bullets.append(
            f"30m OR held with a {getattr(patterns, 'orb_30_bias', 'Neutral').lower()} break; "
            "60m OR more likely to hold unless a clean reversal setup appears."
        )
    elif or30_bias in ("Bullish", "Bearish"):
        bullets.append(
            f"30m OR is mixed ({or30_bias.lower()}); treat the 60m OR as the decision point."
        )

    if or60_failed:
        bullets.append(
            f"60m OR has already failed ({getattr(patterns, 'failed_orb_60_bias', 'Neutral').lower()}); "
            "shift to fade and mean-revert setups until structure resets."
        )
    elif or60_break:
        bullets.append(
            f"60m OR break confirmed ({getattr(patterns, 'orb_60_bias', 'Neutral').lower()}); "
            "favor continuation in that direction on pullbacks."
        )

    if zone_confluence is not None and zone_confluence.bias in ("Bullish", "Bearish"):
        bullets.append(
            f"HTF zones lean {zone_confluence.bias.lower()} (score {zone_confluence.score:.2f}); "
            f"prioritize reactions at those zones."
        )

    if trend_bias in ("Bullish", "Bearish"):
        bullets.append(
            f"Trend context: {trend_bias.lower()} trend; favor continuation after confirmed breaks."
        )

    if getattr(patterns, "whipsaw", False):
        bullets.append("Whipsaw risk: wait for clear confirmation before entry.")

    news_comment = getattr(bias, "news_comment", "") or ""
    if news_comment and "no" not in news_comment.lower():
        bullets.append(f"News risk: {news_comment}")

    session_note = _session_context(sessions)
    if session_note:
        bullets.append(f"Session tape: {session_note}.")

    if not bullets:
        bullets.append("No strong confluences detected; trade smaller or wait for clean structure.")

    return PostBiasGuidance(summary=summary, bullets=bullets)