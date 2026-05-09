import datetime as dt
import pandas as pd
from typing import Optional

from storage.history_manager import AccuracySummary, BiasSummary
from data.session_reference import get_session_windows_for_date
from engines.probability import bias_probabilities


def _slice_trading_day(df: pd.DataFrame, trading_date: Optional[dt.date]) -> pd.DataFrame:
    if df is None or df.empty or trading_date is None or "timestamp" not in df.columns:
        return df
    start = dt.datetime.combine(trading_date - dt.timedelta(days=1), dt.time(18, 0))
    end = dt.datetime.combine(trading_date, dt.time(17, 0))
    sdf = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if sdf is None or sdf.empty:
        return sdf
    return sdf.sort_values("timestamp")


def _actual_direction(df_today: pd.DataFrame) -> str:
    if df_today is None or df_today.empty:
        return "Neutral"
    sdf = df_today.sort_values("timestamp") if "timestamp" in df_today.columns else df_today
    open_ = sdf["open"].iloc[0]
    close = sdf["close"].iloc[-1]
    diff = close - open_
    if diff > 0:
        return "Bullish"
    if diff < 0:
        return "Bearish"
    return "Neutral"


def _actual_direction_from_window(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Neutral"
    open_ = float(df["open"].iloc[0])
    close = float(df["close"].iloc[-1])
    diff = close - open_
    if diff > 0:
        return "Bullish"
    if diff < 0:
        return "Bearish"
    return "Neutral"


def _actual_direction_window(
    df_today: pd.DataFrame,
    minutes: int,
    trading_date: Optional[dt.date] = None,
) -> str:
    if df_today is None or df_today.empty or "timestamp" not in df_today.columns:
        return "Neutral"
    date = trading_date or df_today["timestamp"].iloc[0].date()
    start = pd.Timestamp.combine(date, pd.Timestamp("09:30").time())
    end = start + pd.Timedelta(minutes=minutes)
    window = df_today[(df_today["timestamp"] >= start) & (df_today["timestamp"] <= end)]
    if window.empty:
        return "Neutral"
    open_ = window["open"].iloc[0]
    close = window["close"].iloc[-1]
    diff = close - open_
    if diff > 0:
        return "Bullish"
    if diff < 0:
        return "Bearish"
    return "Neutral"


def _slice_window(df: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    sdf = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if sdf is None or sdf.empty:
        return pd.DataFrame()
    return sdf.sort_values("timestamp")


def _close_position_pct(df_trade: pd.DataFrame) -> Optional[float]:
    if df_trade is None or df_trade.empty:
        return None
    high = float(df_trade["high"].max())
    low = float(df_trade["low"].min())
    close = float(df_trade["close"].iloc[-1])
    rng = max(high - low, 1e-6)
    return ((close - low) / rng) * 100.0


def _close_quality_label(position_pct: Optional[float]) -> tuple[str, str]:
    if position_pct is None:
        return ("n/a", "n/a")
    if position_pct <= 10:
        return ("capitulative close", "very weak")
    if position_pct <= 30:
        return ("defensive close", "weak")
    if position_pct <= 45:
        return ("fading close", "slightly weak")
    if position_pct <= 55:
        return ("balanced close", "normal")
    if position_pct <= 70:
        return ("constructive close", "normal")
    if position_pct <= 90:
        return ("initiative close", "strong")
    return ("dominant close", "very strong")


def _session_accuracy_map(
    df_source: pd.DataFrame,
    bias: BiasSummary,
    trading_date: dt.date,
) -> dict:
    windows = get_session_windows_for_date(trading_date)
    trading_start = dt.datetime.combine(trading_date - dt.timedelta(days=1), dt.time(18, 0))
    market_open_end = windows["Asia"]["start"] - dt.timedelta(minutes=15)
    ny_open_start = windows["US"]["start"]
    ny_open_end = dt.datetime.combine(trading_date, dt.time(9, 45))

    # Prior trading day is used to form pre-session market-open expectations.
    prev_date = trading_date - dt.timedelta(days=1)
    prev_start = dt.datetime.combine(prev_date - dt.timedelta(days=1), dt.time(18, 0))
    prev_end = dt.datetime.combine(prev_date, dt.time(17, 0))

    prev_df = _slice_window(df_source, prev_start, prev_end)
    prev_dir = _actual_direction_from_window(prev_df)

    market_open_df = _slice_window(df_source, trading_start, market_open_end)
    asia_df = _slice_window(df_source, windows["Asia"]["start"], windows["Asia"]["end"])
    london_df = _slice_window(df_source, windows["London"]["start"], windows["London"]["end"])
    ny_open_df = _slice_window(df_source, ny_open_start, ny_open_end)
    ny_df = _slice_window(df_source, windows["US"]["start"], windows["US"]["end"])

    asia_actual = _actual_direction_from_window(asia_df)

    daily = bias.daily_bias if bias.daily_bias in ("Bullish", "Bearish") else "Neutral"
    market_open_pred = prev_dir if prev_dir in ("Bullish", "Bearish") else daily
    asia_pred = market_open_pred if market_open_pred in ("Bullish", "Bearish") else daily
    london_pred = asia_actual if asia_actual in ("Bullish", "Bearish") else asia_pred
    ny_open_pred = (
        bias.us_open_bias_30
        if getattr(bias, "us_open_bias_30", None) in ("Bullish", "Bearish")
        else bias.us_open_bias
    )
    ny_session_pred = (
        bias.us_open_bias_60
        if getattr(bias, "us_open_bias_60", None) in ("Bullish", "Bearish")
        else daily
    )

    market_open_conf = max(0.35, min(0.90, float(getattr(bias, "daily_confidence", 0.5) or 0.5) * 0.75))
    asia_conf = market_open_conf
    london_conf = market_open_conf
    ny_open_conf = float(
        getattr(bias, "us_open_confidence_30", None)
        if getattr(bias, "us_open_confidence_30", None) is not None
        else getattr(bias, "us_open_confidence", 0.5)
    )
    ny_open_conf = max(0.0, min(1.0, ny_open_conf))
    ny_session_conf = float(
        getattr(bias, "us_open_confidence_60", None)
        if getattr(bias, "us_open_confidence_60", None) is not None
        else getattr(bias, "daily_confidence", 0.5)
    )
    ny_session_conf = max(0.0, min(1.0, ny_session_conf))

    def _row(predicted: str, actual: str, finalized_at: str, confidence: float) -> dict:
        bull_prob, bear_prob = bias_probabilities(predicted, confidence)
        if abs(bull_prob - bear_prob) < 0.01:
            favored_side = "Neutral"
        else:
            favored_side = "Bullish" if bull_prob > bear_prob else "Bearish"

        favorability_correct = None
        if favored_side in ("Bullish", "Bearish") and actual in ("Bullish", "Bearish"):
            favorability_correct = favored_side == actual

        if predicted not in ("Bullish", "Bearish"):
            return {
                "predicted": predicted,
                "actual": actual,
                "correct": None,
                "confidence": confidence,
                "prob_bullish": bull_prob,
                "prob_bearish": bear_prob,
                "favored_side": favored_side,
                "favorability_correct": favorability_correct,
                "finalized_at": finalized_at,
            }
        return {
            "predicted": predicted,
            "actual": actual,
            "correct": predicted == actual,
            "confidence": confidence,
            "prob_bullish": bull_prob,
            "prob_bearish": bear_prob,
            "favored_side": favored_side,
            "favorability_correct": favorability_correct,
            "finalized_at": finalized_at,
        }

    return {
        "market_open": _row(market_open_pred, _actual_direction_from_window(market_open_df), "17:30 ET", market_open_conf),
        "asia": _row(asia_pred, _actual_direction_from_window(asia_df), "Asia +15m ET", asia_conf),
        "london": _row(london_pred, _actual_direction_from_window(london_df), "London +15m ET", london_conf),
        "ny_open": _row(ny_open_pred, _actual_direction_from_window(ny_open_df), "09:15 ET", ny_open_conf),
        "ny_session": _row(ny_session_pred, _actual_direction_from_window(ny_df), "10:45 ET", ny_session_conf),
    }


def evaluate_bias_accuracy(
    df_today: pd.DataFrame,
    bias: Optional[BiasSummary],
    trading_date: Optional[dt.date] = None,
) -> AccuracySummary:
    """
    Compares the Daily Bias to the actual full-day direction.
    """
    if bias is None:
        df_slice = _slice_trading_day(df_today, trading_date)
        close_pos = _close_position_pct(df_slice)
        close_quality, close_band = _close_quality_label(close_pos)
        return AccuracySummary(
            actual_direction=_actual_direction(df_today),
            bias_correct=False,
            explanation="Bias data was unavailable, so accuracy could not be evaluated.",
            used_bias="n/a",
            us_open_bias_correct=False,
            us_open_bias_correct_30=None,
            us_open_bias_correct_60=None,
            close_position_pct=close_pos,
            close_quality=close_quality,
            close_quality_band=close_band,
            session_accuracy=None,
        )
    df_slice = _slice_trading_day(df_today, trading_date)
    actual = _actual_direction(df_slice)
    us_open_correct = actual == bias.us_open_bias
    us_open_actual_30 = _actual_direction_window(df_today, minutes=30, trading_date=trading_date)
    us_open_actual_60 = _actual_direction_window(df_today, minutes=60, trading_date=trading_date)
    us_open_bias_30 = getattr(bias, "us_open_bias_30", None)
    us_open_bias_60 = getattr(bias, "us_open_bias_60", None)
    if us_open_bias_30 in ("Bullish", "Bearish"):
        us_open_correct_30 = us_open_actual_30 == us_open_bias_30
    else:
        us_open_correct_30 = None
    if us_open_bias_60 in ("Bullish", "Bearish"):
        us_open_correct_60 = us_open_actual_60 == us_open_bias_60
    else:
        us_open_correct_60 = None
    use_us_open = bias.daily_bias == "Neutral" and bias.us_open_bias in ("Bullish", "Bearish")
    used_bias = "US Open (fallback)" if use_us_open else "Daily"
    correct = actual == (bias.us_open_bias if use_us_open else bias.daily_bias)
    close_pos = _close_position_pct(df_slice)
    close_quality, close_band = _close_quality_label(close_pos)
    session_accuracy = _session_accuracy_map(df_today, bias, trading_date or df_today["timestamp"].iloc[-1].date())
    close_pos_text = f"{close_pos:.1f}%" if close_pos is not None else "n/a"

    if use_us_open:
        if correct:
            explanation = (
                f"Daily Bias was Neutral, so accuracy fell back to US Open Bias ({bias.us_open_bias}). "
                f"The market closed {actual}, so the fallback bias was correct. "
                f"Close quality was {close_quality} ({close_pos_text} of session range)."
            )
        else:
            explanation = (
                f"Daily Bias was Neutral, so accuracy fell back to US Open Bias ({bias.us_open_bias}). "
                f"The market closed {actual}, so the fallback bias was not correct. "
                f"Close quality was {close_quality} ({close_pos_text} of session range)."
            )
    elif correct:
        explanation = (
            f"Daily Bias was {bias.daily_bias} and the market closed {actual}. "
            f"The bias correctly anticipated the full-day direction. Close quality was {close_quality} "
            f"({close_pos_text} of session range)."
        )
    else:
        explanation = (
            f"Daily Bias was {bias.daily_bias}, but the market closed {actual}. "
            "The bias did not match the actual outcome; review session structure, VWAP posture, "
            f"and news context for this day. Close quality was {close_quality} ({close_pos_text} of session range)."
        )

    return AccuracySummary(
        actual_direction=actual,
        bias_correct=correct,
        explanation=explanation,
        used_bias=used_bias,
        us_open_bias_correct=us_open_correct,
        us_open_bias_correct_30=us_open_correct_30,
        us_open_bias_correct_60=us_open_correct_60,
        close_position_pct=close_pos,
        close_quality=close_quality,
        close_quality_band=close_band,
        session_accuracy=session_accuracy,
    )