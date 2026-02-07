import pandas as pd
from typing import Optional

from storage.history_manager import AccuracySummary, BiasSummary


def _actual_direction(df_today: pd.DataFrame) -> str:
    if df_today.empty:
        return "Neutral"
    open_ = df_today["open"].iloc[0]
    close = df_today["close"].iloc[-1]
    diff = close - open_
    if diff > 0:
        return "Bullish"
    if diff < 0:
        return "Bearish"
    return "Neutral"


def _actual_direction_window(df_today: pd.DataFrame, minutes: int) -> str:
    if df_today.empty or "timestamp" not in df_today.columns:
        return "Neutral"
    date = df_today["timestamp"].iloc[0].date()
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


def evaluate_bias_accuracy(df_today: pd.DataFrame, bias: Optional[BiasSummary]) -> AccuracySummary:
    """
    Compares the Daily Bias to the actual full-day direction.
    """
    if bias is None:
        return AccuracySummary(
            actual_direction=_actual_direction(df_today),
            bias_correct=False,
            explanation="Bias data was unavailable, so accuracy could not be evaluated.",
            used_bias="n/a",
            us_open_bias_correct=False,
            us_open_bias_correct_30=None,
            us_open_bias_correct_60=None,
        )
    actual = _actual_direction(df_today)
    us_open_correct = actual == bias.us_open_bias
    us_open_actual_30 = _actual_direction_window(df_today, minutes=30)
    us_open_actual_60 = _actual_direction_window(df_today, minutes=60)
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

    if use_us_open:
        if correct:
            explanation = (
                f"Daily Bias was Neutral, so accuracy fell back to US Open Bias ({bias.us_open_bias}). "
                f"The market closed {actual}, so the fallback bias was correct."
            )
        else:
            explanation = (
                f"Daily Bias was Neutral, so accuracy fell back to US Open Bias ({bias.us_open_bias}). "
                f"The market closed {actual}, so the fallback bias was not correct."
            )
    elif correct:
        explanation = (
            f"Daily Bias was {bias.daily_bias} and the market closed {actual}. "
            "The bias correctly anticipated the full-day direction."
        )
    else:
        explanation = (
            f"Daily Bias was {bias.daily_bias}, but the market closed {actual}. "
            "The bias did not match the actual outcome; review session structure, VWAP posture, "
            "and news context for this day."
        )

    return AccuracySummary(
        actual_direction=actual,
        bias_correct=correct,
        explanation=explanation,
        used_bias=used_bias,
        us_open_bias_correct=us_open_correct,
        us_open_bias_correct_30=us_open_correct_30,
        us_open_bias_correct_60=us_open_correct_60,
    )