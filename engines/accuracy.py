import pandas as pd

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


def evaluate_bias_accuracy(df_today: pd.DataFrame, bias: BiasSummary) -> AccuracySummary:
    """
    Compares the Daily Bias to the actual full-day direction.
    """
    actual = _actual_direction(df_today)
    us_open_correct = actual == bias.us_open_bias
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
    )