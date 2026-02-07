from storage.history_manager import BiasSummary, TradeSuggestion


def build_trade_suggestion(bias: BiasSummary) -> TradeSuggestion:
    """
    Suggests whether to trade, trade small, or stand aside based on
    Daily Bias, US Open Bias, and confidence.
    """
    daily = bias.daily_bias
    us = getattr(bias, "us_open_bias_60", None) or bias.us_open_bias
    conf = bias.daily_confidence

    if daily == "Neutral" or us == "Neutral":
        action = "Don't Trade"
        rationale = (
            "Bias is Neutral or conflicting; structure does not provide a clear edge. "
            "Stand aside until a clearer directional picture emerges."
        )
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

    return TradeSuggestion(action=action, rationale=rationale)