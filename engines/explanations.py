def explain_indicator(name: str) -> str:
    """
    Short, human-readable explanations for key indicators and stats.
    """
    mapping = {
        "zscore": "Z-score shows how far today's price is from its recent average in standard deviations.",
        "rvol": "Relative Volume compares today's volume to typical volume for this lookback window.",
        "roc": "Rate of Change measures the speed of price movement over a chosen lookback.",
        "momentum": "Momentum shows how much price has moved over a recent window.",
        "sma": "Simple Moving Average smooths price over a fixed window to show trend direction.",
        "ema": "Exponential Moving Average weights recent prices more heavily to react faster.",
        "vwap": "VWAP is the volume-weighted average price, often used as a fair value reference.",
        "daily_vwap": "Daily anchored VWAP starts at the beginning of the trading day.",
        "weekly_vwap": "Weekly anchored VWAP starts at the beginning of the trading week.",
        "volatility": "Volatility measures how wide price is swinging compared to recent history.",
        "trend": "Trend classification identifies whether price is broadly rising, falling, or neutral.",
        "sessions": "Session stats summarize Asia, London, and US behavior for the day.",
        "patterns": "Patterns highlight structural behaviors like London Breakout, Whipsaw, and Trend Days.",
        "bias": "Bias is the model's directional view based on sessions, VWAP, volatility, and other factors.",
        "news": "News tone adjusts confidence but does not override the session-based structure.",
    }
    return mapping.get(name, "")