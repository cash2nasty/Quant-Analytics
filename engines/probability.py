from typing import Optional, Tuple


def bias_probabilities(
    bias: str,
    confidence: Optional[float],
    score: Optional[float] = None,
) -> Tuple[float, float]:
    conf = 0.0 if confidence is None else max(0.0, min(1.0, float(confidence)))

    # If score is present, map directly around 50/50 with bounded confidence.
    if score is not None:
        try:
            s = max(-1.0, min(1.0, float(score)))
            bull = 0.5 + 0.5 * s
            bull = max(0.01, min(0.99, bull))
            return bull, 1.0 - bull
        except Exception:
            pass

    text = str(bias or "Neutral")
    if text == "Bullish":
        bull = 0.5 + 0.5 * conf
    elif text == "Bearish":
        bull = 0.5 - 0.5 * conf
    else:
        bull = 0.5

    bull = max(0.01, min(0.99, bull))
    return bull, 1.0 - bull
