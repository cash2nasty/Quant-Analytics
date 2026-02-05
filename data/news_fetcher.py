from dataclasses import dataclass
from typing import Literal


NewsTone = Literal["risk_on", "risk_off", "neutral", "uncertain"]


@dataclass
class NewsSummary:
    source: str
    tone: NewsTone
    notes: str


@dataclass
class CombinedNewsSignal:
    tone: NewsTone
    confidence: float
    explanation: str


def fetch_forexfactory_summary() -> NewsSummary:
    """
    Conceptual ForexFactory-style summary.
    In a real implementation, you'd map calendar + impact to tone.
    """
    return NewsSummary(
        source="ForexFactory",
        tone="neutral",
        notes="No major high-impact events dominating today's tone.",
    )


def fetch_seekingalpha_summary() -> NewsSummary:
    """
    Conceptual SeekingAlpha-style macro sentiment.
    """
    return NewsSummary(
        source="SeekingAlpha",
        tone="neutral",
        notes="Macro commentary is mixed; no strong risk-on/off consensus.",
    )


def combine_news_signals() -> CombinedNewsSignal:
    """
    Combine ForexFactory + SeekingAlpha into a single tone.
    News is a confidence modifier, not a primary bias driver.
    """
    ff = fetch_forexfactory_summary()
    sa = fetch_seekingalpha_summary()

    tones = [ff.tone, sa.tone]
    if tones.count("risk_on") >= 2:
        tone = "risk_on"
        conf = 0.7
    elif tones.count("risk_off") >= 2:
        tone = "risk_off"
        conf = 0.7
    elif "uncertain" in tones:
        tone = "uncertain"
        conf = 0.5
    else:
        tone = "neutral"
        conf = 0.5

    explanation = (
        f"ForexFactory: {ff.tone} ({ff.notes}) | "
        f"SeekingAlpha: {sa.tone} ({sa.notes}). "
        "News tone adjusts bias confidence but does not override session structure."
    )

    return CombinedNewsSignal(tone=tone, confidence=conf, explanation=explanation)