from storage.history_manager import SessionStats, PatternSummary
from typing import Dict


def detect_patterns(sessions: Dict[str, SessionStats]) -> PatternSummary:
    """
    Simple structural pattern detection using session stats.
    """
    asia = sessions.get("Asia")
    london = sessions.get("London")
    us = sessions.get("US")

    london_breakout = False
    whipsaw = False
    trend_day = False
    vol_expansion = False
    notes_parts = []

    if asia and london:
        # London Breakout: London range extends beyond Asia high/low
        if london.high > asia.high and london.low < asia.low:
            london_breakout = True
            notes_parts.append("London extended beyond Asia range (London Breakout).")

    if london and us:
        # Whipsaw: London up, US down (or vice versa)
        london_dir = london.close - london.open
        us_dir = us.close - us.open
        if london_dir * us_dir < 0:
            whipsaw = True
            notes_parts.append("London and US moved in opposite directions (Whipsaw).")

        # Trend Day: same direction across London and US with strong ranges
        if london_dir * us_dir > 0 and abs(london_dir) > london.range * 0.3 and abs(us_dir) > us.range * 0.3:
            trend_day = True
            notes_parts.append("London and US aligned directionally with strong ranges (Trend Day).")

        # Volatility Expansion: US range > 1.5x London range
        if us.range > 1.5 * london.range:
            vol_expansion = True
            notes_parts.append("US range expanded significantly vs London (Volatility Expansion).")

    if not notes_parts:
        notes_parts.append("No major structural patterns detected.")

    return PatternSummary(
        london_breakout=london_breakout,
        whipsaw=whipsaw,
        trend_day=trend_day,
        volatility_expansion=vol_expansion,
        notes=" ".join(notes_parts),
    )