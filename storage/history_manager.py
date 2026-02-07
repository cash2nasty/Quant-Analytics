import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)


@dataclass
class SessionStats:
    name: str
    high: float
    low: float
    open: float
    close: float
    range: float
    volume: float


@dataclass
class PatternSummary:
    london_breakout: bool
    whipsaw: bool
    trend_day: bool
    volatility_expansion: bool
    notes: str
    asia_range_hold: Optional[bool] = None
    asia_range_sweep: Optional[bool] = None
    asia_range_sweep_bias: Optional[str] = None
    london_continuation: Optional[bool] = None
    london_continuation_bias: Optional[str] = None
    us_open_gap_fill: Optional[bool] = None
    us_open_gap_fill_bias: Optional[str] = None
    orb_30: Optional[bool] = None
    orb_30_bias: Optional[str] = None
    orb_60: Optional[bool] = None
    orb_60_bias: Optional[str] = None
    failed_orb_30: Optional[bool] = None
    failed_orb_30_bias: Optional[str] = None
    failed_orb_60: Optional[bool] = None
    failed_orb_60_bias: Optional[str] = None
    power_hour_trend: Optional[bool] = None
    power_hour_bias: Optional[str] = None
    vwap_reclaim_reject: Optional[bool] = None
    vwap_reclaim_reject_bias: Optional[str] = None


@dataclass
class BiasSummary:
    daily_bias: str
    daily_confidence: float
    us_open_bias: str
    us_open_confidence: float
    explanation: str
    vwap_comment: str
    news_comment: str
    us_open_bias_30: Optional[str] = None
    us_open_confidence_30: Optional[float] = None
    us_open_bias_60: Optional[str] = None
    us_open_confidence_60: Optional[float] = None
    amd_summary: Optional[str] = None


@dataclass
class TradeSuggestion:
    action: str
    rationale: str


@dataclass
class AccuracySummary:
    actual_direction: str
    bias_correct: bool
    explanation: str
    used_bias: str
    us_open_bias_correct: bool
    us_open_bias_correct_30: Optional[bool] = None
    us_open_bias_correct_60: Optional[bool] = None


@dataclass
class DaySummary:
    date: str
    symbol: str
    sessions: Dict[str, SessionStats]
    patterns: PatternSummary
    bias: BiasSummary
    trade_suggestion: TradeSuggestion
    accuracy: AccuracySummary
    day_high: Optional[float] = None
    day_low: Optional[float] = None


def save_day_summary(summary: DaySummary) -> None:
    path = HISTORY_DIR / f"{summary.date}_{summary.symbol.replace('/', '_')}.json"
    serializable = {
        "date": summary.date,
        "symbol": summary.symbol,
        "sessions": {k: asdict(v) for k, v in summary.sessions.items()},
        "patterns": asdict(summary.patterns),
        "bias": asdict(summary.bias),
        "trade_suggestion": asdict(summary.trade_suggestion),
        "accuracy": asdict(summary.accuracy),
        "day_high": summary.day_high,
        "day_low": summary.day_low,
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_all_summaries() -> List[DaySummary]:
    summaries: List[DaySummary] = []
    for path in HISTORY_DIR.glob("*.json"):
        with open(path, "r") as f:
            data = json.load(f)
        sessions = {k: SessionStats(**v) for k, v in data["sessions"].items()}
        p = data.get("patterns", {})
        patterns = PatternSummary(
            london_breakout=p.get("london_breakout", False),
            whipsaw=p.get("whipsaw", False),
            trend_day=p.get("trend_day", False),
            volatility_expansion=p.get("volatility_expansion", False),
            notes=p.get("notes", ""),
            asia_range_hold=p.get("asia_range_hold"),
            asia_range_sweep=p.get("asia_range_sweep"),
            asia_range_sweep_bias=p.get("asia_range_sweep_bias"),
            london_continuation=p.get("london_continuation"),
            london_continuation_bias=p.get("london_continuation_bias"),
            us_open_gap_fill=p.get("us_open_gap_fill"),
            us_open_gap_fill_bias=p.get("us_open_gap_fill_bias"),
            orb_30=p.get("orb_30"),
            orb_30_bias=p.get("orb_30_bias"),
            orb_60=p.get("orb_60"),
            orb_60_bias=p.get("orb_60_bias"),
            failed_orb_30=p.get("failed_orb_30"),
            failed_orb_30_bias=p.get("failed_orb_30_bias"),
            failed_orb_60=p.get("failed_orb_60"),
            failed_orb_60_bias=p.get("failed_orb_60_bias"),
            power_hour_trend=p.get("power_hour_trend"),
            power_hour_bias=p.get("power_hour_bias"),
            vwap_reclaim_reject=p.get("vwap_reclaim_reject"),
            vwap_reclaim_reject_bias=p.get("vwap_reclaim_reject_bias"),
        )
        bias = BiasSummary(**data["bias"])
        trade_suggestion = TradeSuggestion(**data["trade_suggestion"])
        acc_data = data.get("accuracy", {})
        accuracy = AccuracySummary(
            actual_direction=acc_data.get("actual_direction", "n/a"),
            bias_correct=acc_data.get("bias_correct", False),
            explanation=acc_data.get("explanation", ""),
            used_bias=acc_data.get("used_bias", "Daily"),
            us_open_bias_correct=acc_data.get("us_open_bias_correct", False),
            us_open_bias_correct_30=acc_data.get("us_open_bias_correct_30"),
            us_open_bias_correct_60=acc_data.get("us_open_bias_correct_60"),
        )
        summaries.append(
            DaySummary(
                date=data["date"],
                symbol=data["symbol"],
                sessions=sessions,
                patterns=patterns,
                bias=bias,
                trade_suggestion=trade_suggestion,
                accuracy=accuracy,
                day_high=data.get("day_high"),
                day_low=data.get("day_low"),
            )
        )
    summaries.sort(key=lambda x: x.date)
    return summaries