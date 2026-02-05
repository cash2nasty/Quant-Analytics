import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

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


@dataclass
class BiasSummary:
    daily_bias: str
    daily_confidence: float
    us_open_bias: str
    us_open_confidence: float
    explanation: str
    vwap_comment: str
    news_comment: str


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


@dataclass
class DaySummary:
    date: str
    symbol: str
    sessions: Dict[str, SessionStats]
    patterns: PatternSummary
    bias: BiasSummary
    trade_suggestion: TradeSuggestion
    accuracy: AccuracySummary


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
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_all_summaries() -> List[DaySummary]:
    summaries: List[DaySummary] = []
    for path in HISTORY_DIR.glob("*.json"):
        with open(path, "r") as f:
            data = json.load(f)
        sessions = {k: SessionStats(**v) for k, v in data["sessions"].items()}
        patterns = PatternSummary(**data["patterns"])
        bias = BiasSummary(**data["bias"])
        trade_suggestion = TradeSuggestion(**data["trade_suggestion"])
        acc_data = data.get("accuracy", {})
        accuracy = AccuracySummary(
            actual_direction=acc_data.get("actual_direction", "n/a"),
            bias_correct=acc_data.get("bias_correct", False),
            explanation=acc_data.get("explanation", ""),
            used_bias=acc_data.get("used_bias", "Daily"),
            us_open_bias_correct=acc_data.get("us_open_bias_correct", False),
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
            )
        )
    summaries.sort(key=lambda x: x.date)
    return summaries