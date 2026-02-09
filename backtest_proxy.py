from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from storage.history_manager import DaySummary, load_all_summaries


@dataclass
class SignalResult:
    date: str
    symbol: str
    signal_type: str
    signal_dir: str
    actual_dir: str
    win: Optional[bool]
    daily_bias: str
    us_open_bias: str


def _pick_signal(summary: DaySummary) -> Tuple[Optional[str], str]:
    patterns = summary.patterns
    if patterns.orb_30:
        return "ORB_30", patterns.orb_30_bias or "Neutral"
    if patterns.failed_orb_30:
        return "FAILED_ORB_30", patterns.failed_orb_30_bias or "Neutral"
    if patterns.orb_60:
        return "ORB_60", patterns.orb_60_bias or "Neutral"
    if patterns.failed_orb_60:
        return "FAILED_ORB_60", patterns.failed_orb_60_bias or "Neutral"
    return None, "Neutral"


def _is_tradeable(direction: str) -> bool:
    return direction in ("Bullish", "Bearish")


def evaluate_proxy_strategy(summaries: List[DaySummary]) -> List[SignalResult]:
    results: List[SignalResult] = []
    for summary in summaries:
        signal_type, signal_dir = _pick_signal(summary)
        if signal_type is None or not _is_tradeable(signal_dir):
            continue

        actual_dir = summary.accuracy.actual_direction
        win: Optional[bool]
        if _is_tradeable(actual_dir):
            win = signal_dir == actual_dir
        else:
            win = None

        results.append(
            SignalResult(
                date=summary.date,
                symbol=summary.symbol,
                signal_type=signal_type,
                signal_dir=signal_dir,
                actual_dir=actual_dir,
                win=win,
                daily_bias=summary.bias.daily_bias,
                us_open_bias=summary.bias.us_open_bias,
            )
        )
    return results


def summarize(results: List[SignalResult]) -> Dict[str, object]:
    total = len(results)
    wins = sum(1 for r in results if r.win is True)
    losses = sum(1 for r in results if r.win is False)
    pushes = sum(1 for r in results if r.win is None)
    win_rate = wins / (wins + losses) if (wins + losses) else 0.0

    by_type: Dict[str, Dict[str, int]] = {}
    for r in results:
        bucket = by_type.setdefault(r.signal_type, {"wins": 0, "losses": 0, "pushes": 0, "total": 0})
        bucket["total"] += 1
        if r.win is True:
            bucket["wins"] += 1
        elif r.win is False:
            bucket["losses"] += 1
        else:
            bucket["pushes"] += 1

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "by_type": by_type,
    }


def _print_summary(stats: Dict[str, object]) -> None:
    print("Proxy ORB Strategy Backtest")
    print("-" * 32)
    print(f"Total signals: {stats['total']}")
    print(f"Wins: {stats['wins']}  Losses: {stats['losses']}  Pushes: {stats['pushes']}")
    print(f"Win rate (ex-push): {stats['win_rate']:.2%}")
    print()

    by_type = stats["by_type"]
    print("By signal type:")
    for signal_type in sorted(by_type.keys()):
        bucket = by_type[signal_type]
        wins = bucket["wins"]
        losses = bucket["losses"]
        pushes = bucket["pushes"]
        total = bucket["total"]
        wr = wins / (wins + losses) if (wins + losses) else 0.0
        print(
            f"  {signal_type:14s}  total={total:3d}  win%={wr:6.2%}  "
            f"wins={wins:3d} losses={losses:3d} pushes={pushes:3d}"
        )


def main() -> None:
    summaries = load_all_summaries()
    results = evaluate_proxy_strategy(summaries)
    stats = summarize(results)
    _print_summary(stats)


if __name__ == "__main__":
    main()
