import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from data.data_fetcher import fetch_intraday_ohlcv
from data.session_reference import get_session_windows_for_date
from engines.zones import build_htf_zones


POINT_VALUE = 20.0
DAILY_PROFIT_CAP = 2000.0
DAILY_LOSS_CAP = -1000.0


@dataclass
class TradeResult:
	date: dt.date
	po3_type: str
	variant: str
	direction: str
	entry_time: dt.datetime
	entry_price: float
	exit_time: dt.datetime
	exit_price: float
	r_multiple: float
	win: bool


def _trading_date_for_timestamp(ts: dt.datetime) -> dt.date:
	date = ts.date()
	if ts >= dt.datetime.combine(date, dt.time(18, 0)):
		date = date + dt.timedelta(days=1)
	while date.weekday() >= 5:
		date += dt.timedelta(days=1)
	return date


def _trading_day_window(trading_date: dt.date) -> Tuple[dt.datetime, dt.datetime]:
	start = dt.datetime.combine(trading_date - dt.timedelta(days=1), dt.time(18, 0))
	end = dt.datetime.combine(trading_date, dt.time(17, 0))
	return start, end


def _zscore(series: pd.Series, length: int = 20) -> pd.Series:
	if series.empty:
		return pd.Series(dtype=float)
	mean = series.rolling(length).mean()
	std = series.rolling(length).std()
	return (series - mean) / std.replace(0, pd.NA)


def _slice(df: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
	return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()


def _structure_signal(df: pd.DataFrame, window: int = 3) -> str:
	if df is None or df.empty:
		return "None"
	if len(df) < (window * 2 + 2):
		return "None"
	data = df.copy().reset_index(drop=True)
	last_swing_high = None
	last_swing_low = None
	last_dir = "None"
	for i in range(window, len(data) - window):
		high = data["high"].iloc[i]
		low = data["low"].iloc[i]
		if high == max(data["high"].iloc[i - window : i + window + 1]):
			last_swing_high = float(high)
		if low == min(data["low"].iloc[i - window : i + window + 1]):
			last_swing_low = float(low)
		if i == 0:
			continue
		close = float(data["close"].iloc[i])
		prev_close = float(data["close"].iloc[i - 1])
		if last_swing_high is not None:
			if prev_close <= last_swing_high and close > last_swing_high:
				last_dir = "Bullish"
		if last_swing_low is not None:
			if prev_close >= last_swing_low and close < last_swing_low:
				last_dir = "Bearish"
	return last_dir


def _find_manipulation_direction(
	df: pd.DataFrame,
	acc_high: float,
	acc_low: float,
) -> Optional[str]:
	for _, row in df.iterrows():
		if row["high"] > acc_high and row["close"] <= acc_high:
			return "Bearish"
		if row["low"] < acc_low and row["close"] >= acc_low:
			return "Bullish"
	return None


def _simulate_trade(
	df: pd.DataFrame,
	direction: str,
	entry_price: float,
	stop: float,
	final_target: float,
	closer_target: float,
	initial_risk: float,
) -> Tuple[dt.datetime, float, float, bool]:
	if df.empty:
		return df.index[0], entry_price, 0.0, False
	breakeven_set = False
	for _, row in df.iterrows():
		if direction == "Bullish":
			stop_hit = row["low"] <= stop
			target_hit = row["high"] >= final_target
			if stop_hit and target_hit:
				exit_price = stop
				r_mult = -1.0
				return row["timestamp"], exit_price, r_mult, False
			if stop_hit:
				exit_price = stop
				r_mult = -1.0
				return row["timestamp"], exit_price, r_mult, False
			if target_hit:
				exit_price = final_target
				r_mult = (final_target - entry_price) / max(initial_risk, 1e-6)
				return row["timestamp"], exit_price, r_mult, True

			if not breakeven_set and row["close"] >= closer_target:
				stop = entry_price
				breakeven_set = True
			elif breakeven_set and row["close"] < closer_target:
				exit_price = float(row["close"])
				r_mult = (exit_price - entry_price) / max(initial_risk, 1e-6)
				return row["timestamp"], exit_price, r_mult, exit_price > entry_price
		else:
			stop_hit = row["high"] >= stop
			target_hit = row["low"] <= final_target
			if stop_hit and target_hit:
				exit_price = stop
				r_mult = -1.0
				return row["timestamp"], exit_price, r_mult, False
			if stop_hit:
				exit_price = stop
				r_mult = -1.0
				return row["timestamp"], exit_price, r_mult, False
			if target_hit:
				exit_price = final_target
				r_mult = (entry_price - final_target) / max(initial_risk, 1e-6)
				return row["timestamp"], exit_price, r_mult, True

			if not breakeven_set and row["close"] <= closer_target:
				stop = entry_price
				breakeven_set = True
			elif breakeven_set and row["close"] > closer_target:
				exit_price = float(row["close"])
				r_mult = (entry_price - exit_price) / max(initial_risk, 1e-6)
				return row["timestamp"], exit_price, r_mult, exit_price < entry_price

	last = df.iloc[-1]
	exit_price = float(last["close"])
	if direction == "Bullish":
		r_mult = (exit_price - entry_price) / max(initial_risk, 1e-6)
	else:
		r_mult = (entry_price - exit_price) / max(initial_risk, 1e-6)
	return last["timestamp"], exit_price, r_mult, r_mult > 0


def _breakout_entry(
	df: pd.DataFrame,
	direction: str,
	acc_high: float,
	acc_low: float,
) -> Optional[Tuple[dt.datetime, float]]:
	for _, row in df.iterrows():
		if direction == "Bullish" and row["close"] > acc_high and row["z"] >= 0.5:
			return row["timestamp"], float(row["close"])
		if direction == "Bearish" and row["close"] < acc_low and row["z"] <= -0.5:
			return row["timestamp"], float(row["close"])
	return None


def _continuation_entry(
	df: pd.DataFrame,
	direction: str,
	acc_high: float,
	acc_low: float,
) -> Optional[Tuple[dt.datetime, float]]:
	breakout_seen = False
	for _, row in df.iterrows():
		if not breakout_seen:
			if direction == "Bullish" and row["close"] > acc_high:
				breakout_seen = True
			elif direction == "Bearish" and row["close"] < acc_low:
				breakout_seen = True
			continue
		if direction == "Bullish" and row["low"] <= acc_high and abs(row["z"]) <= 0.5:
			return row["timestamp"], float(row["close"])
		if direction == "Bearish" and row["high"] >= acc_low and abs(row["z"]) <= 0.5:
			return row["timestamp"], float(row["close"])
	return None


def _run_variants(
	df_dist: pd.DataFrame,
	df_day: pd.DataFrame,
	direction: str,
	acc_high: float,
	acc_low: float,
	manip_high: float,
	manip_low: float,
	structure_required: bool,
	structure_align: bool,
) -> Dict[str, Optional[TradeResult]]:
	entries = {
		"breakout": _breakout_entry(df_dist, direction, acc_high, acc_low),
		"continuation": _continuation_entry(df_dist, direction, acc_high, acc_low),
	}
	if df_day is not None and not df_day.empty:
		zones = build_htf_zones(df_day)
	else:
		zones = []
	results: Dict[str, Optional[TradeResult]] = {}
	for variant, entry in entries.items():
		if entry is None:
			results[variant] = None
			continue
		entry_time, entry_price = entry
		context = df_day[df_day["timestamp"] <= entry_time]
		if context.empty:
			results[variant] = None
			continue
		if structure_required:
			structure_dir = _structure_signal(context)
			if structure_dir not in ("Bullish", "Bearish"):
				results[variant] = None
				continue
			if structure_align and structure_dir != direction:
				results[variant] = None
				continue

		def stdv_targets_swing() -> List[float]:
			if direction == "Bullish":
				base = abs(acc_high - manip_low)
				anchor = acc_high
			else:
				base = abs(manip_high - acc_low)
				anchor = acc_low
			base = max(base, 1e-6)
			levels = [2.0, 2.5, 3.0, 3.5]
			candidates: List[float] = []
			for level in levels:
				if direction == "Bullish":
					candidates.append(anchor + level * base)
				else:
					candidates.append(anchor - level * base)
			return candidates

		stdv_targets = stdv_targets_swing()

		rth_start = dt.datetime.combine(entry_time.date(), dt.time(9, 30))
		rth_context = context[context["timestamp"] >= rth_start]
		if rth_context.empty:
			structure_target = None
		elif direction == "Bullish":
			structure_target = float(rth_context["high"].max())
		else:
			structure_target = float(rth_context["low"].min())

		candidates: List[float] = [t for t in stdv_targets]
		if structure_target is not None:
			candidates.append(structure_target)

		if direction == "Bullish":
			candidates = [t for t in candidates if t > entry_price]
			if not candidates:
				results[variant] = None
				continue
			closer_target = min(candidates)
			final_target = max(candidates)
			stop = acc_low
			initial_risk = max(entry_price - stop, 1e-6)
		else:
			candidates = [t for t in candidates if t < entry_price]
			if not candidates:
				results[variant] = None
				continue
			closer_target = max(candidates)
			final_target = min(candidates)
			stop = acc_high
			initial_risk = max(stop - entry_price, 1e-6)
		df_after = df_dist[df_dist["timestamp"] >= entry_time]
		exit_time, exit_price, r_mult, win = _simulate_trade(
			df_after,
			direction,
			entry_price,
			stop,
			final_target,
			closer_target,
			initial_risk,
		)
		results[variant] = TradeResult(
			date=entry_time.date(),
			po3_type="",
			variant=variant,
			direction=direction,
			entry_time=entry_time,
			entry_price=entry_price,
			exit_time=exit_time,
			exit_price=exit_price,
			r_multiple=r_mult,
			win=win,
		)
	return results


def _evaluate_po3_sessions(
	df_day: pd.DataFrame,
	trading_date: dt.date,
	structure_required: bool,
	structure_align: bool,
) -> Dict[str, Optional[TradeResult]]:
	windows = get_session_windows_for_date(trading_date)
	asia = windows.get("Asia")
	london = windows.get("London")
	us = windows.get("US")
	if not asia or not london or not us:
		return {"breakout": None, "continuation": None}

	df_acc = _slice(df_day, asia["start"], asia["end"])
	df_manip = _slice(df_day, london["start"], london["end"])
	df_dist = _slice(df_day, us["start"], us["end"])

	if df_acc.empty or df_manip.empty or df_dist.empty:
		return {"breakout": None, "continuation": None}

	acc_high = float(df_acc["high"].max())
	acc_low = float(df_acc["low"].min())
	manip_high = float(df_manip["high"].max())
	manip_low = float(df_manip["low"].min())
	direction = _find_manipulation_direction(df_manip, acc_high, acc_low)
	if direction is None:
		return {"breakout": None, "continuation": None}

	return _run_variants(
		df_dist,
		df_day,
		direction,
		acc_high,
		acc_low,
		manip_high,
		manip_low,
		structure_required,
		structure_align,
	)


def _summarize(trades: Iterable[TradeResult]) -> Dict[str, float]:
	trades = list(trades)
	if not trades:
		return {
			"count": 0,
			"win_rate": 0.0,
			"avg_r": 0.0,
			"total_r": 0.0,
			"avg_dollars": 0.0,
			"total_dollars": 0.0,
			"profit_factor": 0.0,
		}
	wins = [t for t in trades if t.win]
	losses = [t for t in trades if not t.win]
	total_r = sum(t.r_multiple for t in trades)
	avg_r = total_r / max(len(trades), 1)
	total_dollars = sum(_trade_pnl_dollars(t) for t in trades)
	avg_dollars = total_dollars / max(len(trades), 1)
	win_rate = len(wins) / max(len(trades), 1)
	win_sum = sum(t.r_multiple for t in wins)
	loss_sum = sum(abs(t.r_multiple) for t in losses)
	profit_factor = win_sum / loss_sum if loss_sum > 0 else float("inf")
	return {
		"count": len(trades),
		"win_rate": win_rate,
		"avg_r": avg_r,
		"total_r": total_r,
		"avg_dollars": avg_dollars,
		"total_dollars": total_dollars,
		"profit_factor": profit_factor,
	}


def backtest_po3_stdv(
	symbol: str,
	start_date: dt.date,
	end_date: dt.date,
) -> List[TradeResult]:
	df, resolved = fetch_intraday_ohlcv(symbol, lookback_days=(start_date, end_date))
	if df.empty:
		print("No intraday data returned. Check symbol or date range.")
		return []
	if resolved:
		print(f"Using symbol: {resolved}")

	return _backtest_from_df(
		df,
		structure_required=False,
		structure_align=False,
	)


def _load_csv_data(path: str) -> pd.DataFrame:
	try:
		df = pd.read_csv(path)
	except Exception as exc:
		print(f"Failed to read CSV: {exc}")
		return pd.DataFrame()

	if df.empty:
		return df

	columns = {c.lower(): c for c in df.columns}
	if "time" in columns:
		df.rename(columns={columns["time"]: "timestamp"}, inplace=True)
	if "volume" in columns:
		df.rename(columns={columns["volume"]: "volume"}, inplace=True)
	for col in ("open", "high", "low", "close"):
		if col in columns:
			df.rename(columns={columns[col]: col}, inplace=True)

	required = {"timestamp", "open", "high", "low", "close"}
	if not required.issubset(set(df.columns)):
		print("CSV is missing required columns: timestamp, open, high, low, close.")
		return pd.DataFrame()

	try:
		ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
		ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
		df["timestamp"] = ts
	except Exception as exc:
		print(f"Failed to parse timestamps: {exc}")
		return pd.DataFrame()

	for col in ("open", "high", "low", "close", "volume"):
		if col not in df.columns:
			if col == "volume":
				df[col] = 0
			else:
				df[col] = pd.NA

	return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna(subset=["timestamp"])


def _backtest_from_df(
	df: pd.DataFrame,
	structure_required: bool,
	structure_align: bool,
) -> List[TradeResult]:
	if df.empty:
		return []

	df = df.sort_values("timestamp").reset_index(drop=True)
	df["trading_date"] = df["timestamp"].apply(_trading_date_for_timestamp)
	df["z"] = _zscore(df["close"], length=20)

	results: List[TradeResult] = []
	for trading_date, df_day in df.groupby("trading_date"):
		day_start, day_end = _trading_day_window(trading_date)
		df_day = _slice(df_day, day_start, day_end)
		if df_day.empty:
			continue

		session_trades = _evaluate_po3_sessions(
			df_day,
			trading_date,
			structure_required,
			structure_align,
		)
		for variant, trade in session_trades.items():
			if trade is None:
				continue
			trade.po3_type = "sessions"
			trade.variant = variant
			results.append(trade)

	return _apply_daily_caps(
		results,
		profit_cap=DAILY_PROFIT_CAP,
		loss_cap=DAILY_LOSS_CAP,
		per_variant=True,
	)


def _trade_pnl_dollars(trade: TradeResult) -> float:
	if trade.direction == "Bullish":
		points = trade.exit_price - trade.entry_price
	else:
		points = trade.entry_price - trade.exit_price
	return float(points * POINT_VALUE)


def _apply_daily_caps(
	trades: List[TradeResult],
	profit_cap: float,
	loss_cap: float,
	per_variant: bool,
) -> List[TradeResult]:
	if not trades:
		return []

	filtered: List[TradeResult] = []
	if per_variant:
		key_fn = lambda t: (t.date, t.po3_type, t.variant)
	else:
		key_fn = lambda t: (t.date,)

	groups: Dict[Tuple[object, ...], List[TradeResult]] = {}
	for trade in trades:
		groups.setdefault(key_fn(trade), []).append(trade)

	for _, group in groups.items():
		group_sorted = sorted(group, key=lambda t: t.entry_time)
		running_pnl = 0.0
		for trade in group_sorted:
			if running_pnl >= profit_cap or running_pnl <= loss_cap:
				break
			trade_pnl = _trade_pnl_dollars(trade)
			running_pnl += trade_pnl
			filtered.append(trade)

	return filtered


def _print_summary(trades: List[TradeResult]) -> None:
	if not trades:
		print("No trades generated.")
		return
	buckets: Dict[Tuple[str, str], List[TradeResult]] = {}
	for t in trades:
		buckets.setdefault((t.po3_type, t.variant), []).append(t)

	print("PO3 + STDV Backtest")
	print("-" * 36)
	for key in sorted(buckets.keys()):
		po3_type, variant = key
		stats = _summarize(buckets[key])
		print(
			f"{po3_type:7s} | {variant:13s} | trades={stats['count']:3d} "
			f"win%={stats['win_rate']:.2%} avgR={stats['avg_r']:.2f} "
			f"totalR={stats['total_r']:.2f} avg$={stats['avg_dollars']:.0f} "
			f"total$={stats['total_dollars']:.0f} PF={stats['profit_factor']:.2f}"
		)


def summarize_overall(trades: List[TradeResult]) -> Dict[str, float]:
	if not trades:
		return {
			"total_pnl": 0.0,
			"total_profit": 0.0,
			"total_loss": 0.0,
			"days_traded": 0,
			"avg_trades_per_day": 0.0,
			"win_rate": 0.0,
		}
	pnls = [_trade_pnl_dollars(t) for t in trades]
	total_pnl = sum(pnls)
	total_profit = sum(p for p in pnls if p > 0)
	total_loss = sum(p for p in pnls if p < 0)
	by_day: Dict[dt.date, int] = {}
	for t in trades:
		by_day[t.date] = by_day.get(t.date, 0) + 1
	days_traded = len(by_day)
	avg_trades = len(trades) / max(days_traded, 1)
	win_rate = sum(1 for t in trades if t.win) / max(len(trades), 1)
	return {
		"total_pnl": total_pnl,
		"total_profit": total_profit,
		"total_loss": total_loss,
		"days_traded": days_traded,
		"avg_trades_per_day": avg_trades,
		"win_rate": win_rate,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Backtest PO3 + STDV strategy on NQ futures.")
	parser.add_argument("--symbol", default="NQH26", help="Symbol (will map to NQ=F if needed).")
	parser.add_argument("--start", required=False, help="Start date (YYYY-MM-DD).")
	parser.add_argument("--end", required=False, help="End date (YYYY-MM-DD).")
	parser.add_argument("--csv", help="Path to a 1m CSV export from TradingView.")
	parser.add_argument(
		"--structure",
		choices=["off", "present", "aligned"],
		default="off",
		help="Require 5m structure confirmation.",
	)
	args = parser.parse_args()
	structure_required = args.structure in ("present", "aligned")
	structure_align = args.structure == "aligned"

	if args.csv:
		df = _load_csv_data(args.csv)
		trades = _backtest_from_df(
			df,
			structure_required=structure_required,
			structure_align=structure_align,
		)
	else:
		if not args.start or not args.end:
			raise SystemExit("--start and --end are required when not using --csv")
		start_date = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
		end_date = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
		df, _ = fetch_intraday_ohlcv(args.symbol, lookback_days=(start_date, end_date))
		trades = _backtest_from_df(
			df,
			structure_required=structure_required,
			structure_align=structure_align,
		)
	_print_summary(trades)
	overall = summarize_overall(trades)
	print(
		"Overall: "
		f"total$={overall['total_pnl']:.0f} profit$={overall['total_profit']:.0f} "
		f"loss$={overall['total_loss']:.0f} days={overall['days_traded']} "
		f"avg_trades/day={overall['avg_trades_per_day']:.2f} win%={overall['win_rate']:.2%}"
	)


if __name__ == "__main__":
	main()
