import datetime as dt
from typing import Dict

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


def _is_dst(date: dt.date) -> bool:
    """Return True if the given date in US/Eastern is observing DST."""
    if ZoneInfo is None:
        # Fallback: assume DST between March and November (approx)
        return 3 <= date.month <= 11
    tz = ZoneInfo("America/New_York")
    dt_local = dt.datetime(date.year, date.month, date.day, 12, tzinfo=tz)
    return bool(dt_local.dst())


def get_session_windows_for_date(date: dt.date) -> Dict[str, Dict[str, dt.datetime]]:
    """Return session start/end datetimes in US/Eastern for the given trading date.

    Rules:
    - Asia session spans the previous calendar day into the trading date.
      Without DST: Asia = 19:00 (prev day) -> 01:00 (date). With DST: +1 hour.
    - London session is on the trading date: without DST 03:00-08:00, with DST +1 hour.
    - US session is fixed: 09:30-16:00 (ET) regardless of DST.

    All returned datetimes are naive (no tzinfo) but represent US/Eastern local times.
    """
    dst = _is_dst(date)
    # Asia: start on previous day at 19:00 (non-DST) or 20:00 (DST), end on date at 01:00/02:00
    asia_start_hour = 20 if dst else 19
    asia_end_hour = 2 if dst else 1

    # London: starts on date at 04:00-09:00 (DST) or 03:00-08:00 (non-DST)
    london_start_hour = 4 if dst else 3
    london_end_hour = 9 if dst else 8

    us_start = dt.datetime.combine(date, dt.time(9, 30))
    us_end = dt.datetime.combine(date, dt.time(16, 0))

    asia_start_date = date - dt.timedelta(days=1)
    asia_start = dt.datetime.combine(asia_start_date, dt.time(asia_start_hour, 0))
    asia_end = dt.datetime.combine(date, dt.time(asia_end_hour, 0))

    london_start = dt.datetime.combine(date, dt.time(london_start_hour, 0))
    london_end = dt.datetime.combine(date, dt.time(london_end_hour, 0))

    return {
        "Asia": {"start": asia_start, "end": asia_end},
        "London": {"start": london_start, "end": london_end},
        "US": {"start": us_start, "end": us_end},
    }