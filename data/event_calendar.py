import datetime as dt
import json
from pathlib import Path
from typing import Dict, List
from urllib.error import URLError
from urllib.request import urlopen

_MANUAL_PATH = Path("data") / "event_calendar.json"
_CACHE_PATH = Path("data") / "event_calendar_cache.json"
_AUTO_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]
_KEYWORDS = ("CPI", "FOMC", "Non-Farm", "NFP")


def _normalize_events(items: List[dict]) -> Dict[str, List[str]]:
    events: Dict[str, List[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        date = item.get("date") or item.get("Date")
        name = item.get("event") or item.get("name") or item.get("Event")
        if not date or not name:
            continue
        if not any(key.lower() in str(name).lower() for key in _KEYWORDS):
            continue
        events.setdefault(str(date), []).append(str(name))
    return events


def _fetch_auto_events() -> Dict[str, List[str]]:
    items: List[dict] = []
    for url in _AUTO_URLS:
        try:
            with urlopen(url, timeout=10) as resp:
                payload = json.load(resp)
                if isinstance(payload, list):
                    items.extend(payload)
        except URLError:
            continue
        except Exception:
            continue
    return _normalize_events(items)


def _load_manual_events() -> Dict[str, List[str]]:
    if not _MANUAL_PATH.exists():
        return {}
    try:
        with open(_MANUAL_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        return {}

    items = data.get("events", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return {}
    return _normalize_events(items)


def _load_cache() -> Dict[str, List[str]]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        with open(_CACHE_PATH, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    fetched_on = data.get("fetched_on")
    if fetched_on != dt.date.today().isoformat():
        return {}
    events = data.get("events")
    if not isinstance(events, dict):
        return {}
    return {str(k): list(v) for k, v in events.items()}


def _save_cache(events: Dict[str, List[str]]) -> None:
    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump({"fetched_on": dt.date.today().isoformat(), "events": events}, f, indent=2)
    except Exception:
        return


def load_event_calendar() -> Dict[str, List[str]]:
    cached = _load_cache()
    if cached:
        auto_events = cached
    else:
        auto_events = _fetch_auto_events()
        _save_cache(auto_events)

    manual_events = _load_manual_events()
    merged: Dict[str, List[str]] = {}
    for date, names in auto_events.items():
        merged.setdefault(date, []).extend(names)
    for date, names in manual_events.items():
        merged.setdefault(date, []).extend(names)

    # Deduplicate while preserving order
    for date, names in merged.items():
        seen = set()
        merged[date] = [n for n in names if not (n in seen or seen.add(n))]

    return merged
