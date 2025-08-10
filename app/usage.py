import json
from pathlib import Path
from datetime import date

USAGE_PATH = Path("app/data/.alpha_usage.json")
DAILY_LIMIT = 25  # Alpha Vantage free tier

def _load():
    try:
        return json.loads(USAGE_PATH.read_text())
    except Exception:
        return {}

def _save(obj):
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USAGE_PATH.write_text(json.dumps(obj))

def log_alpha_call():
    data = _load()
    today = date.today().isoformat()
    data[today] = int(data.get(today, 0)) + 1
    _save(data)
    return data[today]

def used_today():
    return int(_load().get(date.today().isoformat(), 0))

def left_today():
    return max(0, DAILY_LIMIT - used_today())
