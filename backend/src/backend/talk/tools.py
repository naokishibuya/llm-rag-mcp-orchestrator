import math
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from ..llm import tool


@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.

    Supports standard math functions (sqrt, sin, cos, log, etc.),
    abs, round, min, and max.

    Args:
        expression: A mathematical expression string, e.g. "sqrt(144)" or "0.15 * 230".
    """
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max})
    return float(eval(expression, {"__builtins__": {}}, allowed))


@tool
def get_current_time(tz: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        tz: IANA timezone name, e.g. "Asia/Tokyo", "America/New_York". Defaults to "UTC".
    """
    try:
        zone = ZoneInfo(tz)
    except KeyError:
        zone = timezone.utc
    now = datetime.now(zone)
    return now.strftime(f"%Y-%m-%d %H:%M:%S %Z ({tz})")


TOOLS = {"calculate": calculate, "get_current_time": get_current_time}
