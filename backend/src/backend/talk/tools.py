import math

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


TOOLS = {"calculate": calculate}
