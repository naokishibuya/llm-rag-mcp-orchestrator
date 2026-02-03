import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from services.toy_finance import client as finance_client
from ..safety import extract_symbol


async def fetch_quote(symbol: str) -> dict | None:
    try:
        return await finance_client.get_stock_price(symbol)
    except Exception:
        return None


async def get_finance_quote(text: str) -> str:
    symbol = extract_symbol(text)
    if not symbol:
        return "I couldn't spot a ticker symbol in that request."

    payload = await fetch_quote(symbol)
    if not payload:
        return "Sorry, I couldn't reach the finance quote service right now."

    price = payload.get("price")
    if price is None:
        return f"I couldn't find a price for {symbol}."

    return f"Mock quote for {symbol}: ${float(price):.2f}."
