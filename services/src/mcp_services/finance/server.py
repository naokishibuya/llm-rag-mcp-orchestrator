"""FastMCP server exposing stock price lookup tool.

Uses yfinance for real stock quotes (free, no API key required).
https://github.com/ranaroussi/yfinance
"""

import logging
from fastmcp import FastMCP
import yfinance as yf


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("FinanceServer")


APP_NAME = "finance-server"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8030
DEFAULT_PATH = "/mcp"

INSTRUCTIONS = """Finance MCP Server for stock quote lookup.

Available tools:
- get_stock_price: Get current stock price for a ticker symbol (e.g., AAPL, GOOG, TSLA)

Data source: Yahoo Finance via yfinance (free, no API key required).
"""

mcp = FastMCP(APP_NAME, instructions=INSTRUCTIONS)


@mcp.tool
async def get_stock_price(symbol: str) -> dict:
    """Get current stock price, change, and market cap for a ticker symbol (e.g. AAPL, GOOG, TSLA)."""
    ticker = symbol.upper()

    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info

        price = info.last_price
        prev_close = info.previous_close

        if price is None or prev_close is None:
            logger.warning(f"No price data for {ticker}")
            return {"symbol": ticker, "error": f"No price data for {ticker}"}

        change = price - prev_close
        change_pct = (change / prev_close) * 100

        result = {
            "symbol": ticker,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_percent": f"{change_pct:+.2f}%",
            "currency": info.currency,
            "market_cap": info.market_cap,
        }
        logger.info(f"Quote for {ticker}: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch quote for {ticker}: {e}")
        return {"symbol": ticker, "error": f"Could not fetch data for {ticker}"}


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        path=DEFAULT_PATH,
    )
