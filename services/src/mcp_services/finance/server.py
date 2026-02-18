"""FastMCP server exposing financial analysis tools.

Uses yfinance for real stock data (free, no API key required).
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

INSTRUCTIONS = """Finance MCP Server for stock data and financial analysis.

Available tools:
- get_stock_price: Current stock price, change, and market cap
- get_income_statement: Revenue, net income, EPS (annual or quarterly)
- get_balance_sheet: Assets, liabilities, equity, debt (annual or quarterly)
- get_cash_flow: Operating, investing, financing cash flows (annual or quarterly)
- get_historical_prices: Historical OHLCV data (1mo to 5y)
- get_company_info: Company overview — sector, industry, description
- get_key_ratios: Valuation and profitability ratios (P/E, ROE, margins, etc.)
- compare_stocks: Side-by-side comparison of multiple stocks

Data source: Yahoo Finance via yfinance (free, no API key required).
"""

mcp = FastMCP(APP_NAME, instructions=INSTRUCTIONS)


def _safe_round(value, decimals=2):
    """Round a value if it's a number, otherwise return None."""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def _safe_get(info: dict, key: str, default=None):
    """Safely get a value from a dict, returning default for None/NaN."""
    value = info.get(key, default)
    if value is None:
        return default
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


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


@mcp.tool
async def get_income_statement(symbol: str, period: str = "annual") -> dict:
    """Get income statement data (revenue, net income, EPS) for a ticker symbol.

    Args:
        symbol: Ticker symbol (e.g. AAPL, GOOG, TSLA)
        period: "annual" or "quarterly"
    """
    ticker = symbol.upper()

    try:
        stock = yf.Ticker(ticker)
        df = stock.quarterly_income_stmt if period == "quarterly" else stock.income_stmt

        if df is None or df.empty:
            return {"symbol": ticker, "error": f"No income statement data for {ticker}"}

        # Limit to most recent 4 periods
        df = df.iloc[:, :4]

        results = []
        for col in df.columns:
            results.append({
                "date": col.strftime("%Y-%m-%d"),
                "total_revenue": _safe_round(df.at["Total Revenue", col]) if "Total Revenue" in df.index else None,
                "net_income": _safe_round(df.at["Net Income", col]) if "Net Income" in df.index else None,
                "gross_profit": _safe_round(df.at["Gross Profit", col]) if "Gross Profit" in df.index else None,
                "ebitda": _safe_round(df.at["EBITDA", col]) if "EBITDA" in df.index else None,
                "basic_eps": _safe_round(df.at["Basic EPS", col]) if "Basic EPS" in df.index else None,
            })

        logger.info(f"Income statement for {ticker}: {len(results)} periods")
        return {"symbol": ticker, "period": period, "statements": results}

    except Exception as e:
        logger.error(f"Failed to fetch income statement for {ticker}: {e}")
        return {"symbol": ticker, "error": f"Could not fetch income statement for {ticker}"}


@mcp.tool
async def get_balance_sheet(symbol: str, period: str = "annual") -> dict:
    """Get balance sheet data (assets, liabilities, equity, debt) for a ticker symbol.

    Args:
        symbol: Ticker symbol (e.g. AAPL, GOOG, TSLA)
        period: "annual" or "quarterly"
    """
    ticker = symbol.upper()

    try:
        stock = yf.Ticker(ticker)
        df = stock.quarterly_balance_sheet if period == "quarterly" else stock.balance_sheet

        if df is None or df.empty:
            return {"symbol": ticker, "error": f"No balance sheet data for {ticker}"}

        df = df.iloc[:, :4]

        results = []
        for col in df.columns:
            results.append({
                "date": col.strftime("%Y-%m-%d"),
                "total_assets": _safe_round(df.at["Total Assets", col]) if "Total Assets" in df.index else None,
                "total_liabilities": _safe_round(df.at["Total Liabilities Net Minority Interest", col]) if "Total Liabilities Net Minority Interest" in df.index else None,
                "stockholders_equity": _safe_round(df.at["Stockholders Equity", col]) if "Stockholders Equity" in df.index else None,
                "total_debt": _safe_round(df.at["Total Debt", col]) if "Total Debt" in df.index else None,
                "cash_and_equivalents": _safe_round(df.at["Cash And Cash Equivalents", col]) if "Cash And Cash Equivalents" in df.index else None,
            })

        logger.info(f"Balance sheet for {ticker}: {len(results)} periods")
        return {"symbol": ticker, "period": period, "statements": results}

    except Exception as e:
        logger.error(f"Failed to fetch balance sheet for {ticker}: {e}")
        return {"symbol": ticker, "error": f"Could not fetch balance sheet for {ticker}"}


@mcp.tool
async def get_cash_flow(symbol: str, period: str = "annual") -> dict:
    """Get cash flow statement (operating, investing, financing) for a ticker symbol.

    Args:
        symbol: Ticker symbol (e.g. AAPL, GOOG, TSLA)
        period: "annual" or "quarterly"
    """
    ticker = symbol.upper()

    try:
        stock = yf.Ticker(ticker)
        df = stock.quarterly_cash_flow if period == "quarterly" else stock.cash_flow

        if df is None or df.empty:
            return {"symbol": ticker, "error": f"No cash flow data for {ticker}"}

        df = df.iloc[:, :4]

        results = []
        for col in df.columns:
            results.append({
                "date": col.strftime("%Y-%m-%d"),
                "operating_cash_flow": _safe_round(df.at["Operating Cash Flow", col]) if "Operating Cash Flow" in df.index else None,
                "investing_cash_flow": _safe_round(df.at["Investing Cash Flow", col]) if "Investing Cash Flow" in df.index else None,
                "financing_cash_flow": _safe_round(df.at["Financing Cash Flow", col]) if "Financing Cash Flow" in df.index else None,
                "free_cash_flow": _safe_round(df.at["Free Cash Flow", col]) if "Free Cash Flow" in df.index else None,
            })

        logger.info(f"Cash flow for {ticker}: {len(results)} periods")
        return {"symbol": ticker, "period": period, "statements": results}

    except Exception as e:
        logger.error(f"Failed to fetch cash flow for {ticker}: {e}")
        return {"symbol": ticker, "error": f"Could not fetch cash flow for {ticker}"}


@mcp.tool
async def get_historical_prices(symbol: str, period: str = "6mo") -> dict:
    """Get historical price data (OHLCV) for a ticker symbol.

    Args:
        symbol: Ticker symbol (e.g. AAPL, GOOG, TSLA)
        period: "1mo", "3mo", "6mo", "1y", "2y", or "5y"
    """
    ticker = symbol.upper()

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df is None or df.empty:
            return {"symbol": ticker, "error": f"No historical data for {ticker}"}

        # Downsample to ~30 data points max
        max_points = 30
        if len(df) > max_points:
            step = len(df) // max_points
            df = df.iloc[::step]

        results = []
        for idx, row in df.iterrows():
            results.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": _safe_round(row.get("Open")),
                "high": _safe_round(row.get("High")),
                "low": _safe_round(row.get("Low")),
                "close": _safe_round(row.get("Close")),
                "volume": int(row.get("Volume", 0)),
            })

        logger.info(f"Historical prices for {ticker}: {len(results)} data points")
        return {"symbol": ticker, "period": period, "prices": results}

    except Exception as e:
        logger.error(f"Failed to fetch historical prices for {ticker}: {e}")
        return {"symbol": ticker, "error": f"Could not fetch historical data for {ticker}"}


@mcp.tool
async def get_company_info(symbol: str) -> dict:
    """Get company overview — name, sector, industry, description, and more.

    Args:
        symbol: Ticker symbol (e.g. AAPL, GOOG, TSLA)
    """
    ticker = symbol.upper()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("trailingPegRatio") is None and info.get("shortName") is None:
            return {"symbol": ticker, "error": f"No company info for {ticker}"}

        result = {
            "symbol": ticker,
            "name": _safe_get(info, "longName") or _safe_get(info, "shortName"),
            "sector": _safe_get(info, "sector"),
            "industry": _safe_get(info, "industry"),
            "description": _safe_get(info, "longBusinessSummary"),
            "employees": _safe_get(info, "fullTimeEmployees"),
            "website": _safe_get(info, "website"),
            "country": _safe_get(info, "country"),
            "exchange": _safe_get(info, "exchange"),
        }

        logger.info(f"Company info for {ticker}: {result.get('name')}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch company info for {ticker}: {e}")
        return {"symbol": ticker, "error": f"Could not fetch company info for {ticker}"}


@mcp.tool
async def get_key_ratios(symbol: str) -> dict:
    """Get key valuation and profitability ratios (P/E, ROE, margins, etc.) for a ticker symbol.

    Args:
        symbol: Ticker symbol (e.g. AAPL, GOOG, TSLA)
    """
    ticker = symbol.upper()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return {"symbol": ticker, "error": f"No ratio data for {ticker}"}

        result = {
            "symbol": ticker,
            "pe_ratio": _safe_round(_safe_get(info, "trailingPE")),
            "forward_pe": _safe_round(_safe_get(info, "forwardPE")),
            "pb_ratio": _safe_round(_safe_get(info, "priceToBook")),
            "ps_ratio": _safe_round(_safe_get(info, "priceToSalesTrailing12Months")),
            "peg_ratio": _safe_round(_safe_get(info, "pegRatio")),
            "roe": _safe_round(_safe_get(info, "returnOnEquity")),
            "roa": _safe_round(_safe_get(info, "returnOnAssets")),
            "profit_margin": _safe_round(_safe_get(info, "profitMargins")),
            "operating_margin": _safe_round(_safe_get(info, "operatingMargins")),
            "debt_to_equity": _safe_round(_safe_get(info, "debtToEquity")),
            "current_ratio": _safe_round(_safe_get(info, "currentRatio")),
            "dividend_yield": _safe_round(_safe_get(info, "dividendYield"), 4),
            "beta": _safe_round(_safe_get(info, "beta")),
        }

        logger.info(f"Key ratios for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch key ratios for {ticker}: {e}")
        return {"symbol": ticker, "error": f"Could not fetch ratios for {ticker}"}


@mcp.tool
async def compare_stocks(symbols: str) -> dict:
    """Compare multiple stocks side by side — price, market cap, and key ratios.

    Args:
        symbols: Comma-separated ticker symbols (e.g. "AAPL,MSFT,GOOG")
    """
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    if not tickers:
        return {"error": "No valid symbols provided"}

    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            fast = stock.fast_info
            info = stock.info

            price = fast.last_price
            prev_close = fast.previous_close
            change_pct = ((price - prev_close) / prev_close * 100) if price and prev_close else None

            results.append({
                "symbol": ticker,
                "price": _safe_round(price),
                "change_percent": f"{change_pct:+.2f}%" if change_pct is not None else None,
                "market_cap": fast.market_cap,
                "pe_ratio": _safe_round(_safe_get(info, "trailingPE")),
                "forward_pe": _safe_round(_safe_get(info, "forwardPE")),
                "pb_ratio": _safe_round(_safe_get(info, "priceToBook")),
                "profit_margin": _safe_round(_safe_get(info, "profitMargins")),
                "roe": _safe_round(_safe_get(info, "returnOnEquity")),
                "debt_to_equity": _safe_round(_safe_get(info, "debtToEquity")),
                "dividend_yield": _safe_round(_safe_get(info, "dividendYield"), 4),
                "beta": _safe_round(_safe_get(info, "beta")),
            })
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            results.append({"symbol": ticker, "error": f"Could not fetch data for {ticker}"})

    logger.info(f"Compared {len(results)} stocks: {[r['symbol'] for r in results]}")
    return {"comparison": results}


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        path=DEFAULT_PATH,
    )
