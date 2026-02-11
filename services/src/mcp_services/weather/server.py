"""FastMCP server exposing weather lookup tool.

Uses Open-Meteo API for weather data (free, no API key required).
https://open-meteo.com/en/docs
"""

import httpx
import logging
from fastmcp import FastMCP


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("WeatherServer")


APP_NAME = "weather-server"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8031
DEFAULT_PATH = "/mcp"

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes to human-readable conditions
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

INSTRUCTIONS = """Weather MCP Server for weather lookup.

Available tools:
- get_weather: Get current weather for a city (e.g., "New York", "London", "Tokyo")

Data source: Open-Meteo API (free, no API key required).
"""

mcp = FastMCP(APP_NAME, instructions=INSTRUCTIONS)


@mcp.tool
async def get_weather(city: str) -> dict:
    """Get current weather conditions for a city (e.g. New York, London, Tokyo). Returns temperature, humidity, wind speed, and conditions."""
    coords = await _geocode(city)
    if not coords:
        return {"city": city, "error": f"Could not find location: {city}"}

    weather = await _fetch_weather(coords["latitude"], coords["longitude"])
    if not weather:
        return {"city": city, "error": f"Could not fetch weather for {city}"}

    result = {"city": coords["name"], **weather}
    logger.info(f"Weather for {city}: {result}")
    return result


async def _geocode(city: str) -> dict | None:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                GEOCODING_URL,
                params={"name": city, "count": 1, "language": "en"},
                timeout=10.0,
            )
            data = resp.json()

        results = data.get("results", [])
        if not results:
            logger.warning(f"Geocoding found no results for: {city}")
            return None

        loc = results[0]
        logger.info(f"Geocoded {city} -> {loc['name']} ({loc['latitude']}, {loc['longitude']})")
        return loc
    except Exception as e:
        logger.error(f"Geocoding failed for {city}: {e}")
        return None


async def _fetch_weather(lat: float, lon: float) -> dict | None:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                WEATHER_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                },
                timeout=10.0,
            )
            data = resp.json()

        current = data.get("current", {})
        if not current:
            return None

        weather_code = current.get("weather_code", -1)
        return {
            "temp": round(current.get("temperature_2m", 0)),
            "feels_like": round(current.get("apparent_temperature", 0)),
            "condition": WMO_CODES.get(weather_code, f"Unknown ({weather_code})"),
            "humidity": current.get("relative_humidity_2m", 0),
            "wind_speed": round(current.get("wind_speed_10m", 0)),
        }
    except Exception as e:
        logger.error(f"Weather fetch failed for ({lat}, {lon}): {e}")
        return None


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        path=DEFAULT_PATH,
    )
