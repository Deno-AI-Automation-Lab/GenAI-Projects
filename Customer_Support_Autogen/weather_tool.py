"""Live weather from Open-Meteo (free, no API key), used by the weather agent.

News search rarely contains the current temperature for a city, so the weather
agent gets real measurements here and uses web_search only for alerts/warnings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from typing import Annotated, Any

from config import settings
from search_tool import EVIDENCE

logger = logging.getLogger("support_bot.weather")

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> text
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers", 85: "snow showers",
    86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def _http_get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "support-bot/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 (fixed https hosts)
        return json.loads(resp.read().decode("utf-8"))


def _geocode(location: str) -> dict[str, Any] | None:
    name = location.split(",")[0].strip()  # "Chennai, Tamil Nadu" -> "Chennai"
    data = _http_get_json(_GEO_URL, {"name": name, "count": 5, "language": "en", "format": "json"})
    results = data.get("results") or []
    if not results:
        return None
    region_cc = getattr(settings, "search_region", "in-en").split("-")[0].upper()
    hint = location.lower()
    # Prefer a match whose state/country appears in the user's text, then the user's region, then the top hit.
    for r in results:
        if any(str(r.get(k, "")).lower() in hint for k in ("admin1", "country") if r.get(k)):
            return r
    return next((r for r in results if r.get("country_code") == region_cc), results[0])


def _fetch(location: str, days: int) -> str:
    place = _geocode(location)
    if not place:
        return f"Weather lookup: no place called '{location}' was found. Ask the customer to check the spelling."

    data = _http_get_json(_FORECAST_URL, {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,"
                   "wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                 "precipitation_sum",
        "timezone": "auto",
        "forecast_days": days,
    })
    label = ", ".join(x for x in (place.get("name"), place.get("admin1"), place.get("country")) if x)
    c, cu = data.get("current", {}), data.get("current_units", {})
    now = (
        f"Current ({c.get('time', '?')} local): {_WMO.get(c.get('weather_code'), 'unknown')}, "
        f"{c.get('temperature_2m')}{cu.get('temperature_2m', '°C')} "
        f"(feels like {c.get('apparent_temperature')}{cu.get('apparent_temperature', '°C')}), "
        f"humidity {c.get('relative_humidity_2m')}%, wind {c.get('wind_speed_10m')} {cu.get('wind_speed_10m', 'km/h')}, "
        f"precipitation {c.get('precipitation')} mm."
    )
    d = data.get("daily", {})
    lines = [now]
    for i, day in enumerate(d.get("time", [])):
        lines.append(
            f"{day}: {_WMO.get(d['weather_code'][i], 'unknown')}, "
            f"{d['temperature_2m_min'][i]}-{d['temperature_2m_max'][i]}°C, "
            f"rain chance {d['precipitation_probability_max'][i]}%, rain {d['precipitation_sum'][i]} mm."
        )
    body = "\n".join(lines)

    source_id = EVIDENCE.add(
        {"title": f"Live weather for {label} (Open-Meteo)", "body": body,
         "url": f"https://open-meteo.com/en/docs#latitude={place['latitude']}&longitude={place['longitude']}",
         "date": str(c.get("time", ""))[:10], "source": "Open-Meteo"},
        max_chars=2000,
    )
    return f"[{source_id}] Live weather for {label} (Open-Meteo)\n{body}"


async def get_weather(
    location: Annotated[str, "City or place name, e.g. 'Chennai' or 'Kanyakumari, Tamil Nadu'."],
    days: Annotated[int, "Number of forecast days to include, 1-7. Default 3."] = 3,
) -> str:
    """Get live current weather and a short daily forecast for a place.

    Use this for any question about current weather, temperature, humidity,
    wind, rain chances or the forecast for the next few days. It returns
    measured values (not news), labelled with an id like [S1] to cite.

    Args:
        location: City, town or place name. Add the state or country if the name is ambiguous.
        days: How many forecast days to return (1-7).

    Returns:
        Current conditions and a day-by-day forecast, or a message explaining
        why the lookup failed.
    """
    location = (location or "").strip()
    if not location:
        return "Weather lookup skipped: no location given."
    days = max(1, min(int(days), 7))
    EVIDENCE.searches.append(f"get_weather('{location}', days={days})")
    try:
        return await asyncio.to_thread(_fetch, location, days)
    except Exception as exc:  # network errors, unexpected payloads
        logger.warning("Weather lookup failed for %s: %s", location, exc)
        return (f"Live weather lookup failed ({type(exc).__name__}). "
                "Use web_search for the forecast and say live data was unavailable.")
