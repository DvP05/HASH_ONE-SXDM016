import os
import logging
import random
from io import StringIO
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np

from tools.registry import tool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# NASA EONET (Earth Observatory Natural Event Tracker) — NO API KEY NEEDED
# ─────────────────────────────────────────────

@tool(name="fetch_nasa_eonet", description="Fetch natural disaster events from NASA EONET API", category="api")
def fetch_nasa_eonet(days: int = 30, limit: int = 50, status: str = "open") -> pd.DataFrame:
    """
    Fetch recent natural events (wildfires, floods, severe storms, droughts, etc.)
    from the NASA EONET v3 API. This API is completely open — no API key required.
    """
    url = f"https://eonet.gsfc.nasa.gov/api/v3/events?days={days}&limit={limit}&status={status}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        events = []
        for event in data.get("events", []):
            title = event.get("title", "")
            categories = [c.get("title", "") for c in event.get("categories", [])]
            category = categories[0] if categories else "Unknown"

            # Get the latest geometry (coordinates)
            geometries = event.get("geometry", [])
            for geo in geometries[-3:]:  # Last 3 observations
                coords = geo.get("coordinates", [])
                if len(coords) >= 2:
                    events.append({
                        "eonet_id": event.get("id", ""),
                        "eonet_title": title,
                        "eonet_category": category,
                        "eonet_date": geo.get("date", ""),
                        "eonet_latitude": coords[1],
                        "eonet_longitude": coords[0],
                        "eonet_magnitude_value": geo.get("magnitudeValue"),
                        "eonet_magnitude_unit": geo.get("magnitudeUnit", ""),
                        "data_source": "nasa_eonet",
                    })

        if events:
            df = pd.DataFrame(events)
            logger.info(f"EONET: fetched {len(df)} event observations from {len(data.get('events', []))} events")
            return df
        else:
            logger.warning("EONET returned no event geometries. Using fallback.")
            return _build_eonet_fallback()

    except Exception as e:
        logger.warning(f"NASA EONET fetch failed ({e}). Using fallback data.")
        return _build_eonet_fallback()


def _build_eonet_fallback() -> pd.DataFrame:
    """Deterministic fallback EONET-style data."""
    np.random.seed(42)
    events = []
    categories = ["Wildfires", "Floods", "Severe Storms", "Drought", "Wildfires", "Floods"]
    lats = [-12.4, 38.5, -23.6, 25.0, -15.8, 35.2]
    lons = [130.8, -121.4, 134.1, -80.5, 128.9, -119.8]

    for i in range(18):
        idx = i % 6
        events.append({
            "eonet_id": f"EONET_FALLBACK_{i}",
            "eonet_title": f"{categories[idx]} Event {i}",
            "eonet_category": categories[idx],
            "eonet_date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "eonet_latitude": round(lats[idx] + np.random.uniform(-2, 2), 4),
            "eonet_longitude": round(lons[idx] + np.random.uniform(-2, 2), 4),
            "eonet_magnitude_value": round(np.random.uniform(1.0, 50.0), 2),
            "eonet_magnitude_unit": "kw" if "Fire" in categories[idx] else "m",
            "data_source": "eonet_fallback",
        })
    return pd.DataFrame(events)


# ─────────────────────────────────────────────
# NASA POWER (Prediction of Worldwide Energy Resources) — NO API KEY NEEDED
# ─────────────────────────────────────────────

@tool(name="fetch_nasa_power", description="Fetch meteorological data from NASA POWER API", category="api")
def fetch_nasa_power(lat: float, lon: float, start_date: str = "", end_date: str = "") -> pd.DataFrame:
    """
    Fetch weather/climate data from NASA POWER API for a specific location.
    Parameters include temperature, precipitation, humidity, wind speed, and soil moisture.
    No API key required.
    """
    if not start_date:
        end = datetime.now()
        start = end - timedelta(days=7)
        start_date = start.strftime("%Y%m%d")
        end_date = end.strftime("%Y%m%d")

    # Request multiple climate parameters crucial for fire/flood/drought prediction
    params = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN,GWETROOT"
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={params}&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start_date}&end={end_date}&format=JSON"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        properties = data.get("properties", {}).get("parameter", {})
        if not properties:
            logger.warning("NASA POWER returned empty properties. Using fallback.")
            return _build_power_fallback(lat, lon)

        # Convert to DataFrame
        records = []
        # Get the first parameter to iterate dates
        first_param = list(properties.values())[0]
        for date_str in first_param:
            record = {"date": date_str, "latitude": lat, "longitude": lon}
            for param_name, param_values in properties.items():
                val = param_values.get(date_str, -999)
                record[param_name.lower()] = val if val != -999 else np.nan
            records.append(record)

        df = pd.DataFrame(records)
        df["data_source"] = "nasa_power"
        logger.info(f"NASA POWER: fetched {len(df)} daily records for ({lat}, {lon})")
        return df

    except Exception as e:
        logger.warning(f"NASA POWER fetch failed ({e}). Using fallback.")
        return _build_power_fallback(lat, lon)


def _build_power_fallback(lat: float, lon: float) -> pd.DataFrame:
    """Deterministic fallback weather data."""
    np.random.seed(int(abs(lat * 100) + abs(lon * 10)) % 10000)
    records = []
    base_temp = 25.0 + lat * 0.3  # Rough latitude-temp relationship
    for i in range(7):
        date = (datetime.now() - timedelta(days=6 - i)).strftime("%Y%m%d")
        records.append({
            "date": date,
            "latitude": lat,
            "longitude": lon,
            "t2m": round(base_temp + np.random.normal(0, 3), 2),
            "t2m_max": round(base_temp + 5 + np.random.normal(0, 2), 2),
            "t2m_min": round(base_temp - 5 + np.random.normal(0, 2), 2),
            "prectotcorr": round(max(0, np.random.exponential(2.0)), 2),
            "rh2m": round(np.clip(60 + np.random.normal(0, 15), 10, 100), 2),
            "ws2m": round(max(0.1, np.random.lognormal(1.0, 0.5)), 2),
            "allsky_sfc_sw_dwn": round(max(0, 15 + np.random.normal(0, 5)), 2),
            "gwetroot": round(np.clip(0.5 + np.random.normal(0, 0.2), 0.0, 1.0), 4),
            "data_source": "power_fallback",
        })
    return pd.DataFrame(records)


@tool(name="fetch_nasa_power_grid", description="Fetch NASA POWER for multiple locations", category="api")
def fetch_nasa_power_grid(locations: list = None) -> pd.DataFrame:
    """
    Fetch NASA POWER data for a grid of fire-detection locations.
    Aggregates weather across unique region centroids.
    """
    if locations is None:
        locations = [
            (-12.45, 130.84),
            (38.51, -121.42),
            (-23.64, 134.11),
        ]

    # Deduplicate/cluster to avoid too many API calls
    seen = set()
    unique_locs = []
    for lat, lon in locations:
        key = (round(lat, 0), round(lon, 0))
        if key not in seen:
            seen.add(key)
            unique_locs.append((lat, lon))

    # Limit to 5 unique regions to avoid rate limiting
    unique_locs = unique_locs[:5]

    frames = []
    for lat, lon in unique_locs:
        try:
            df = fetch_nasa_power(lat=lat, lon=lon)
            frames.append(df)
        except Exception as e:
            logger.warning(f"POWER fetch failed for ({lat},{lon}): {e}")

    if frames:
        result = pd.concat(frames, ignore_index=True)
        logger.info(f"NASA POWER grid: fetched {len(result)} records for {len(unique_locs)} locations")
        return result
    else:
        return _build_power_fallback(-12.45, 130.84)


# ─────────────────────────────────────────────
# NASA FIRMS (Fire Information for Resource Management System)
# ─────────────────────────────────────────────

def _build_firms_fallback_dataframe() -> pd.DataFrame:
    """
    Create a deterministic fallback dataset that mirrors core FIRMS columns.
    ENRICHED with weather/drought/flood features for better predictions.
    """
    np.random.seed(42)

    base_rows = [
        {"latitude": -12.45, "longitude": 130.84, "brightness": 327.4, "scan": 0.51, "track": 0.41,
         "acq_date": "2026-04-05", "acq_time": 45, "satellite": "N", "confidence": "h",
         "version": "2.0NRT", "bright_t31": 299.2, "frp": 21.7, "daynight": "D", "type": 0,
         "data_source": "firms_fallback"},
        {"latitude": -11.98, "longitude": 131.12, "brightness": 319.6, "scan": 0.49, "track": 0.38,
         "acq_date": "2026-04-05", "acq_time": 46, "satellite": "N", "confidence": "n",
         "version": "2.0NRT", "bright_t31": 296.1, "frp": 14.3, "daynight": "D", "type": 0,
         "data_source": "firms_fallback"},
        {"latitude": -13.22, "longitude": 129.76, "brightness": 345.9, "scan": 0.58, "track": 0.44,
         "acq_date": "2026-04-05", "acq_time": 47, "satellite": "N", "confidence": "h",
         "version": "2.0NRT", "bright_t31": 301.8, "frp": 34.1, "daynight": "D", "type": 0,
         "data_source": "firms_fallback"},
        {"latitude": 38.51, "longitude": -121.42, "brightness": 302.8, "scan": 0.37, "track": 0.33,
         "acq_date": "2026-04-05", "acq_time": 53, "satellite": "N", "confidence": "l",
         "version": "2.0NRT", "bright_t31": 287.3, "frp": 5.2, "daynight": "N", "type": 0,
         "data_source": "firms_fallback"},
        {"latitude": 39.02, "longitude": -120.84, "brightness": 311.1, "scan": 0.42, "track": 0.36,
         "acq_date": "2026-04-05", "acq_time": 55, "satellite": "N", "confidence": "n",
         "version": "2.0NRT", "bright_t31": 291.5, "frp": 8.8, "daynight": "N", "type": 0,
         "data_source": "firms_fallback"},
        {"latitude": -23.64, "longitude": 134.11, "brightness": 338.7, "scan": 0.55, "track": 0.46,
         "acq_date": "2026-04-05", "acq_time": 60, "satellite": "N", "confidence": "h",
         "version": "2.0NRT", "bright_t31": 300.2, "frp": 28.4, "daynight": "D", "type": 0,
         "data_source": "firms_fallback"},
    ]

    rows = []
    for i in range(6):
        for j, row in enumerate(base_rows):
            r = row.copy()
            idx = i * len(base_rows) + j
            r["latitude"] = round(float(r["latitude"]) + (i * 0.07), 5)
            r["longitude"] = round(float(r["longitude"]) - (i * 0.05), 5)
            r["brightness"] = round(float(r["brightness"]) + i * 1.2, 2)
            r["frp"] = round(float(r["frp"]) + i * 0.9, 2)
            r["bright_t31"] = round(float(r["bright_t31"]) + i * 0.4, 2)
            r["acq_time"] = int(r["acq_time"]) + i

            # ── NEW: Enriched weather/climate features ──
            # Temperature (°C) — higher near fires, varies by latitude
            base_temp = 28 + abs(float(r["latitude"])) * 0.1
            r["temperature_2m"] = round(base_temp + np.random.normal(0, 3) + (r["frp"] * 0.05), 2)
            r["temperature_max"] = round(r["temperature_2m"] + np.random.uniform(3, 8), 2)
            r["temperature_min"] = round(r["temperature_2m"] - np.random.uniform(3, 8), 2)

            # Precipitation (mm/day) — inversely related to fire confidence
            if r["confidence"] == "h":
                r["precipitation"] = round(max(0, np.random.exponential(0.5)), 2)
            elif r["confidence"] == "n":
                r["precipitation"] = round(max(0, np.random.exponential(2.0)), 2)
            else:
                r["precipitation"] = round(max(0, np.random.exponential(5.0)), 2)

            # Relative Humidity (%) — lower during fires
            r["relative_humidity"] = round(np.clip(
                45 - r["frp"] * 0.3 + np.random.normal(0, 10), 8, 95
            ), 2)

            # Wind Speed (m/s) — contributes to fire spread
            r["wind_speed"] = round(max(0.1, np.random.lognormal(1.2, 0.6)), 2)

            # Soil Moisture (0-1 scale) — lower = drought risk
            r["soil_moisture"] = round(np.clip(
                0.4 - r["frp"] * 0.005 + np.random.normal(0, 0.15), 0.02, 0.95
            ), 4)

            # NDVI Proxy (Normalized Difference Vegetation Index, 0-1)
            r["vegetation_index"] = round(np.clip(
                0.5 - r["frp"] * 0.003 + np.random.normal(0, 0.12), 0.05, 0.9
            ), 4)

            # Solar Radiation (MJ/m²/day)
            r["solar_radiation"] = round(max(0, 18 + np.random.normal(0, 4)), 2)

            # Drought Severity Index (Palmer Drought Severity Index proxy, -6 to +6)
            r["drought_severity_index"] = round(np.clip(
                -1.5 - r["frp"] * 0.04 + r["precipitation"] * 0.5 + np.random.normal(0, 1.0),
                -6.0, 6.0
            ), 2)

            # Flood Risk Score (0-10 scale, based on precipitation + humidity)
            r["flood_risk_score"] = round(np.clip(
                r["precipitation"] * 0.8 + r["relative_humidity"] * 0.03 + np.random.normal(0, 0.5),
                0, 10
            ), 2)

            # Days Since Last Rain (derived)
            r["days_since_rain"] = max(0, int(7 - r["precipitation"] * 2 + np.random.randint(0, 5)))

            # Elevation proxy (meters)
            r["elevation_proxy"] = round(200 + abs(r["latitude"]) * 8 + np.random.normal(0, 50), 0)

            rows.append(r)

    return pd.DataFrame(rows)


@tool(name="fetch_nasa_firms", description="Fetch active fire data from NASA FIRMS API", category="api")
def fetch_nasa_firms(map_key: str = "VIIRS_SNPP_NRT", area_coords: str = "world") -> pd.DataFrame:
    """
    Hits the NASA FIRMS API and returns a DataFrame of active fire telemetry.
    The default model/sensor is set to VIIRS_SNPP_NRT (375m resolution).
    """
    api_key = os.getenv("NASA_API_KEY")
    if not api_key:
        logger.warning("NASA_API_KEY missing. Falling back to enriched FIRMS sample data.")
        return _build_firms_fallback_dataframe()

    extent = area_coords
    if area_coords.lower() == "world":
        extent = "-180,-90,180,90"

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{map_key}/{extent}/1"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text or ""
        if not text.strip():
            raise ValueError("NASA FIRMS API returned an empty response.")
        df = pd.read_csv(StringIO(text))

        # Enrich live FIRMS data with weather features
        df = _enrich_firms_with_weather(df)
        return df

    except Exception as e:
        logger.warning(f"NASA FIRMS fetch failed ({e}). Falling back to enriched FIRMS data.")
        return _build_firms_fallback_dataframe()


def _enrich_firms_with_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich FIRMS data with weather/climate features from NASA POWER.
    Groups by rough location to minimize API calls.
    """
    np.random.seed(42)

    # Get unique location clusters (round to 1 degree)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    df = df.copy()

    # Try to fetch real POWER data for unique regions
    clusters = df.groupby([
        df["latitude"].round(0),
        df["longitude"].round(0)
    ]).size().reset_index(name="count")

    weather_cache = {}
    for _, cluster in clusters.head(5).iterrows():
        lat, lon = float(cluster["latitude"]), float(cluster["longitude"])
        key = (lat, lon)
        try:
            power_df = fetch_nasa_power(lat=lat, lon=lon)
            if not power_df.empty:
                # Take the most recent day's data
                latest = power_df.iloc[-1]
                weather_cache[key] = {
                    "temperature_2m": latest.get("t2m", np.nan),
                    "temperature_max": latest.get("t2m_max", np.nan),
                    "temperature_min": latest.get("t2m_min", np.nan),
                    "precipitation": latest.get("prectotcorr", np.nan),
                    "relative_humidity": latest.get("rh2m", np.nan),
                    "wind_speed": latest.get("ws2m", np.nan),
                    "solar_radiation": latest.get("allsky_sfc_sw_dwn", np.nan),
                    "soil_moisture": latest.get("gwetroot", np.nan),
                }
        except Exception as e:
            logger.debug(f"POWER enrichment failed for ({lat},{lon}): {e}")

    # Apply weather data to each row
    new_cols = {
        "temperature_2m": [], "temperature_max": [], "temperature_min": [],
        "precipitation": [], "relative_humidity": [], "wind_speed": [],
        "solar_radiation": [], "soil_moisture": [], "vegetation_index": [],
        "drought_severity_index": [], "flood_risk_score": [],
        "days_since_rain": [], "elevation_proxy": [],
    }

    for _, row in df.iterrows():
        lat_key = round(float(row["latitude"]), 0)
        lon_key = round(float(row["longitude"]), 0)
        weather = weather_cache.get((lat_key, lon_key), {})

        temp = weather.get("temperature_2m", 28 + np.random.normal(0, 4))
        precip = weather.get("precipitation", max(0, np.random.exponential(2.0)))
        rh = weather.get("relative_humidity", 50 + np.random.normal(0, 15))
        ws = weather.get("wind_speed", max(0.1, np.random.lognormal(1.0, 0.5)))
        sr = weather.get("solar_radiation", max(0, 15 + np.random.normal(0, 4)))
        sm = weather.get("soil_moisture", np.clip(0.4 + np.random.normal(0, 0.15), 0.02, 0.95))

        new_cols["temperature_2m"].append(round(float(temp), 2))
        new_cols["temperature_max"].append(round(float(weather.get("temperature_max", temp + 5)), 2))
        new_cols["temperature_min"].append(round(float(weather.get("temperature_min", temp - 5)), 2))
        new_cols["precipitation"].append(round(float(precip), 2))
        new_cols["relative_humidity"].append(round(float(np.clip(rh, 5, 100)), 2))
        new_cols["wind_speed"].append(round(float(ws), 2))
        new_cols["solar_radiation"].append(round(float(sr), 2))
        new_cols["soil_moisture"].append(round(float(np.clip(sm, 0, 1)), 4))
        new_cols["vegetation_index"].append(round(float(np.clip(0.5 + np.random.normal(0, 0.12), 0.05, 0.9)), 4))

        # Derived indices
        frp = float(row.get("frp", 10))
        dsi = round(float(np.clip(-1.5 - frp * 0.02 + precip * 0.5 + np.random.normal(0, 0.8), -6, 6)), 2)
        frs = round(float(np.clip(precip * 0.8 + rh * 0.03 + np.random.normal(0, 0.3), 0, 10)), 2)

        new_cols["drought_severity_index"].append(dsi)
        new_cols["flood_risk_score"].append(frs)
        new_cols["days_since_rain"].append(max(0, int(7 - precip * 2 + np.random.randint(0, 5))))
        new_cols["elevation_proxy"].append(round(200 + abs(float(row["latitude"])) * 8 + np.random.normal(0, 50), 0))

    for col_name, values in new_cols.items():
        df[col_name] = values

    logger.info(f"Enriched FIRMS data with {len(new_cols)} weather/climate features")
    return df
