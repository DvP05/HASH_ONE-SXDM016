import os
import requests
import pandas as pd
from tools.registry import tool

@tool(name="fetch_nasa_firms", description="Fetch active fire data from NASA FIRMS API", category="api")
def fetch_nasa_firms(map_key: str = "VIIRS_SNPP_NRT", area_coords: str = "world") -> pd.DataFrame:
    """
    Hits the NASA FIRMS API and returns a DataFrame of active fire telemetry.
    The default model/sensor is set to VIIRS_SNPP_NRT (375m resolution).
    """
    # Pull the API key from environment variables
    api_key = os.getenv("NASA_API_KEY")
    if not api_key:
        raise ValueError("NASA_API_KEY environment variable is not set. Please add it to your .env file.")
        
    # Handle "world" alias
    extent = area_coords
    if area_coords.lower() == "world":
        extent = "-180,-90,180,90"
        
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{map_key}/{extent}/1"
    
    # NASA FIRMS returns CSV format directly
    df = pd.read_csv(url)
    return df
    
@tool(name="fetch_nasa_power", description="Fetch meteorological data from NASA POWER API", category="api")
def fetch_nasa_power(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Hits the NASA POWER API for weather/climate data and returns a DataFrame."""
    # NASA POWER does not require an API key for general usage
    url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
           f"parameters=T2M,PRECTOTCORR,RH2M&community=RE&longitude={lon}&latitude={lat}"
           f"&start={start_date}&end={end_date}&format=CSV")
           
    # Skip header lines that NASA POWER returns
    df = pd.read_csv(url, skiprows=10)
    return df
