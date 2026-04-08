import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NASA_API_KEY")
map_key = "VIIRS_SNPP_NRT"
extent = "-180,-90,180,90"
day_range = 1

url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{map_key}/{extent}/{day_range}"

print(f"Testing URL: {url.replace(api_key, 'REDACTED')}")

try:
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success!")
        print(response.text[:500])
    else:
        print(f"Error Message: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
