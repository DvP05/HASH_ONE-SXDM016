import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NASA_API_KEY")

test_cases = [
    {"source": "VIIRS_SNPP_NRT", "extent": "world", "day": 1},
    {"source": "MODIS_NRT", "extent": "-180,-90,180,90", "day": 1},
    {"source": "VIIRS_SNPP_NRT", "extent": "-120,30,-110,40", "day": 1},
]

for i, case in enumerate(test_cases):
    source = case["source"]
    extent = case["extent"]
    if extent == "world": extent = "-180,-90,180,90"
    day = case["day"]
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{extent}/{day}"
    print(f"\nTest {i+1}: {source} | {extent}")
    try:
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Success! Rows: {len(response.text.splitlines())}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
