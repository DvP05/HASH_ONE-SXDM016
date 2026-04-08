import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NASA_API_KEY")

# USA roughly: -125 (xmin), 25 (ymin), -65 (xmax), 50 (ymax)
# That's 60x25 degrees. Might be too big for VIIRS (50x50).
# Let's try a smaller 30x30 degree chunk: -120 (xmin), 30 (ymin), -90 (xmax), 60 (ymax)
test_cases = [
    {"source": "VIIRS_SNPP_NRT", "extent": "-120,30,-90,60", "day": 1},
    {"source": "MODIS_NRT", "extent": "-120,30,-70,60", "day": 1}, # MODIS allows 100x100
]

for i, case in enumerate(test_cases):
    source = case["source"]
    extent = case["extent"]
    day = case["day"]
    
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{extent}/{day}"
    print(f"\nTest {i+1}: {source} | {extent}")
    try:
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Success! Rows: {len(response.text.splitlines())}")
            print(response.text[:200])
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
