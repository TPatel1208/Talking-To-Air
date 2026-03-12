import earthaccess
from dotenv import load_dotenv
import os 
load_dotenv() # EARTHDATA_USERNAME, EARTHDATA_PASSWORD

auth = earthaccess.login()

if not auth:
    #login with interactive strategy if not already logged in
    earthaccess.login(strategy="interactive",persist=True)

print(earthaccess.__version__)

results = earthaccess.search_data(
    short_name = "TEMPO_NO2_L3",
    temporal = ("2026-01-11 12:00", "2026-01-12 12:00")
)

print("found", len(results), "results")