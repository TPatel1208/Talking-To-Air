from dotenv import load_dotenv
import os
import shutil
import earthaccess
import xarray as xr

# -----------------------------------------------------------------------------
# 1. Authentication
# -----------------------------------------------------------------------------
load_dotenv()
auth = earthaccess.login(strategy="environment")
print(f"Authenticated: {auth}")

# -----------------------------------------------------------------------------
# 2. Clean download directory
# -----------------------------------------------------------------------------
SAVE_DIR = os.path.abspath("../data")
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 3. Search TEMPO NO2 L3
# -----------------------------------------------------------------------------
results = earthaccess.search_data(
    short_name="TEMPO_NO2_L3",
    temporal=("2024-07-01", "2024-07-02"),
)
print(f"Number of granules found: {len(results)}")

# ----------------------------
# 4. Download serially (Windows safe)
# ----------------------------
paths = []
for granule in results:
    path = earthaccess.download(granule, SAVE_DIR,show_progress=True)
    paths.append(path)
print(f"Number of files downloaded: {len(paths)}")

# ----------------------------
# 5. Load with xarray
# ----------------------------
ds = xr.open_mfdataset(
    sorted(paths),
    combine="nested",      # critical for TEMPO L3
    concat_dim="time",
    engine="netcdf4",
    parallel=False,
    chunks={"time": 1}
)

print("Dataset loaded successfully")
print(ds)