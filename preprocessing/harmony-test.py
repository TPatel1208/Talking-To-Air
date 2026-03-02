from dotenv import load_dotenv
from harmony import BBox, Client, Collection, Request, Environment
import datetime as dt
import json
import os
import time
import hashlib
import xarray as xr
import zarr
import concurrent.futures

load_dotenv()

harmony_client = Client(env=Environment.PROD,  auth=(os.getenv("EDL_USERNAME"), os.getenv("EDL_PASSWORD")))

COLLECTION_ID = "C3685896708-LARC_CLOUD"
start_str = "2026-02-10T18:00:00Z"
end_str   = "2026-02-10T19:00:00Z"
fmt = "%Y-%m-%dT%H:%M:%SZ"
start_dt = dt.datetime.strptime(start_str, fmt)
end_dt   = dt.datetime.strptime(end_str, fmt)

BBOX = [-125, 25, -66.5, 49.5]  # CONUS bounding box
CACHE_PATH = "cache.zarr"

def make_group_key(collection_id, start_str, end_str, bbox):
    bbox_str = "_".join(map(str, bbox))
    raw = f"{collection_id}_{start_str}_{end_str}_{bbox_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def is_cached(cache_path, group_key):
    try:
        store = zarr.open(cache_path, mode="r")
        return group_key in store
    except:
        return False

group_key = make_group_key(COLLECTION_ID, start_str, end_str, BBOX)
print(f"Cache group key: {group_key}")

if is_cached(CACHE_PATH, group_key):
    print("Cache hit — loading from Zarr")
    combined = xr.open_zarr(CACHE_PATH, group=group_key)
else:
    print("Cache miss — fetching from Harmony")
    collection = Collection(id=COLLECTION_ID)

    request = Request(
        collection=collection,
        spatial=BBox(*BBOX),
        temporal={
            'start': start_dt,
            'stop': end_dt
        },
        variables=['product/vertical_column_troposphere'],
        max_results=10,
        format='application/x-netcdf4'
    )
    print(request.is_valid())

    job_id = harmony_client.submit(request)
    status = harmony_client.status(job_id)
    print(json.dumps(status, default=lambda x: x.isoformat() if isinstance(x, dt.datetime) else str(x), indent=2))

    start = time.time()
    harmony_client.wait_for_processing(job_id, show_progress=True)

    datasets = []
    futures = harmony_client.download_all(job_id, directory='./downloads', overwrite=True)

    for future in concurrent.futures.as_completed(futures):
        filename = future.result()
        print("Downloaded:", filename)
        try:
            ds = xr.open_dataset(filename, group="product", engine="netcdf4")
        except Exception:
            ds = xr.open_dataset(filename, engine="netcdf4")
        datasets.append(ds)

    if len(datasets) == 1:
        combined = datasets[0]
    elif len(datasets) > 1:
        combined = xr.concat(datasets, dim='time')

    # Save to Zarr cache
    combined.to_zarr(CACHE_PATH, group=group_key, mode="w", consolidated=True)
    print(f"Cached to: {CACHE_PATH}/{group_key}")
    print("Total fetch time:", time.time() - start)