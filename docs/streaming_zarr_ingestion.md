# Streaming Zarr Ingestion

Harmony granule ingestion now keeps parsed datasets bounded by `GRANULE_CONCURRENCY`.

Before this change, the Harmony fallback parsed every downloaded NetCDF granule, kept
each parsed `xarray.Dataset` in a request-wide list, concatenated the full list, and
then wrote the combined dataset to Zarr. Peak memory therefore scaled with the total
number of parseable granules in the request.

The Harmony path now processes fixed-size windows:

1. Submit the Harmony job and retrieve result links.
2. Download up to `GRANULE_CONCURRENCY` granules.
3. Parse that same window and unlink every NetCDF file in the parse `finally` block.
4. Normalize the parsed window for append-safe Zarr writes.
5. Write the first window with `mode="w"` and later windows with `append_dim="time"`.
6. Close and release the window datasets before downloading the next window.
7. Return `xr.open_zarr(...)` from the completed cache group.

The normalization layer standardizes coordinate dtypes, variable dtypes, non-append
dimensions, and Zarr chunk encoding against the first written window. Incompatible
windows fail before append with a clear normalization error.

Memory now scales with the configured window size:

```text
Peak parsed dataset memory = O(GRANULE_CONCURRENCY)
```

It no longer scales with the total number of granules in the Harmony request. NetCDF
disk usage is also bounded to the active download/parse window for the link-based
Harmony path.
