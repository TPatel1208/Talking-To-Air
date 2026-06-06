#!/usr/bin/env python3
"""
datasets/tools/generate_collection.py

Auto-generate a collections.yaml entry for a NASA dataset by querying CMR
and inspecting a sample granule.

Usage Examples
--------------
    # Print YAML to stdout (review before committing):
    python datasets/tools/generate_collection.py --short-name TEMPO_NO2_L3 --version V04

    # Append directly to collections.yaml after review prompt:
    python datasets/tools/generate_collection.py --short-name TEMPO_NO2_L3 --version V04 --append

    # Skip granule download (faster; fill_value will be REVIEW_REQUIRED):
    python datasets/tools/generate_collection.py --short-name TEMPO_NO2_L3 --version V04 --no-granule

    # Validate an existing entry against live CMR data:
    python datasets/tools/generate_collection.py --validate TEMPO_NO2_V04

What This Does
---------------
1. CMR collection search  → concept ID, description, version
2. CMR UMM-Var fetch      → variable list, Harmony capability flags
3. Granule inspection     → fill_value, valid_min, valid_max, groups

Output: a YAML block with REVIEW_REQUIRED markers on fields needing human
decision (primary_var, units). Everything else is auto-populated from CMR.

Requirements
------------
    earthaccess >= 0.9  (for granule download)
    netCDF4 or h5py     (for granule inspection)
    pyyaml, pydantic    (already in requirements.txt)
"""

import argparse
import logging
import math
import pathlib
import sys
import tempfile
from typing import Any, Optional

import yaml

# ─────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────

_HERE = pathlib.Path(__file__).resolve().parent  # datasets/tools/
_BACKEND = _HERE.parent.parent  # Backend/
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_DEFAULT_REGISTRY = _HERE.parent / "collections.yaml"
CMR_BASE = "https://cmr.earthdata.nasa.gov/search"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_REVIEW = "REVIEW_REQUIRED"


# ─────────────────────────────────────────────────────────────────────────
# CMR API Helpers
# ─────────────────────────────────────────────────────────────────────────

def _cmr_get(endpoint: str, params: dict) -> dict:
    """GET a CMR endpoint, return JSON, raise on error."""
    import requests
    url = f"{CMR_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_collection_meta(short_name: str, version: str) -> dict:
    """Fetch CMR UMM-JSON collection metadata."""
    data = _cmr_get(
        "collections.umm_json",
        {"short_name": short_name, "version": version, "page_size": 1},
    )
    items = data.get("items", [])
    if not items:
        raise ValueError(
            f"No CMR collection found: short_name={short_name!r} version={version!r}\n"
            "Check https://search.earthdata.nasa.gov"
        )
    return items[0]


def fetch_umm_variables(concept_id: str) -> list[dict]:
    """Fetch UMM-Var records for a collection."""
    try:
        data = _cmr_get(
            "variables.umm_json",
            {"concept_id": concept_id, "page_size": 100},
        )
        return data.get("items", [])
    except Exception as exc:
        logger.warning("UMM-Var fetch failed (non-fatal): %s", exc)
        return []


def parse_harmony_capabilities(collection_meta: dict) -> dict:
    """Extract Harmony service capabilities from UMM-JSON."""
    defaults = {
        "has_variables": False,
        "has_spatial_subsetting": False,
        "has_temporal_subsetting": False,
        "has_formats": False,
    }

    umm = collection_meta.get("umm", {})
    services = umm.get("AssociatedServices", []) or []

    for svc in services:
        opts = svc.get("ServiceOptions", {}) or {}
        subset = opts.get("Subset", {}) or {}
        if subset.get("VariableSubset"):
            defaults["has_variables"] = True
        if subset.get("SpatialSubset"):
            defaults["has_spatial_subsetting"] = True
        if subset.get("TemporalSubset"):
            defaults["has_temporal_subsetting"] = True

    return defaults


# ─────────────────────────────────────────────────────────────────────────
# Granule Inspection
# ─────────────────────────────────────────────────────────────────────────

def download_sample_granule(concept_id: str, dest: pathlib.Path) -> Optional[pathlib.Path]:
    """Download one granule, return its path or None."""
    try:
        import earthaccess
        earthaccess.login(strategy="environment")
        granules = earthaccess.search_data(concept_id=concept_id, count=1)
        if not granules:
            logger.warning("No granules found for %s", concept_id)
            return None
        files = earthaccess.download(granules[:1], local_path=str(dest))
        return pathlib.Path(files[0]) if files else None
    except Exception as exc:
        logger.warning("Granule download failed: %s", exc)
        return None


def inspect_granule(filepath: pathlib.Path) -> dict[str, Any]:
    """Open a granule file, extract groups, variables, fill values, valid ranges."""
    result = {
        "groups": [],
        "variables": [],
        "fill_values": {},
        "valid_mins": {},
        "valid_maxes": {},
    }

    # Try netCDF4 first
    try:
        import netCDF4 as nc
        with nc.Dataset(str(filepath), "r") as ds:
            for varname, var in ds.variables.items():
                result["variables"].append(varname)
                _extract_nc_attrs(var, varname, result)
            for grp_name, grp in ds.groups.items():
                result["groups"].append(grp_name)
                for varname, var in grp.variables.items():
                    path = f"{grp_name}/{varname}"
                    result["variables"].append(path)
                    _extract_nc_attrs(var, path, result)
        return result
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("netCDF4 failed: %s", exc)

    # Fall back to h5py
    try:
        import h5py
        with h5py.File(str(filepath), "r") as f:
            def visit(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                parts = name.split("/")
                if len(parts) > 1 and parts[0] not in result["groups"]:
                    result["groups"].append(parts[0])
                result["variables"].append(name)
                _extract_h5_attrs(obj, name, result)

            f.visititems(visit)
        return result
    except ImportError:
        logger.warning("Neither netCDF4 nor h5py available — skipping granule inspection")
    except Exception as exc:
        logger.warning("h5py failed: %s", exc)

    return result


def _extract_nc_attrs(var, path: str, result: dict):
    """Extract _FillValue, valid_min, valid_max from netCDF4 Variable."""
    try:
        result["fill_values"][path] = float(var._FillValue)
    except AttributeError:
        pass
    for attr in ("valid_min", "valid_max"):
        try:
            val = getattr(var, attr)
            result[attr + "s"][path] = float(val)
        except AttributeError:
            pass


def _extract_h5_attrs(obj, path: str, result: dict):
    """Extract fillvalue, valid_min, valid_max from h5py Dataset."""
    if obj.fillvalue is not None:
        try:
            result["fill_values"][path] = float(obj.fillvalue)
        except (TypeError, ValueError):
            pass
    attrs = dict(obj.attrs)
    for attr in ("valid_min", "valid_max"):
        val = attrs.get(attr)
        if val is not None:
            try:
                result[attr + "s"][path] = float(val)
            except (TypeError, ValueError):
                pass


# ─────────────────────────────────────────────────────────────────────────
# Registry Entry Assembly
# ─────────────────────────────────────────────────────────────────────────

def make_registry_key(short_name: str, version: str) -> str:
    """Generate a stable registry key from short_name and version."""
    base = short_name.upper().replace("-", "_")
    ver = version.upper().replace(".", "_").replace("/", "_")
    # Avoid double-appending
    if base.endswith(f"_{ver}") or base.endswith(ver):
        return base
    return f"{base}_{ver}"


def merge_variables(umm_vars: list[str], granule_vars: list[str]) -> list[str]:
    """Combine and deduplicate variable lists, return sorted."""
    seen = set()
    result = []
    for v in umm_vars + granule_vars:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return sorted(result)


def pick_fill_value(fill_values: dict[str, float]) -> Optional[float]:
    """Return most common fill_value or None."""
    if not fill_values:
        return None
    candidates = [v for v in fill_values.values() if not math.isnan(v)]
    if not candidates:
        return None
    from collections import Counter
    return Counter(candidates).most_common(1)[0][0]


def pick_stat(stats: dict[str, float], kind: str) -> Optional[float]:
    """Return min or max of non-NaN values."""
    if not stats:
        return None
    candidates = [v for v in stats.values() if not math.isnan(v)]
    if not candidates:
        return None
    return min(candidates) if kind == "min" else max(candidates)


def build_entry(
    short_name: str,
    version: str,
    fetch_granule: bool = True,
) -> tuple[str, dict, list[str]]:
    """
    Build a complete registry entry.

    Returns
    -------
    (registry_key, entry_dict, warnings_list)
    """
    warnings = []

    # 1. CMR collection
    logger.info("Fetching CMR collection metadata for %s %s …", short_name, version)
    col_meta = fetch_collection_meta(short_name, version)
    concept_id = col_meta["meta"]["concept-id"]
    umm = col_meta.get("umm", {})
    description = umm.get("Abstract") or umm.get("EntryTitle") or ""
    logger.info("  concept_id: %s", concept_id)

    # 2. Harmony capabilities
    caps = parse_harmony_capabilities(col_meta)
    supports_subsetting = caps["has_variables"]

    # 3. UMM-Var
    logger.info("Fetching UMM-Var variable list …")
    umm_vars = fetch_umm_variables(concept_id)
    var_names = [v["umm"]["Name"] for v in umm_vars if v.get("umm", {}).get("Name")]
    logger.info("  Found %d variables in UMM-Var", len(var_names))

    # 4. Granule inspection
    granule_info = {
        "groups": [],
        "variables": [],
        "fill_values": {},
        "valid_mins": {},
        "valid_maxes": {},
    }

    if fetch_granule:
        with tempfile.TemporaryDirectory() as tmp:
            logger.info("Downloading sample granule …")
            sample = download_sample_granule(concept_id, pathlib.Path(tmp))
            if sample:
                logger.info("  Inspecting %s …", sample.name)
                granule_info = inspect_granule(sample)
                logger.info(
                    "  Found %d vars, %d groups",
                    len(granule_info["variables"]),
                    len(granule_info["groups"]),
                )
            else:
                warnings.append("Granule download failed — fill_value/valid_range will be REVIEW_REQUIRED")
    else:
        warnings.append("--no-granule: fill_value/valid_range set to REVIEW_REQUIRED")

    # 5. Merge variables
    all_vars = merge_variables(var_names, granule_info["variables"])

    # 6. Fill values and ranges
    fill_value = pick_fill_value(granule_info["fill_values"])
    valid_min = pick_stat(granule_info["valid_mins"], "min")
    valid_max = pick_stat(granule_info["valid_maxes"], "max")

    if fill_value is None:
        warnings.append("fill_value not found in granule")
    if valid_min is None:
        warnings.append("valid_min not found in granule")
    if valid_max is None:
        warnings.append("valid_max not found in granule")

    # 7. Assemble
    key = make_registry_key(short_name, version)
    entry = {
        "collection_id": concept_id,
        "short_name": short_name,
        "version": version,
        "description": description,
        "primary_var": _REVIEW,
        "quality_flag_var": None,
        "variables": all_vars if supports_subsetting else [],
        "supports_variable_subsetting": supports_subsetting,
        "units": _REVIEW,
        "fill_value": fill_value if fill_value is not None else _REVIEW,
        "valid_min": valid_min if valid_min is not None else _REVIEW,
        "valid_max": valid_max if valid_max is not None else _REVIEW,
        "groups": granule_info["groups"],
    }

    warnings.insert(0, f"primary_var — choose from: {', '.join(all_vars) or '(none discovered)'}")
    qa_vars = [v for v in all_vars if any(kw in v.lower() for kw in ("qa", "quality", "flag", "qf"))]
    if qa_vars:
        warnings.append(f"quality_flag_var candidates: {', '.join(qa_vars)}")

    return key, entry, warnings


# ─────────────────────────────────────────────────────────────────────────
# YAML Rendering
# ─────────────────────────────────────────────────────────────────────────

def _float_representer(dumper, value):
    """Render inf as .inf, keep floats readable."""
    if math.isinf(value):
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".inf" if value > 0 else "-.inf")
    if math.isnan(value):
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".nan")
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:g}")


yaml.add_representer(float, _float_representer)


def render_yaml(key: str, entry: dict, warnings: list[str]) -> str:
    """Render entry as commented YAML block."""
    lines = [
        "# " + "─" * 72,
        "# AUTO-GENERATED — review all REVIEW_REQUIRED fields before committing",
        "# " + "─" * 72,
    ]

    for w in warnings:
        lines.append(f"# ⚠  {w}")

    if warnings:
        lines.append("#")

    lines.append(f"{key}:")
    for field, value in entry.items():
        if value == _REVIEW:
            lines.append(f"  {field}: {_REVIEW}  # ← MUST REVIEW")
        elif isinstance(value, list):
            if value:
                lines.append(f"  {field}:")
                for item in value:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"  {field}: []")
        elif isinstance(value, bool):
            lines.append(f"  {field}: {'true' if value else 'false'}")
        elif value is None:
            lines.append(f"  {field}: null")
        elif isinstance(value, str) and value.replace(".", "").replace("-", "").isdigit():
            lines.append(f'  {field}: "{value}"')
        else:
            lines.append(f"  {field}: {value}")

    return "\n".join(lines) + "\n"


def validate_entry(registry_key: str, registry_path: pathlib.Path):
    """Re-fetch CMR data and check for drift in an existing entry."""
    from datasets.registry import load_registry, reload_registry

    reload_registry()
    registry = load_registry()

    if registry_key not in registry:
        print(f"ERROR: '{registry_key}' not in {registry_path}")
        sys.exit(1)

    col = registry[registry_key]
    print(f"\nValidating {registry_key} …")

    try:
        meta = fetch_collection_meta(col.short_name, col.version)
        live_id = meta["meta"]["concept-id"]
        if live_id != col.collection_id:
            print(f"  ⚠  concept_id MISMATCH — CMR now has {live_id}")
        else:
            print(f"  ✓  concept_id matches")
    except Exception as exc:
        print(f"  ✗  CMR lookup failed: {exc}")
        return

    # Check variables exist in sample granule
    with tempfile.TemporaryDirectory() as tmp:
        sample = download_sample_granule(col.collection_id, pathlib.Path(tmp))
        if not sample:
            print("  ⚠  Granule download failed — skipping variable check")
            return

        info = inspect_granule(sample)
        discovered = set(info["variables"])

        for check_var, label in [(col.primary_var, "primary_var"), (col.quality_flag_var, "quality_flag_var")]:
            if not check_var:
                continue
            found = check_var in discovered or any(v.endswith(f"/{check_var}") for v in discovered)
            status = "✓" if found else "⚠  NOT FOUND"
            print(f"  {status}  {label}: {check_var}")

    print("Validation complete.\n")


def append_to_registry(yaml_block: str, registry_path: pathlib.Path):
    """Append yaml_block after user confirmation."""
    print("\n" + "─" * 76)
    print("Will append to:")
    print(f"  {registry_path}\n")
    print(yaml_block)
    print("─" * 76)

    answer = input("Append? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    with open(registry_path, "a", encoding="utf-8") as f:
        f.write("\n" + yaml_block)

    print(f"✓  Appended to {registry_path}")
    print("Review REVIEW_REQUIRED fields before restarting the backend.")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate or validate a collections.yaml entry from CMR."
    )
    p.add_argument("--short-name", help="CMR short_name (e.g. TEMPO_NO2_L3)")
    p.add_argument("--version", help="Collection version (e.g. V04)")
    p.add_argument("--append", action="store_true", help="Append to collections.yaml interactively")
    p.add_argument("--registry", default=str(_DEFAULT_REGISTRY), help="Path to collections.yaml")
    p.add_argument("--no-granule", action="store_true", help="Skip granule download")
    p.add_argument("--validate", metavar="KEY", help="Validate existing entry (e.g. --validate TEMPO_NO2_V04)")
    p.add_argument("--verbose", "-v", action="store_true")

    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    registry_path = pathlib.Path(args.registry)

    # Validate mode
    if args.validate:
        validate_entry(args.validate, registry_path)
        return

    # Generate mode
    if not args.short_name or not args.version:
        p.error("--short-name and --version required (or use --validate)")

    try:
        key, entry, warnings = build_entry(args.short_name, args.version, not args.no_granule)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    yaml_block = render_yaml(key, entry, warnings)

    if args.append:
        append_to_registry(yaml_block, registry_path)
    else:
        print(yaml_block)
        print("─" * 76)
        print("Review and re-run with --append to add to registry")


if __name__ == "__main__":
    main()