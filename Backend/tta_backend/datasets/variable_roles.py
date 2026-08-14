"""
datasets/variable_roles.py
==========================
A satellite product is a bundle of variables playing different roles: the
geophysical product a scientist wants (``vertical_column_troposphere``), the
quality variables that say whether to trust it (``main_data_quality_flag``,
``*_uncertainty``), viewing geometry and atmospheric/surface context bands
(``solar_zenith_angle``, ``radiative_cloud_frac``), and internal
retrieval/diagnostic fields (``amf_troposphere``, ``fitted_slant_column``).
``describe_dataset`` returns a flat variable inventory but classifies none of
it (PRD T35).

``classify_variable`` assigns each variable one of four canonical roles plus
an explicit fallback, combining evidence in strict precedence — the same
override->metadata->attrs house style as ``mask_info.resolve_mask_info`` and
``aggregation_service.identify_time`` (CF metadata over literal names):

  1. explicit CF ``standard_name`` (High)
  2. strong group membership -- ``qa_statistics/`` -> quality,
     ``geolocation/`` -> context (High)
  3. coordinate names (latitude/longitude/time/x/y) -> context (High)
  4. deterministic marker rules on the variable name (Medium)
  5. a recognized geophysical naming stem, as *positive* science evidence
     (Low), or membership in a science group (``product``/
     ``key_science_data``, High)
  6. otherwise -> ``unclassified`` (None)

**Science requires positive evidence** -- a CF ``standard_name``, a science
group, or a recognized geophysical stem -- and is never the residual default.
``processing_version`` / ``scan_line`` / ``orbit_number`` / ``weight`` fall to
``unclassified`` rather than being force-fit into a bucket: honest
incompleteness over false certainty.

Ordering note (verified against the real inventories captured under
``tests/fixtures/variable_inventories/``): exception markers are checked
before science, because a science stem is only a default *after* markers miss
(so ``product/radiative_cloud_frac`` is context, not science, and
``product/o3_below_cloud`` is a retrieval intermediate). Retrieval markers
(``amf*``, ``slant_column*``, ``*_below_cloud``, ``ghost*``) are checked
before the generic cloud/context markers so ``amf_cloud_fraction`` and
``o3_below_cloud`` read as retrieval intermediates rather than context; and
the cloud context markers are specific (``cloud_fraction`` / ``cloud_pressure``,
not bare ``cloud``) so a cloud-*screened* science column
(``ColumnAmountNO2CloudScreened``) is not mistaken for a cloud band.

Pure functions -- no I/O, no network -- so the classifier is trivially
unit-testable and has one canonical interpretation shared by the inventory UI
and (later) T36.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# ── Canonical roles ──────────────────────────────────────────────────────────
ROLE_SCIENCE = "science"
ROLE_QUALITY = "quality"
ROLE_CONTEXT = "context"
ROLE_RETRIEVAL_METADATA = "retrieval-metadata"
ROLE_UNCLASSIFIED = "unclassified"

# ── Confidence tiers (describe the decision, not the variable) ───────────────
CONFIDENCE_HIGH = "high"      # explicit CF metadata, strong group, coordinate
CONFIDENCE_MEDIUM = "medium"  # deterministic marker rule
CONFIDENCE_LOW = "low"        # geophysical naming stem (heuristic keyword)
CONFIDENCE_NONE: Optional[str] = None  # unclassified


# ── Group priors ─────────────────────────────────────────────────────────────
# Groups whose membership alone settles the role, regardless of the leaf name
# (PRD: "qa_statistics/* -> quality even when the name matches no rule;
# geolocation/* -> geometry unambiguously").
_QUALITY_GROUPS = {"qa_statistics"}
_GEOMETRY_GROUPS = {"geolocation"}
# Groups that are *positive evidence* for science -- but only for a variable
# that matched no earlier exception marker (checked last, so a cloud-fraction
# band living under product/ still reads as context).
_SCIENCE_GROUPS = {"product", "key_science_data"}

# ── Standard-name vocabulary ────────────────────────────────────────────────
_COORDINATE_STANDARD_NAMES = {
    "latitude", "longitude", "time",
    "projection_x_coordinate", "projection_y_coordinate",
}
# Substrings of a CF standard_name that mark a geophysical (science) quantity.
_SCIENCE_STANDARD_NAME_SUBSTRS = (
    "mole_content", "mole_fraction", "optical_depth", "optical_thickness",
    "column_number_density", "column_density", "number_density_of",
    "mass_content_of", "mixing_ratio",
)
# Substrings of a CF standard_name that mark a quality/diagnostic quantity.
_QUALITY_STANDARD_NAME_SUBSTRS = (
    "quality_flag", "status_flag", "standard_error",
    "number_of_observations",
)

# ── Name-based markers (matched on the normalized leaf name) ─────────────────
# Coordinate leaf names -- dimensional axes fold into context (geolocation) in
# the four-bucket taxonomy.
_COORDINATE_NAMES = {
    "latitude", "longitude", "lat", "lon", "time",
    "xdim", "ydim", "x", "y",
}
# Quality: uncertainties, flags, precisions, standard deviations, sample counts.
_QUALITY_SUFFIXES = ("uncertainty", "qualityflag", "flag", "precision", "std", "stddev", "error")
# The error-magnitude subset of the quality suffixes: a `<plotted-stem>` +
# one of these is that variable's uncertainty companion (never the flag
# suffixes — the QA flag is its own sibling slot). "uncertainty" first: the
# exact spelling always wins when a product carries several.
_UNCERTAINTY_SUFFIXES = ("uncertainty", "precision", "stddev", "std", "error")
_QUALITY_SUBSTRS = ("uncertainty", "qualityflag", "precision",
                    "numberofpixel", "numberofsample", "numsample", "numberofobs")
# Geometry (context): viewing/solar angles.
_GEOMETRY_SUBSTRS = ("zenithangle", "azimuth")
# Retrieval-metadata: algorithm intermediates. Checked *before* the cloud/
# context markers so AMF-family cloud inputs and below-cloud partials read as
# intermediates, not context.
_RETRIEVAL_PREFIXES = ("amf",)
_RETRIEVAL_SUBSTRS = ("slantcolumn", "belowcloud", "ghost")
# Context: atmosphere/surface bands. Cloud markers are specific (cloud_fraction
# / cloud_pressure / cloud_frac), never bare "cloud", so a cloud-screened
# science column is not swept up here.
_CONTEXT_SUBSTRS = (
    "cloudfraction", "cloudfrac", "cloudpressure", "cloudtop", "cloudheight",
    "cloudalbedo", "pressure", "albedo", "reflectivity", "terrain",
    "aerosolindex", "aerosol", "pbl", "snowice", "temperature",
    # A vertical axis is a coordinate, the same as latitude/longitude/time, so
    # it belongs in context beside them. "pressure" above already caught one
    # spelling of it; without "altitude" the SAME axis in different units
    # (TEMPO_O3PROF publishes both) landed in different buckets, and the
    # related-variables panel offered one half of a toggle.
    "altitude",
)
# Science: recognized geophysical naming stems -- positive evidence, never a
# residual default.
_SCIENCE_STEM_SUBSTRS = (
    "verticalcolumn", "columnamount", "columndensity", "columnnumberdensity",
    "molecontent", "molefraction", "opticaldepth", "opticalthickness",
)
_SCIENCE_STEM_TOKENS = ("aod",)  # short tokens matched as whole-ish fragments


def _norm(value: str | None) -> str:
    """Lowercase and strip every non-alphanumeric character, so CamelCase
    (``SolarZenithAngle``), snake_case (``solar_zenith_angle``) and spaced
    names collapse to one comparable form (``solarzenithangle``)."""
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _leaf(name: str | None) -> str:
    """The bare variable name: the last path segment of a group-qualified
    ``describe_dataset`` name (``product/column_amount_o3`` -> ``column_amount_o3``,
    ``HDFEOS/GRIDS/ColumnAmountO3/Data_Fields/ColumnAmountO3`` -> ``ColumnAmountO3``)."""
    if not name:
        return ""
    return name.rsplit("/", 1)[-1]


def _group_segments(name: str | None, group: str | None) -> set[str]:
    """Every group a variable belongs to, lowercased: the path segments of the
    group-qualified name plus an explicitly supplied ``group`` (itself split
    on ``/`` so a nested ``group_path`` matches by segment, exactly as a
    qualified name does). The leaf name itself is excluded."""
    segments: set[str] = set()
    if name and "/" in name:
        for seg in name.split("/")[:-1]:
            if seg:
                segments.add(seg.lower())
    if group:
        for seg in str(group).split("/"):
            if seg:
                segments.add(seg.lower())
    return segments


def classify_variable(
    name: str,
    *,
    group: str | None = None,
    standard_name: str | None = None,
    long_name: str | None = None,
    units: str | None = None,
) -> tuple[str, Optional[str]]:
    """Classify one variable into (role, confidence) by ordered evidence.

    ``group`` is optional -- when omitted, group membership is read from the
    slash-delimited path in ``name`` (the shape ``describe_dataset`` returns).
    Returns one of the five canonical roles and a confidence tier
    (``high``/``medium``/``low``/``None``); ``unclassified`` always pairs with
    ``None``.
    """
    leaf = _leaf(name)
    norm = _norm(leaf)
    groups = _group_segments(name, group)
    sn = (standard_name or "").strip().lower()

    # 1. Explicit CF standard_name (High) — the most authoritative signal.
    if sn:
        if sn in _COORDINATE_STANDARD_NAMES:
            return ROLE_CONTEXT, CONFIDENCE_HIGH
        if any(s in sn for s in _QUALITY_STANDARD_NAME_SUBSTRS):
            return ROLE_QUALITY, CONFIDENCE_HIGH
        if any(s in sn for s in _SCIENCE_STANDARD_NAME_SUBSTRS):
            return ROLE_SCIENCE, CONFIDENCE_HIGH

    # 2. Strong, terminal group priors (High).
    if groups & _QUALITY_GROUPS:
        return ROLE_QUALITY, CONFIDENCE_HIGH
    if groups & _GEOMETRY_GROUPS:
        return ROLE_CONTEXT, CONFIDENCE_HIGH

    # 3. Coordinate axes (High) — geolocation folds into context.
    if norm in _COORDINATE_NAMES:
        return ROLE_CONTEXT, CONFIDENCE_HIGH

    # 4a. Quality markers (Medium) — uncertainties/flags/precision/std/counts.
    #     Checked before science so an uncertainty *of* a science column reads
    #     as quality even inside a science group.
    if any(s in norm for s in _QUALITY_SUBSTRS) or norm.endswith(_QUALITY_SUFFIXES):
        return ROLE_QUALITY, CONFIDENCE_MEDIUM

    # 4b. Geometry markers (Medium) — solar/viewing angles.
    if any(s in norm for s in _GEOMETRY_SUBSTRS):
        return ROLE_CONTEXT, CONFIDENCE_MEDIUM

    # 4c. Retrieval-metadata markers (Medium) — algorithm intermediates,
    #     before the cloud/context markers.
    if norm.startswith(_RETRIEVAL_PREFIXES) or any(s in norm for s in _RETRIEVAL_SUBSTRS):
        return ROLE_RETRIEVAL_METADATA, CONFIDENCE_MEDIUM

    # 4d. Context markers (Medium) — atmosphere/surface bands.
    if any(s in norm for s in _CONTEXT_SUBSTRS):
        return ROLE_CONTEXT, CONFIDENCE_MEDIUM

    # 5a. Science group membership (High) — positive evidence for an
    #     otherwise-unmarked variable.
    if groups & _SCIENCE_GROUPS:
        return ROLE_SCIENCE, CONFIDENCE_HIGH

    # 5b. Recognized geophysical naming stem (Low) — the last positive-evidence
    #     path to science.
    if any(s in norm for s in _SCIENCE_STEM_SUBSTRS) or any(t in norm for t in _SCIENCE_STEM_TOKENS):
        return ROLE_SCIENCE, CONFIDENCE_LOW

    # 6. No positive evidence for any role.
    return ROLE_UNCLASSIFIED, CONFIDENCE_NONE


def classify_inventory(
    variables: list[dict[str, Any]] | None,
    groups: list[str] | None = None,
    *,
    primary_var: str | None = None,
    quality_flag_var: str | None = None,
) -> list[dict[str, Any]]:
    """Classify a ``describe_dataset`` variable list into role-annotated records.

    ``variables`` is the raw ``describe_dataset`` inventory (each entry a dict
    with ``name``/``long_name``/``units`` and, in the detailed view,
    ``standard_name``). ``groups`` is the collection's ``groups:`` list
    (accepted for completeness; per-variable group is read from the name path).

    ``primary_var`` and ``quality_flag_var`` are the registry's curated leaf
    names for this collection -- the single facts the registry *does* know
    authoritatively. They override the name-based classification for that exact
    variable (science / quality, High), which is what lets a product's primary
    science variable classify as science even when ``describe_dataset``'s
    UMM-Var view omits its CF ``standard_name`` (e.g. TROPOMI ``Tropospheric_NO2``
    in the compact, ``detail=False`` view the discovery pane uses).

    Returns one record per variable: ``{name, leaf, group, role, confidence,
    long_name, units}``, preserving input order.
    """
    primary_leaf = _norm(_leaf(primary_var)) if primary_var else None
    quality_leaf = _norm(_leaf(quality_flag_var)) if quality_flag_var else None

    out: list[dict[str, Any]] = []
    for var in variables or []:
        if not isinstance(var, dict):
            continue
        name = var.get("name")
        leaf = _leaf(name)
        norm_leaf = _norm(leaf)
        # An explicit per-record group (a post-open inventory whose names are
        # bare leaves — open_handle stamps ``group_path``) reaches the same
        # group priors a slash-qualified describe_dataset name does.
        explicit_group = var.get("group")
        role, confidence = classify_variable(
            name,
            group=explicit_group,
            standard_name=var.get("standard_name"),
            long_name=var.get("long_name"),
            units=var.get("units"),
        )

        # Registry curated overrides — the collection's own ground truth.
        if primary_leaf and norm_leaf == primary_leaf:
            role, confidence = ROLE_SCIENCE, CONFIDENCE_HIGH
        elif quality_leaf and norm_leaf == quality_leaf:
            role, confidence = ROLE_QUALITY, CONFIDENCE_HIGH

        out.append({
            "name": name,
            "leaf": leaf,
            "group": explicit_group or ("/".join(name.split("/")[:-1]) if name and "/" in name else None),
            "role": role,
            "confidence": confidence,
            "long_name": var.get("long_name"),
            "units": var.get("units"),
        })
    return out


def related_variables(
    variables: list[Any] | None,
    groups: list[str] | None = None,
    *,
    primary_var: str | None = None,
    quality_flag_var: str | None = None,
    plotted_variable: str | None = None,
) -> dict[str, Any]:
    """A lightweight related-variables view for the chart page (PRD T35): the
    plotted variable's role plus its companion siblings, matched cheaply from
    the classified inventory — links, not a re-render of the whole inventory
    (the thin edge of T36).

    ``variables`` may be either ``describe_dataset`` records or bare registry
    variable strings; the chart path passes the registry's curated subset (the
    variables this app actually retrieves for the collection) so no extra MCP
    round trip is needed. Siblings:

    - ``qa_sibling``          -- the registry ``quality_flag_var`` leaf
    - ``uncertainty_sibling`` -- the plotted stem's error companion:
      ``<stem>_uncertainty`` preferred, else ``<stem>`` + precision/std/
      stddev/error (OMI's ``ColumnAmountO3Precision`` spelling)
    - ``context_siblings``    -- the inventory's context-role variable leaves

    Returns an empty-sibling structure (never invented companions) when a
    product carries none — e.g. MODIS AOD.
    """
    records = [
        v if isinstance(v, dict) else {"name": v}
        for v in (variables or [])
        if v
    ]
    classified = classify_inventory(
        records, groups=groups, primary_var=primary_var, quality_flag_var=quality_flag_var,
    )
    plotted_norm = _norm(_leaf(plotted_variable)) if plotted_variable else None

    # The plotted variable's own role — classified directly (with the registry
    # hints) so it is known even when the curated subset doesn't list it, e.g.
    # MODIS AOD whose registry `variables` is empty.
    role, confidence = ROLE_UNCLASSIFIED, CONFIDENCE_NONE
    if plotted_variable:
        plotted = classify_inventory(
            [{"name": plotted_variable}], groups=groups,
            primary_var=primary_var, quality_flag_var=quality_flag_var,
        )
        role, confidence = plotted[0]["role"], plotted[0]["confidence"]

    context_siblings: list[str] = []
    uncertainty_sibling: Optional[str] = None
    for entry in classified:
        entry_norm = _norm(entry["leaf"])
        if plotted_norm is not None and entry_norm == plotted_norm:
            continue  # the plotted variable is not its own sibling
        if entry["role"] == ROLE_CONTEXT:
            context_siblings.append(entry["leaf"])
        if plotted_norm is not None and entry_norm.startswith(plotted_norm):
            suffix = entry_norm[len(plotted_norm):]
            if suffix == "uncertainty":
                uncertainty_sibling = entry["leaf"]  # exact spelling always wins
            elif suffix in _UNCERTAINTY_SUFFIXES and uncertainty_sibling is None:
                uncertainty_sibling = entry["leaf"]

    qa_leaf = _leaf(quality_flag_var) if quality_flag_var else None
    return {
        "variable": _leaf(plotted_variable) if plotted_variable else None,
        "role": role,
        "confidence": confidence,
        "qa_sibling": qa_leaf,
        "uncertainty_sibling": uncertainty_sibling,
        "context_siblings": context_siblings,
    }


# Ordered role list for grouped display (science first, unclassified last).
ROLE_DISPLAY_ORDER = (
    ROLE_SCIENCE, ROLE_QUALITY, ROLE_CONTEXT, ROLE_RETRIEVAL_METADATA, ROLE_UNCLASSIFIED,
)
