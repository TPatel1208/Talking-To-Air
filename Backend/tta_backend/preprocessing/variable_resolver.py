"""
preprocessing/variable_resolver.py
==================================
PRD T48. A pure, I/O-free resolver that chooses ONE science variable out of a
wide, group-structured, unregistered L3 product -- the tail of
``AggregationService.to_dataarray``'s cascade, reached only after the exact
intent/curation tiers (explicit ``variable=``, retrieval-recorded choice,
single-var, pinned ``primary_var``) have all missed.

The origin case (bug diagnosis 2026-07-19): ``AERDA_D3_VIIRS_MODIS`` is a
Yori-aggregated L3 where ~120 groups share the same four leaves
(``Mean``/``Standard_Deviation``/``Pixel_Counts``/``Histogram_Counts``), which
``open_handle``'s collision-rename qualifies into 432 data variables. There is
no ``AOD`` variable to match, no registry default, and 27 of the 49 AOD
``.../Mean`` groups are globally empty in a given granule -- so even a lucky
guess frequently plots blank. Hard filters can't choose here; a scoring model
that combines weak signals (metadata, naming, data completeness, domain
priors) can.

The pipeline is explicit: **discover -> classify -> score -> drop-empty ->
rank -> decide**. It returns a ranked, bounded candidate set plus two
independent axes (resolution confidence x scientific ambiguity) and a
disclosure string, so the caller can auto-pick silently, auto-pick with a
disclosed choice, or refuse -- never a silent wrong pick.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import xarray as xr

from tta_backend.config.error_templates import render_variable_note

# Leaves that are never a science-variable candidate unless the user asks for
# one by name: the statistical moments and pixel bookkeeping a Yori-aggregated
# L3 stamps beside every geophysical Mean. This is the plumbing, not the
# science (PRD three-category taxonomy, category 1).
_IMPLEMENTATION_LEAVES = frozenset({
    "Standard_Deviation",
    "Pixel_Counts",
    "Histogram_Counts",
})

# Name substrings that mark the plumbing categories the PRD taxonomy
# enumerates when no metadata signal is present: viewing/solar geometry,
# reflectances, land/ocean masks, vegetation index, cloud fraction. Matched
# case-insensitively against the full (group-qualified) name. Deliberately a
# fallback -- metadata (standard_name/flag_values) is consulted first.
_IMPLEMENTATION_NAME_PATTERNS = (
    re.compile(r"zenith_angle", re.IGNORECASE),
    re.compile(r"azimuth_angle", re.IGNORECASE),
    re.compile(r"scattering_angle", re.IGNORECASE),
    re.compile(r"reflectance", re.IGNORECASE),
    re.compile(r"land_ocean_mask", re.IGNORECASE),
    re.compile(r"(^|[/_])mask($|[/_])", re.IGNORECASE),
    re.compile(r"(^|[/_])ndvi($|[/_])", re.IGNORECASE),
    re.compile(r"cloud_fraction", re.IGNORECASE),
    # A QA flag with no CF flag_values/flag_meanings attrs (so _is_cf_flag
    # missed it) is still recognizable by name -- the same signal
    # AggregationService._science_vars keys off the registry for.
    re.compile(r"quality_flag", re.IGNORECASE),
    re.compile(r"data_quality", re.IGNORECASE),
)

# Category labels carried on each Candidate (PRD taxonomy).
CATEGORY_IMPLEMENTATION = "implementation"
CATEGORY_EQUIVALENT = "equivalent"
CATEGORY_DISTINCT = "distinct"

# Scoring weights (PRD tunable table -- a documented starting point, calibrated
# against the AERDA granule + existing fixtures, NOT a quality claim). Each is
# a weak signal; the sum ranks populated science candidates. valid_fraction
# contributes separately (a 0%-valid candidate is dropped, not just penalized).
_W_MEAN_LEAF = 100.0
_W_HAS_UNITS = 80.0
_W_GEOPHYSICAL = 40.0
_W_STANDARD_NAME = 30.0
_W_LONG_NAME = 20.0
_W_IMPLEMENTATION = -60.0

# A "strong signal" is what lets the resolver CONFIDENTLY auto-pick among
# contenders (conservative doctrine, keeping T25 where it is load-bearing): a
# candidate carries one when it has units, is an aggregated Mean field, or
# declares a CF standard_name. A candidate distinguished ONLY by a weak
# "name looks geophysical" heuristic has no strong signal -- so multiple such
# contenders refuse-and-ask rather than let the resolver invent a scientific
# choice. These reason strings are produced by ``_score``.
_STRONG_REASONS = frozenset({"has units", "aggregated mean field", "CF standard_name"})
# Cap on alternatives named in a disclosure -- keeps the note (and its re-feed
# cost to the model) bounded no matter how many distinct products a file has.
_MAX_DISCLOSED_ALTERNATIVES = 4

# Sensor/platform and algorithm ORDERING tables. IMPORTANT: these are a
# documented, version-controlled tiebreak to make a within-band tie
# deterministic and to prefer a sensible default -- they are NOT a quality
# claim. Terra-vs-Aqua and Deep-Blue-vs-Dark-Target are context-dependent
# scientific choices; the disclosure always names the pick AND its
# alternatives so a researcher can redirect. Lower rank = preferred default;
# an unlisted token gets _DEFAULT_RANK, so the table is extensible without
# touching resolver logic. Data availability (coverage band) always ranks
# ABOVE these tables -- the resolver never chooses an emptier field just
# because a table prefers its sensor.
_DEFAULT_RANK = 99
_SENSOR_RANK = (
    ("terra", 0),
    ("aqua", 1),
    ("snpp", 2),
    ("noaa20", 3),
    ("noaa-20", 3),
    ("viirs", 4),
)
_ALGORITHM_RANK = (
    ("darktarget", 0),
    ("dark_target", 0),
    ("deepblue", 1),
    ("deep_blue", 1),
)

# Roots that make a name/long_name/standard_name read as a geophysical
# measurement rather than plumbing. Substring match, case-insensitive.
_GEOPHYSICAL_ROOTS = (
    "aod", "aerosol", "optical_depth", "angstrom",
    "no2", "o3", "ozone", "so2", "hcho", "formaldehyde", "ch4", "methane",
    "carbon_monoxide", "column", "concentration", "mixing_ratio",
    "temperature", "precipitation", "radiance", "brightness",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    category: str
    score: float = 0.0
    valid_fraction: float | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Resolution:
    name: str | None
    candidates: list[Candidate]
    resolution_confidence: str
    scientific_ambiguity: str
    disclosure: str | None = None
    # T49: how many data variables were dropped for being entirely empty over
    # the requested range (resolve() drops them before scoring). The picker
    # (build_variable_choice) discloses this count so a researcher understands
    # the candidate list is the populated subset, not the whole file. Additive
    # telemetry of a decision resolve() already makes -- it does not change the
    # scoring/ranking/classification T48 owns.
    excluded_empty_count: int = 0


def _leaf(name: str) -> str:
    """The bare leaf of a group-qualified variable name (open_handle merges HDF
    groups to ``group/leaf`` and collision-renames on clashes)."""
    return name.rsplit("/", 1)[-1]


def _is_cf_flag(attrs: dict) -> bool:
    """A CF quality flag declares both ``flag_values`` and ``flag_meanings`` --
    the machine-readable QA signal, the same one apply_quality_mask keys off."""
    return "flag_values" in attrs and "flag_meanings" in attrs


def _classify(name: str, var: xr.DataArray) -> str:
    """The three-category taxonomy for one variable, metadata before name
    heuristics.

    Category 1 (implementation) is assigned from, in order: the CF flag
    metadata signal (flag_values + flag_meanings); the statistical-moment /
    pixel-bookkeeping leaves a Yori L3 stamps beside every field; then the
    name-heuristic plumbing patterns (geometry, reflectance, mask, NDVI, cloud
    fraction). Everything else is, for now, a distinct product."""
    attrs = dict(getattr(var, "attrs", None) or {})
    if _is_cf_flag(attrs):
        return CATEGORY_IMPLEMENTATION
    if _leaf(name) in _IMPLEMENTATION_LEAVES:
        return CATEGORY_IMPLEMENTATION
    if any(p.search(name) for p in _IMPLEMENTATION_NAME_PATTERNS):
        return CATEGORY_IMPLEMENTATION
    return CATEGORY_DISTINCT


def _matches_request(name: str, requested: str | None) -> bool:
    """Whether ``name`` answers an explicit ``requested`` string -- by exact
    name, by shared bare leaf, or by carrying the request as a token (so a
    group-qualified ``aod_550/Standard_Deviation`` answers a bare
    ``Standard_Deviation`` the earlier exact-match tier could not, because
    collision-renaming qualified the leaf away)."""
    if not requested:
        return False
    req = requested.strip().casefold()
    if not req:
        return False
    n = name.casefold()
    return req == n or req == _leaf(name).casefold() or _leaf(req) == _leaf(name).casefold() or req in n


def _is_geophysical(name: str, attrs: dict) -> bool:
    """Whether a name / CF label reads as a geophysical measurement (one of
    the documented roots), not plumbing."""
    haystack = " ".join(str(x) for x in (
        name, attrs.get("standard_name", ""), attrs.get("long_name", ""),
    )).casefold()
    return any(root in haystack for root in _GEOPHYSICAL_ROOTS)


def _score(name: str, var: xr.DataArray, category: str) -> tuple[float, list[str]]:
    """Combine the weak signals into one score plus the human reasons behind
    it (for disclosure/logging). A pure function of name + CF attrs; data
    completeness (valid_fraction) is folded in separately by the caller."""
    attrs = dict(getattr(var, "attrs", None) or {})
    score = 0.0
    reasons: list[str] = []
    if _leaf(name) == "Mean":
        score += _W_MEAN_LEAF
        reasons.append("aggregated mean field")
    if attrs.get("units") not in (None, ""):
        score += _W_HAS_UNITS
        reasons.append("has units")
    if _is_geophysical(name, attrs):
        score += _W_GEOPHYSICAL
        reasons.append("geophysical quantity")
    if attrs.get("standard_name"):
        score += _W_STANDARD_NAME
        reasons.append("CF standard_name")
    if attrs.get("long_name"):
        score += _W_LONG_NAME
        reasons.append("descriptive long_name")
    if category == CATEGORY_IMPLEMENTATION:
        score += _W_IMPLEMENTATION
    return score, reasons


def _table_rank(name: str, table: tuple[tuple[str, int], ...]) -> int:
    """The best (lowest) ordering rank any token in ``table`` earns by
    appearing in ``name``, or _DEFAULT_RANK when none does. A deterministic
    tiebreak, never a quality signal (see _SENSOR_RANK/_ALGORITHM_RANK)."""
    n = name.casefold().replace(" ", "")
    ranks = [rank for token, rank in table if token in n]
    return min(ranks) if ranks else _DEFAULT_RANK


def _valid_band(valid_fraction: float | None) -> int:
    """The coverage decile a valid_fraction falls in (0..10). Ranking on the
    QUANTIZED band, not raw fraction, is the reproducibility guard (PRD): only
    a materially larger coverage difference reorders sensors, so day-to-day
    valid-pixel jitter within a band can't flip the chosen product. Within a
    band the stable score/sensor tables decide, giving the same answer across
    dates whenever coverage is comparable."""
    if not valid_fraction or valid_fraction <= 0.0:
        return 0
    return min(10, int(valid_fraction * 10))


def _valid_fraction(var: xr.DataArray) -> float:
    """Fraction of finite (non-fill, non-NaN) cells in ``var``, computed
    lazily via ``.notnull().mean()`` -- a single dask reduction, never a full
    ``.load()`` of the field. Cost is bounded because the caller computes this
    only for post-classification science candidates (never a 3-D
    Histogram_Counts ancillary, which classification already excluded).
    An unreadable field counts as empty (0.0) rather than crashing resolution."""
    try:
        return float(var.notnull().mean())
    except Exception:
        return 0.0


def _is_pickable(c: Candidate) -> bool:
    """Whether a candidate has real coverage worth plotting -- above the empty
    floor (band 0 = <10% valid is treated as too sparse to auto-pick). A
    near-empty field is kept in the ranked list but is never the silent pick."""
    return _valid_band(c.valid_fraction) > 0


def _is_strong(c: Candidate) -> bool:
    """Whether a candidate carries a strong disambiguating signal (units, an
    aggregated Mean leaf, or a CF standard_name) -- the bar for a confident
    auto-pick among contenders. See _STRONG_REASONS."""
    return bool(_STRONG_REASONS.intersection(c.reasons))


def _label(name: str, var: xr.DataArray) -> str:
    """A human label for a variable in a disclosure -- its CF ``long_name``
    when present (the product's own description), else the group-qualified
    name itself. Never invented text."""
    long_name = str((getattr(var, "attrs", None) or {}).get("long_name") or "").strip()
    return long_name or name


def resolve(ds: xr.Dataset, *, requested: str | None = None, handle: str | None = None) -> Resolution:
    """Choose one science variable out of ``ds``. Pure and I/O-free."""
    candidates: list[Candidate] = []
    excluded_empty = 0
    for name in ds.data_vars:
        name = str(name)
        var = ds[name]
        category = _classify(name, var)
        requested_here = _matches_request(name, requested)
        # Category-1 plumbing is excluded from the default choice, but an
        # explicit request un-hides the matching one (PRD: reachable by name).
        if category == CATEGORY_IMPLEMENTATION and not requested_here:
            continue
        # Data availability outranks every product-preference signal: an
        # all-empty field is dropped outright, so a lucky guess can never plot
        # blank. Explicitly requested vars are kept even if empty -- the user
        # named it, so an honest "no valid data" beats silently dropping it.
        valid_fraction = _valid_fraction(var)
        if valid_fraction <= 0.0 and not requested_here:
            # A science candidate (not plumbing) dropped only for having no
            # data over this range -- counted so the picker can disclose it.
            excluded_empty += 1
            continue
        score, reasons = _score(name, var, category)
        if requested_here:
            reasons = ["explicitly requested", *reasons]
        candidates.append(Candidate(
            name=name, category=category, score=score,
            valid_fraction=valid_fraction, reasons=reasons,
        ))

    # Rank populated candidates best-first on the documented composite tuple:
    # coverage band (data availability wins first among populated fields),
    # then science score, then the stable sensor/algorithm ordering tables,
    # then canonical name as the final deterministic tiebreak.
    candidates.sort(key=lambda c: (
        -_valid_band(c.valid_fraction),
        -c.score,
        _table_rank(c.name, _SENSOR_RANK),
        _table_rank(c.name, _ALGORITHM_RANK),
        c.name,
    ))

    # An explicitly requested variable is intent, not a guess -- it wins ahead
    # of all scoring, at high confidence, with no disclosure (the user already
    # made the choice).
    requested_hits = [c for c in candidates if "explicitly requested" in c.reasons]
    if requested_hits:
        return Resolution(
            name=requested_hits[0].name,
            candidates=candidates,
            resolution_confidence="high",
            scientific_ambiguity="low",
            excluded_empty_count=excluded_empty,
        )

    # Decide over the candidates that actually have coverage worth plotting.
    pickable = [c for c in candidates if _is_pickable(c)]
    top = pickable[0] if pickable else None
    strong = [c for c in pickable if _is_strong(c)]

    # The conservative decision matrix (resolution confidence x scientific
    # ambiguity):
    #   - no pickable field at all -> low confidence, refuse.
    #   - top-ranked carries a strong signal -> high confidence, auto-pick;
    #     ambiguity is high when 2+ strong, scientifically-distinct products
    #     compete in the top coverage band (a genuine sensor/algorithm fork the
    #     researcher must see), otherwise low (silent).
    #   - exactly one pickable candidate, only weakly signalled -> medium
    #     confidence, auto-pick with a brief note (no contention to refuse over).
    #   - multiple weak, name-only contenders and none strong -> low confidence,
    #     refuse and ask (never invent a scientific choice, T25).
    if top is None:
        return Resolution(
            name=None, candidates=candidates,
            resolution_confidence="low", scientific_ambiguity="low",
            excluded_empty_count=excluded_empty,
        )

    if _is_strong(top):
        confidence = "high"
        top_band = _valid_band(top.valid_fraction)
        serious_distinct = [
            c for c in strong
            if c.category == CATEGORY_DISTINCT and _valid_band(c.valid_fraction) == top_band
        ]
        ambiguity = "high" if len(serious_distinct) >= 2 else "low"
        alternatives = serious_distinct
    elif len(pickable) == 1:
        confidence, ambiguity = "medium", "low"
        alternatives = []
    else:
        return Resolution(
            name=None, candidates=candidates,
            resolution_confidence="low", scientific_ambiguity="low",
            excluded_empty_count=excluded_empty,
        )

    disclosure = None
    if ambiguity == "high" or confidence == "medium":
        chosen_label = _label(top.name, ds[top.name])
        alt_labels = [
            _label(c.name, ds[c.name]) for c in alternatives if c.name != top.name
        ][:_MAX_DISCLOSED_ALTERNATIVES]
        disclosure = render_variable_note(
            chosen_label, alt_labels, ambiguous=(ambiguity == "high"),
        )

    return Resolution(
        name=top.name,
        candidates=candidates,
        resolution_confidence=confidence,
        scientific_ambiguity=ambiguity,
        disclosure=disclosure,
        excluded_empty_count=excluded_empty,
    )
