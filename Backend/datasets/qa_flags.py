"""
datasets/qa_flags.py
======================
T25 Phase 3: the three-tier QA-flag masking doctrine (decision 2026-07-09/10),
kept beside mask_info.py (which owns fill_value/valid_min/valid_max) because
QA is a distinct fact but the same doctrine applies -- this is the one module
where the good-token vocabulary and the ambiguity boundary live, so neither
is scattered across call sites.

Tiers:
  1. A pinned rule in collections.yaml (``qa_good_values``/``qa_bad_values``
     on the registry entry) -> apply, recorded QA_VERIFIED.
  2. The flag variable's own CF ``flag_values``/``flag_meanings`` -> parsed
     deterministically. Every token classifiable via GOOD_TOKENS/BAD_TOKENS
     -> apply with no model involved, recorded QA_CF_DETERMINISTIC. Any token
     outside that vocabulary is ambiguous: applying a mask needs the agent's
     proposal (``proposed_good_tokens``) -- when supplied, recorded
     QA_INFERRED and logged so a human can find promotion candidates; when
     not (yet) supplied, recorded QA_AMBIGUOUS_PENDING rather than guessing.
  3. Neither a pinned rule nor CF flag_values/flag_meanings -> no mask,
     recorded QA_NOT_APPLIED.

Promotion (inferred -> pinned) is a manual collections.yaml edit (PRD:
"do not build the promotion tooling") -- this module only makes inferred
records greppable via the ``qa_flags_inferred_mask`` log event.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

QA_VERIFIED = "verified"
QA_CF_DETERMINISTIC = "cf-deterministic"
QA_INFERRED = "inferred, not verified"
QA_AMBIGUOUS_PENDING = "ambiguous — awaiting classification"
QA_NOT_APPLIED = "not applied — semantics unknown"

# Tokens whose plain-English meaning is unambiguous good/bad quality --
# matched case-insensitively against each flag_meanings token (spaces and
# underscores interchangeable). Anything not listed here is ambiguous: a
# judgment call this module never makes unilaterally.
GOOD_TOKENS: frozenset[str] = frozenset({
    "good", "good_quality", "high_quality", "best", "best_quality",
    "normal", "valid", "clear", "confident_quality", "high_confidence",
})
BAD_TOKENS: frozenset[str] = frozenset({
    "bad", "bad_quality", "poor_quality", "low_quality", "missing",
    "no_data", "nodata", "fill", "fill_value", "invalid", "cloudy",
    "not_confident_quality", "error", "failed_quality",
})


@dataclass(frozen=True)
class FlagMeaningsParse:
    """Result of parsing a variable's CF ``flag_values``/``flag_meanings``."""
    available: bool
    good_values: list[int] = field(default_factory=list)
    bad_values: list[int] = field(default_factory=list)
    ambiguous_tokens: list[str] = field(default_factory=list)
    ambiguous_values: list[int] = field(default_factory=list)

    @property
    def unambiguous(self) -> bool:
        return self.available and not self.ambiguous_tokens


def _normalize_token(token: str) -> str:
    return str(token).strip().strip('"').strip("'").lower().replace(" ", "_")


def _coerce_int_sequence(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, str):
        parts: Any = values.replace(",", " ").split()
    elif hasattr(values, "tolist"):
        parts = values.tolist()
    else:
        try:
            parts = list(values)
        except TypeError:
            return []
    try:
        return [int(float(v)) for v in parts]
    except (TypeError, ValueError):
        return []


def _coerce_token_sequence(meanings: Any) -> list[str]:
    if meanings is None:
        return []
    if isinstance(meanings, str):
        return meanings.split()
    try:
        return [str(t) for t in meanings]
    except TypeError:
        return []


def parse_flag_meanings(flag_values: Any, flag_meanings: Any) -> FlagMeaningsParse:
    """Parse CF ``flag_values``/``flag_meanings`` attrs (positional pairing,
    per the CF conventions) into good/bad/ambiguous buckets. Accepts the
    shapes NetCDF/Zarr attr readers actually hand back: a numpy array or list
    of ints for ``flag_values``, a space-separated string or list of tokens
    for ``flag_meanings``. Malformed or mismatched-length input is treated as
    "not available", never partially trusted.
    """
    values = _coerce_int_sequence(flag_values)
    tokens = _coerce_token_sequence(flag_meanings)
    if not values or not tokens or len(values) != len(tokens):
        return FlagMeaningsParse(available=False)

    good, bad, ambiguous_tokens, ambiguous_values = [], [], [], []
    for value, token in zip(values, tokens):
        norm = _normalize_token(token)
        if norm in GOOD_TOKENS:
            good.append(value)
        elif norm in BAD_TOKENS:
            bad.append(value)
        else:
            ambiguous_tokens.append(token)
            ambiguous_values.append(value)

    return FlagMeaningsParse(
        available=True,
        good_values=good,
        bad_values=bad,
        ambiguous_tokens=ambiguous_tokens,
        ambiguous_values=ambiguous_values,
    )


def resolve_qa_info(
    yaml_info: dict[str, Any] | None = None,
    flag_attrs: dict[str, Any] | None = None,
    *,
    proposed_good_tokens: list[str] | None = None,
    short_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve which flag values count as "good" per the three-tier doctrine.

    Returns ``(qa_col_info, qa_provenance)``: ``qa_col_info`` carries whatever
    ``qa_good_values``/``qa_bad_values`` AggregationService.apply_quality_mask
    needs to actually filter; ``qa_provenance`` is what a caller merges into
    ``result.meta["masking"]`` so the tier that decided is never left
    unstated (``qa_status``/``qa_source`` plus tier-specific detail).
    """
    yaml_info = yaml_info or {}
    flag_attrs = flag_attrs or {}

    pinned_good = yaml_info.get("qa_good_values")
    pinned_bad = yaml_info.get("qa_bad_values")
    # An explicitly empty good-set would key a keep-mask on ``isin([])`` --
    # an all-False mask wiping the whole variable to NaN; an empty bad-set
    # masks nothing yet would still report "verified". Neither is a usable
    # rule, so drop it and fall through rather than mask everything (or
    # nothing) while claiming a pinned rule ran.
    if pinned_good is not None and len(list(pinned_good)) == 0:
        pinned_good = None
    if pinned_bad is not None and len(list(pinned_bad)) == 0:
        pinned_bad = None
    if pinned_good is not None or pinned_bad is not None:
        qa_col_info: dict[str, Any] = {}
        if pinned_good is not None:
            qa_col_info["qa_good_values"] = list(pinned_good)
        else:
            qa_col_info["qa_bad_values"] = list(pinned_bad)
        return qa_col_info, {"qa_status": QA_VERIFIED, "qa_source": "collections_yaml"}

    parsed = parse_flag_meanings(flag_attrs.get("flag_values"), flag_attrs.get("flag_meanings"))
    if not parsed.available:
        return {}, {"qa_status": QA_NOT_APPLIED, "qa_source": "none"}

    if parsed.unambiguous:
        if not parsed.good_values:
            # Every token classified as bad-quality (e.g. "missing cloudy"):
            # there is no positive good class to key a keep-mask on, and
            # ``isin([])`` would wipe the whole variable to NaN. This is not a
            # usable classification -- go ambiguous (apply no mask) rather than
            # mask everything while reporting "cf-deterministic".
            return (
                {},
                {
                    "qa_status": QA_AMBIGUOUS_PENDING,
                    "qa_source": "cf_flag_meanings",
                    "qa_ambiguous_tokens": [],
                    "qa_bad_values": parsed.bad_values,
                    "qa_note": (
                        "every flag_meanings token classifies as bad-quality; "
                        "no good class to key a mask on -- no mask applied"
                    ),
                },
            )
        return (
            {"qa_good_values": parsed.good_values},
            {
                "qa_status": QA_CF_DETERMINISTIC,
                "qa_source": "cf_flag_meanings",
                "qa_good_values": parsed.good_values,
                "qa_bad_values": parsed.bad_values,
            },
        )

    if proposed_good_tokens:
        normalized_proposed = {_normalize_token(t) for t in proposed_good_tokens}
        inferred_tokens = [t for t in parsed.ambiguous_tokens if _normalize_token(t) in normalized_proposed]
        inferred_values = [
            v for v, t in zip(parsed.ambiguous_values, parsed.ambiguous_tokens)
            if _normalize_token(t) in normalized_proposed
        ]
        good_values = list(parsed.good_values) + inferred_values
        logger.info(
            "qa_flags_inferred_mask",
            extra={
                "_event": "qa_flags_inferred_mask",
                "_short_name": short_name,
                "_ambiguous_tokens": parsed.ambiguous_tokens,
                "_inferred_tokens": inferred_tokens,
            },
        )
        return (
            {"qa_good_values": good_values},
            {
                "qa_status": QA_INFERRED,
                "qa_source": "cf_flag_meanings",
                "qa_good_values": good_values,
                "qa_ambiguous_tokens": parsed.ambiguous_tokens,
                "qa_inferred_tokens": inferred_tokens,
            },
        )

    return (
        {},
        {
            "qa_status": QA_AMBIGUOUS_PENDING,
            "qa_source": "cf_flag_meanings",
            "qa_ambiguous_tokens": parsed.ambiguous_tokens,
        },
    )
