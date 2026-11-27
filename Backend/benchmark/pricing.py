"""Estimated model cost from the token counts in an Outcome.

Prices are Anthropic's first-party list prices in US dollars per million
tokens. The configured models run through Portkey on Google Vertex, which
prices Claude separately, so these are estimates, not the bill. Edit the table
when prices or models change.

Cache reads and cache writes are both part of the input count (langchain's
``input_tokens`` is the sum of every input kind), and each is charged at its
own price instead of the input price. The cache-write rate is the 5-minute
write: 1.25x the input price, Anthropic's published ratio. Cache reads follow
the table below.
"""
from __future__ import annotations

from dataclasses import dataclass

from benchmark.outcome import Outcome


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    # The 5-minute cache write, 1.25x input.
    cache_creation: float


# Matched as a substring of the recorded model id, first match wins, so a
# longer name must come before any name it contains.
PRICES_PER_MILLION: tuple[tuple[str, Price], ...] = (
    ("claude-sonnet-5-5", Price(input=2.00, output=10.00, cache_read=0.10, cache_creation=2.50)),
    ("claude-sonnet-5", Price(input=2.00, output=10.00, cache_read=0.20, cache_creation=2.50)),
    ("claude-haiku-4-5", Price(input=1.00, output=5.00, cache_read=0.10, cache_creation=1.25)),
)


def price_for(model: str) -> Price | None:
    for name, price in PRICES_PER_MILLION:
        if name in model:
            return price
    return None


@dataclass(frozen=True)
class CostEstimate:
    dollars: float
    # Models that used tokens but have no price, so their share is missing.
    unpriced: tuple[str, ...] = ()
    # Models whose cache reads and writes add up to more than their input, so
    # the counts disagree and the cost is understated.
    inconsistent_models: tuple[str, ...] = ()


def estimate_cost(outcome: Outcome) -> CostEstimate | None:
    """The estimated cost of every model call in ``outcome``, or None if no tokens were reported."""
    if not outcome.tokens:
        return None
    by_model: dict[str, dict[str, int]] = {}
    for (model, _agent, kind), count in outcome.tokens.items():
        kinds = by_model.setdefault(model, {})
        kinds[kind] = kinds.get(kind, 0) + count
    dollars = 0.0
    unpriced = []
    inconsistent = []
    for model, kinds in sorted(by_model.items()):
        price = price_for(model)
        if price is None:
            unpriced.append(model)
            continue
        cache_read = kinds.get("cache_read", 0)
        cache_creation = kinds.get("cache_creation", 0)
        if cache_read + cache_creation > kinds.get("input", 0):
            inconsistent.append(model)
        uncached_input = max(kinds.get("input", 0) - cache_read - cache_creation, 0)
        dollars += (
            uncached_input * price.input
            + cache_read * price.cache_read
            + cache_creation * price.cache_creation
            + kinds.get("output", 0) * price.output
        ) / 1_000_000
    return CostEstimate(dollars=dollars, unpriced=tuple(unpriced), inconsistent_models=tuple(inconsistent))
