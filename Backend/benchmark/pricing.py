"""Estimated model cost from the token counts in an Outcome.

Prices are Anthropic's first-party list prices in US dollars per million
tokens. The configured models run through Portkey on Google Vertex, which
prices Claude separately, so these are estimates, not the bill. Edit the table
when prices or models change.

Cache reads are part of the input count and are charged at the cache-read
price instead of the input price. Cache writes are not counted separately, so
they are charged at the plain input price, which understates them slightly.
"""
from __future__ import annotations

from dataclasses import dataclass

from benchmark.outcome import Outcome


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float


# Matched as a substring of the recorded model id, first match wins, so a
# longer name must come before any name it contains.
PRICES_PER_MILLION: tuple[tuple[str, Price], ...] = (
    ("claude-sonnet-5-5", Price(input=2.00, output=10.00, cache_read=0.10)),
    ("claude-sonnet-5", Price(input=2.00, output=10.00, cache_read=0.20)),
    ("claude-haiku-4-5", Price(input=1.00, output=5.00, cache_read=0.10)),
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
    for model, kinds in sorted(by_model.items()):
        price = price_for(model)
        if price is None:
            unpriced.append(model)
            continue
        cache_read = kinds.get("cache_read", 0)
        uncached_input = max(kinds.get("input", 0) - cache_read, 0)
        dollars += (
            uncached_input * price.input
            + cache_read * price.cache_read
            + kinds.get("output", 0) * price.output
        ) / 1_000_000
    return CostEstimate(dollars=dollars, unpriced=tuple(unpriced))
