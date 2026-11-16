"""Scoring one outcome against its case, and the thresholds a run must meet."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from benchmark.cases import BenchmarkCase
from benchmark.checks import CheckResult
from benchmark.outcome import Outcome
from benchmark.pricing import CostEstimate, estimate_cost

# Passing needs this many of TOTAL_CASES. Provisional until a live run confirms it.
PASS_THRESHOLD = 17
TOTAL_CASES = 20

# Per-category wall-clock budgets in seconds. They apply against the fake data
# service, so they measure the system's own overhead, not NASA's. A case over
# its budget fails even when every check passes.
#
# Measured on the current system entered through the full chat stream. Where
# a category ran over, its budget is the second-slowest time seen across two
# full runs plus 20%, rounded up to 5s, so one wandering run still fails.
CATEGORY_BUDGETS: dict[str, float] = {
    "discovery": 40.0,
    "retrieval": 45.0,
    "plotting": 45.0,
    "comparison_setup": 55.0,
    "failure_recovery": 20.0,
    "ground_validation": 40.0,
    "comparison": 45.0,
    "e2e_ground": 15.0,
    "e2e_satellite": 45.0,
    "e2e_cross_source": 90.0,
    "e2e_routing": 50.0,
    "region_refusal": 45.0,
}


@dataclass
class CaseResult:
    case: BenchmarkCase
    outcome: Outcome
    check_results: list[CheckResult] = field(default_factory=list)
    within_budget: bool = True
    passed: bool = False
    # Which checks scored it: "harness" when the case's harness_checks did, else "default".
    check_set: str = "default"

    @property
    def first_failure(self) -> CheckResult | None:
        return next((result for result in self.check_results if not result.passed), None)


def score(case: BenchmarkCase, outcome: Outcome, *, harness: bool = False) -> CaseResult:
    """Score ``outcome``; a harness adapter's run uses the case's ``harness_checks`` when it has them."""
    use_harness = harness and case.harness_checks is not None
    checks = case.harness_checks if use_harness else case.checks
    check_results = [check.evaluate(outcome) for check in checks]
    within_budget = outcome.elapsed_seconds <= CATEGORY_BUDGETS[case.category]
    return CaseResult(
        case=case,
        outcome=outcome,
        check_results=check_results,
        within_budget=within_budget,
        passed=all(result.passed for result in check_results) and within_budget,
        check_set="harness" if use_harness else "default",
    )


# Token columns: input, output, and the share of input read from the provider's cache.
_TOKEN_COLUMNS = (("input", "in tok"), ("output", "out tok"), ("cache_read", "cached"))


def _token_cell(count: int | None) -> str:
    return "-" if count is None else f"{count:,}"


def _cost_cell(cost: CostEstimate | None) -> str:
    """Dollars, with "?" when some model's tokens had no price and "!" when its cache counts exceed its input; "-" when no tokens were reported."""
    if cost is None:
        return "-"
    return f"${cost.dollars:.4f}" + ("?" if cost.unpriced else "") + ("!" if cost.inconsistent_models else "")


def _inconsistent_line(models: list[str]) -> str:
    return f"cached tokens exceed input for: {', '.join(models)} (cost understated)"


def format_results_table(results: list[CaseResult]) -> str:
    """One row per case: id, category, PASS or FAIL, seconds, model tokens,
    estimated cost, and why the first failing check failed; then a total row.
    A dash means the provider reported no tokens of that kind; a "?" on a cost
    means some model had no price, and a footer names it."""
    token_headers = " ".join(f"{label:>9}" for _, label in _TOKEN_COLUMNS)
    header = f"{'case':<42} {'category':<18} {'result':<6} {'seconds':>8} {token_headers} {'est cost':>9}  reason"
    rule = "-" * len(header)
    lines = [header, rule]
    for result in results:
        failure = result.first_failure
        if failure is not None:
            reason = failure.reason
        elif not result.within_budget:
            reason = f"over the {CATEGORY_BUDGETS[result.case.category]:.0f}s budget"
        else:
            reason = ""
        tokens = " ".join(f"{_token_cell(result.outcome.token_total(kind)):>9}" for kind, _ in _TOKEN_COLUMNS)
        cost = estimate_cost(result.outcome)
        lines.append(
            f"{result.case.id:<42} {result.case.category:<18} {'PASS' if result.passed else 'FAIL':<6} "
            f"{result.outcome.elapsed_seconds:>8.2f} {tokens} {_cost_cell(cost):>9}  {reason}".rstrip()
        )
    passed = sum(1 for result in results if result.passed)
    seconds = sum(result.outcome.elapsed_seconds for result in results)
    totals = []
    for kind, _ in _TOKEN_COLUMNS:
        counts = [count for result in results if (count := result.outcome.token_total(kind)) is not None]
        totals.append(f"{_token_cell(sum(counts) if counts else None):>9}")
    costs = [cost for result in results if (cost := estimate_cost(result.outcome)) is not None]
    unpriced = sorted({model for cost in costs for model in cost.unpriced})
    inconsistent = sorted({model for cost in costs for model in cost.inconsistent_models})
    total_cost = (
        CostEstimate(sum(cost.dollars for cost in costs), tuple(unpriced), tuple(inconsistent)) if costs else None
    )
    lines.append(rule)
    lines.append(
        f"{'total':<42} {'':<18} {f'{passed}/{len(results)}':<6} {seconds:>8.2f} {' '.join(totals)} "
        f"{_cost_cell(total_cost):>9}"
    )
    lines.append("est cost: list prices from benchmark/pricing.py, not the provider's bill")
    if unpriced:
        lines.append(f"no price for: {', '.join(unpriced)}")
    return "\n".join(lines)


WORKFLOW_TABLE_HEADER = (
    "Workflow adapter: no model. Each case names the workflow that answers it and its inputs, and the "
    "harness runs it on the fake data boundary.\n"
    "These results are not compared with the pass threshold or the current system's results."
)


def _grouped_table(
    header_text: str, label: str, results: list[CaseResult], uncovered: list[str], key: Callable[[CaseResult], str]
) -> list[str]:
    """One row per group (cases covered, passed, failed), each failing case's first failed
    check beneath it, then the ids of the cases not covered."""
    groups: dict[str, list[CaseResult]] = {}
    for result in results:
        groups.setdefault(key(result), []).append(result)
    header = f"{label:<20} {'covered':>7} {'passed':>6} {'failed':>6}"
    lines = [header_text, "", header, "-" * len(header)]
    for name in sorted(groups):
        group = groups[name]
        passed = sum(1 for result in group if result.passed)
        lines.append(f"{name:<20} {len(group):>7} {passed:>6} {len(group) - passed:>6}")
        for result in group:
            if result.passed:
                continue
            failure = result.first_failure
            reason = failure.reason if failure is not None else (
                f"over the {CATEGORY_BUDGETS[result.case.category]:.0f}s budget"
            )
            lines.append(f"    FAIL {result.case.id}: {reason}")
    passed_total = sum(1 for result in results if result.passed)
    lines.append("-" * len(header))
    lines.append(f"{'total':<20} {len(results):>7} {passed_total:>6} {len(results) - passed_total:>6}")
    lines.append(f"not covered ({len(uncovered)}): {', '.join(uncovered) if uncovered else 'none'}")
    return lines


def format_workflow_table(results: list[CaseResult], uncovered: list[str]) -> str:
    """One row per workflow (cases covered, passed, failed), each failing case's first failed
    check beneath it, then the ids of the cases no workflow covers."""
    return "\n".join(_grouped_table(
        WORKFLOW_TABLE_HEADER, "workflow", results, uncovered, lambda result: result.case.workflow or ""
    ))


EXPLORE_TABLE_HEADER = (
    "Explore adapter: exploratory, with a real model. Each case's prompt is the objective of one exploration task\n"
    "run with the case's bundle on the fake data boundary. Model output varies between runs.\n"
    "These results are not compared with the pass threshold or the current system's results."
)


def format_explore_table(results: list[CaseResult], uncovered: list[str], cost_dollars: float | None) -> str:
    """One row per bundle, each failing case's first failed check beneath it, the cases not
    covered, and the estimated cost."""
    lines = _grouped_table(
        EXPLORE_TABLE_HEADER, "bundle", results, uncovered, lambda result: result.case.bundle or ""
    )
    lines.append(
        "estimated cost: not available (no model tokens reported)" if cost_dollars is None
        else f"estimated cost: ${cost_dollars:.4f} (list prices; models without a price are not counted)"
    )
    costs = [cost for result in results if (cost := estimate_cost(result.outcome)) is not None]
    inconsistent = sorted({model for cost in costs for model in cost.inconsistent_models})
    if inconsistent:
        lines.append(_inconsistent_line(inconsistent))
    return "\n".join(lines)
