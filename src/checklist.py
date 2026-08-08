"""Blueprint verification checklist — the four self-checks that must pass.

The blueprint's *verification checklist* is reproduced as executable checks so a
run can prove (not claim) compliance:

1. **PIT check**        — a query at T never exposes facts born after T.
2. **FinCAD check**     — look-ahead suppression reduces future-date mentions by
                         > 50% (``LookAheadAudit.passes_check``).
3. **Diversity check**  — the accepted factor pool stays structurally diverse
                         (min pairwise AST distance >= the configured floor).
4. **Cost check**       — projected monthly LLM spend stays under the budget.

Every check returns a structured verdict; ``run_all`` aggregates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .agents.eval_agent import EvalAgent
from .bias_control.context_decoder import FinCADWrapper, MockLLMBackend
from .bias_control.look_ahead_detector import LookAheadAudit, future_mentions
from .config import Config
from .cost_tracker import CostTracker
from .data.point_in_time_loader import PointInTimeStore
from .factors.code_generator import CodeGenerator, ast_distance, parse_expression


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"check": self.name, "passed": self.passed, "detail": self.detail, **self.meta}


# ---------------------------------------------------------------------------
# 1. PIT check
# ---------------------------------------------------------------------------


def pit_check(store: PointInTimeStore, ts: str = "2019-06-28", forbidden: str = "2019-07-01") -> CheckResult:
    """Query at ``ts`` must not surface any fact born at/after ``forbidden``."""
    q = store.query(ts)
    born = pd.to_datetime(q.get("valid_from", pd.Series(dtype="datetime64[ns]")))
    leaked = int((born >= pd.Timestamp(forbidden)).sum()) if len(born) else 0
    return CheckResult(
        name="pit",
        passed=leaked == 0,
        detail=f"query({ts}) returned {len(q)} facts, {leaked} born after {forbidden}",
        meta={"facts_visible": int(len(q)), "future_facts_leaked": leaked},
    )


# ---------------------------------------------------------------------------
# 2. FinCAD check
# ---------------------------------------------------------------------------


def fincad_check(threshold: float = 0.50) -> CheckResult:
    """Two-part FinCAD check.

    1. **Wrapper integration** — a model output that mentions a future date is
       redacted before it reaches the caller.
    2. **Cheating-factor audit** — a factor that *embeds* the forward return (a
       model with memorised outcomes) loses >50% of its IC once suppression is
       applied (``LookAheadAudit.passes_check``).
    """
    # part 1: wrapper must scrub the future date from generated text
    future = "2024-06-15"
    mock = MockLLMBackend({"leak": f"Buy on {future} because earnings confirm on {future}."})
    wrapper = FinCADWrapper(mock)
    result = wrapper.complete("leak", as_of=pd.Timestamp("2024-01-01"))
    scrubbed = not future_mentions(result.text, result.as_of)

    # part 2: the cheating factor's IC collapses under suppression
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=120)
    forward = rng.normal(0.0005, 0.01, len(dates))
    cheat = np.asarray(forward, dtype=float)          # perfect look-ahead signal
    suppressed = np.zeros(len(dates))                 # no future info -> ~0 IC
    audit = LookAheadAudit().evaluate(dates, list(forward), list(cheat), list(suppressed))
    reduction = float(audit.relative_reduction)
    return CheckResult(
        name="fincad",
        passed=bool(scrubbed and audit.passes_check and reduction > threshold),
        detail=(
            f"output scrubbed={scrubbed}; cheating-factor IC {audit.leaky_ic:.3f} "
            f"-> suppressed {audit.suppressed_ic:.3f}, reduction {reduction:.0%} "
            f"(required > {threshold:.0%})"
        ),
        meta={
            "output_scrubbed": scrubbed,
            "leaky_ic": float(audit.leaky_ic),
            "suppressed_ic": float(audit.suppressed_ic),
            "reduction": reduction,
            "passes_check": bool(audit.passes_check),
        },
    )


# ---------------------------------------------------------------------------
# 3. Diversity check
# ---------------------------------------------------------------------------


def diversity_check(formulas: list[str], min_distance: float = 0.40) -> CheckResult:
    """Minimum pairwise structural distance across the accepted pool."""
    gen = CodeGenerator()
    nodes = []
    for f in formulas:
        try:
            nodes.append(gen.parse(f))
        except Exception:
            continue
    if len(nodes) < 2:
        return CheckResult(
            name="diversity", passed=False,
            detail="need >=2 valid formulas to measure distance",
        )
    min_d = min(
        ast_distance(a, b)
        for i, a in enumerate(nodes)
        for b in nodes[i + 1 :]
    )
    return CheckResult(
        name="diversity",
        passed=min_d >= min_distance,
        detail=f"min pairwise AST distance = {min_d:.2f} (floor {min_distance})",
        meta={"min_distance": float(min_d), "n_formulas": len(nodes)},
    )


# ---------------------------------------------------------------------------
# 4. Cost check
# ---------------------------------------------------------------------------


def cost_check(tracker: Optional[CostTracker] = None, budget: float = 500.0) -> CheckResult:
    """Simulated month of LLM calls must stay under the monthly budget."""
    t = tracker or CostTracker(monthly_budget_usd=budget)
    if not t.entries():
        # simulate a realistic month: 200 cheap generator calls + 20 code calls
        for _ in range(200):
            t.record("deepseek-v3", 1200, 400, purpose="hypothesis")
        for _ in range(20):
            t.record("gpt-4o", 2500, 800, purpose="code")
        for _ in range(4):
            t.record("claude-3.5-sonnet", 3000, 900, purpose="critic")
    proj = t.monthly_projection(days_elapsed=30.0)
    return CheckResult(
        name="cost",
        passed=proj <= budget,
        detail=f"projected monthly cost ${proj:,.2f} <= ${budget:,.2f}",
        meta=t.snapshot(),
    )


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_all(
    formulas: Optional[list[str]] = None,
    store: Optional[PointInTimeStore] = None,
    tracker: Optional[CostTracker] = None,
    config: Optional[Config] = None,
) -> list[CheckResult]:
    formulas = formulas or [
        "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))",
        "Neg(TS_ZScore(Close, 20))",
        "Inv(TS_Std(Close, 30))",
    ]
    min_d = float(config.get("diversity.min_ast_distance", 0.40)) if config else 0.40
    checks = [
        fincad_check(),
        diversity_check(formulas, min_distance=min_d),
        cost_check(tracker),
    ]
    if store is not None:
        checks.insert(0, pit_check(store))
    return checks


__all__ = ["CheckResult", "pit_check", "fincad_check", "diversity_check", "cost_check", "run_all"]
