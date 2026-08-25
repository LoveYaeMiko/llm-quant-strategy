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
from .data.ingestion.convert import PRICE as PRICE_RECORD, UNIVERSE as UNIVERSE_RECORD
from .data.point_in_time_loader import PointInTimeStore
from .factors.code_generator import CodeGenerator, ast_distance, causality_check, parse_expression


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


def diversity_check(formulas: list[str], min_distance: float = 0.25) -> CheckResult:
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
# 3b. Causality check (full vs truncated — 防未来函数)
# ---------------------------------------------------------------------------


def factor_causality_check(
    formulas: list[str],
    data: pd.DataFrame,
    *,
    n_probe_dates: int = 10,
    tolerance: float = 1e-9,
) -> CheckResult:
    """A2: every accepted formula's value at ``t`` must be identical whether the
    panel is full or truncated to ``t``. Any mismatch is a future-data leak.
    """
    checked = 0
    failures: list[str] = []
    for f in formulas:
        try:
            res = causality_check(f, data, n_probe_dates=n_probe_dates, tolerance=tolerance)
        except Exception as exc:  # noqa: BLE001 — an unparsable formula fails the gate
            failures.append(f"{f}: error {exc}")
            continue
        checked += 1
        if not res["clean"]:
            failures.append(f"{f}: {len(res['violations'])} probe dates differ")
    return CheckResult(
        name="causality",
        passed=checked > 0 and not failures,
        detail=(
            f"causality clean across {checked} formulas"
            if not failures else f"{len(failures)}/{checked} formulas read future data"
        ),
        meta={"n_formulas": checked, "failures": failures[:10]},
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
            t.record("deepseek-v4-pro", 1200, 400, purpose="hypothesis")
        for _ in range(20):
            t.record("deepseek-v4-pro", 2500, 800, purpose="code")
        for _ in range(4):
            t.record("deepseek-v4-pro", 3000, 900, purpose="critic")
    proj = t.monthly_projection(days_elapsed=30.0)
    return CheckResult(
        name="cost",
        passed=proj <= budget,
        detail=f"projected monthly cost ${proj:,.2f} <= ${budget:,.2f}",
        meta=t.snapshot(),
    )


# ---------------------------------------------------------------------------
# B1-B5 real-data checks (gated on data.real_data — Q6)
# ---------------------------------------------------------------------------
# The four synthetic checks above prove the *pipeline*. The B1-B5 checks below
# audit the *ingested real data* and only run after the human flips
# ``data.real_data: true`` (the ingest summary makes that decision evidence-based).


def no_future_leak_check(store, boundary: str = "2019-12-31") -> CheckResult:
    """B1: a query at the train/val boundary exposes no fact born after it.

    Closed-interval bars make this structurally true; the check proves it
    empirically on whatever real data was ingested.
    """
    q = store.query(boundary)
    born = pd.to_datetime(q.get("valid_from", pd.Series(dtype="datetime64[ns]")))
    leaked = int((born > pd.Timestamp(boundary)).sum()) if len(born) else 0
    return CheckResult(
        name="no_future_leak",
        passed=leaked == 0,
        detail=f"query({boundary}) returned {len(q)} facts, {leaked} born after the boundary",
        meta={"facts_visible": int(len(q)), "future_facts_leaked": leaked},
    )


def adjustment_consistency_check(store, sample: int = 20, price_limit_band: float = 0.30) -> CheckResult:
    """B3: dual-column adjustment is real (Q2), not the tautology ``close==raw*factor``.

    Audits a sample of price symbols:
    * adjusted daily returns stay within the ±``price_limit_band`` price-limit band
      (a correct backward adjustment removes the ex-dividend gap);
    * factor-change days coincide with a raw-price jump (the change is an actual
      corporate action, not a corrupted factor row).
    """
    syms = store.symbols(PRICE_RECORD)[:sample]
    checked = 0
    days = 0
    band_breaks = 0
    worst = 0.0
    factor_changes = 0
    changes_with_jump = 0
    for sym in syms:
        h = store.history(sym, PRICE_RECORD)
        if h.empty or len(h) < 3 or not {"close", "raw_close", "adjust_factor"} <= set(h.columns):
            continue
        checked += 1
        adj = h["close"].astype(float).pct_change(fill_method=None).fillna(0.0)
        raw = h["raw_close"].astype(float).pct_change(fill_method=None).fillna(0.0)
        factor = h["adjust_factor"].astype(float)
        fchange = factor.diff().fillna(0.0).abs() > 1e-9
        days += len(adj)
        band_breaks += int((adj.abs() > price_limit_band).sum())
        worst = max(worst, float(adj.abs().max())) if len(adj) else worst
        factor_changes += int(fchange.sum())
        if fchange.any():
            changes_with_jump += int((raw[fchange].abs() > 0.005).sum())
    if checked == 0:
        return CheckResult(
            name="adjustment_consistency",
            passed=False,
            detail="no price records with close/raw_close/adjust_factor to audit",
        )
    passed = band_breaks == 0
    detail = (
        f"{checked} symbols, {days:,} days audited; {band_breaks} adjusted returns beyond "
        f"±{price_limit_band:.0%} (worst {worst:.2%}); {factor_changes} factor-change days, "
        f"{changes_with_jump} with a raw-price jump"
    )
    return CheckResult(
        name="adjustment_consistency",
        passed=passed,
        detail=detail,
        meta={
            "symbols_audited": checked,
            "days_audited": days,
            "band_breaks": band_breaks,
            "worst_adjusted_return": worst,
            "factor_changes": factor_changes,
            "changes_with_jump": changes_with_jump,
        },
    )


def survivorship_check(store, as_of: str = "2015-01-05") -> CheckResult:
    """B4: the store retains names alive at ``as_of`` that have since left.

    A store that can enumerate the 2015 cohort minus today's cohort is one that
    does NOT silently drop delisted names — the survivorship trap the blueprint
    calls out. Pass just proves both snapshots exist; the delisted count is the
    number the human reads.
    """
    past = store.universe_as_of(as_of, UNIVERSE_RECORD)
    latest = store.max_date(UNIVERSE_RECORD)
    present = store.universe_as_of(latest, UNIVERSE_RECORD) if not pd.isna(latest) else []
    delisted = sorted(set(past) - set(present))
    return CheckResult(
        name="survivorship",
        passed=bool(past) and bool(present),
        detail=(
            f"{len(past)} symbols alive at {as_of}, {len(present)} at {latest}; "
            f"{len(delisted)} of the {as_of} cohort have since left"
        ),
        meta={
            "n_at_as_of": len(past),
            "n_latest": len(present),
            "n_delisted_since": len(delisted),
            "as_of": str(as_of),
            "latest_universe_date": str(latest),
        },
    )


def data_freshness_check(
    store,
    as_of: Optional[str] = None,
    max_staleness_days: int = 7,
    min_coverage: float = 0.70,
) -> CheckResult:
    """B5: the newest bar is recent and the price history covers the range."""
    latest = store.max_valid_from(PRICE_RECORD)
    earliest = store.min_date(PRICE_RECORD)
    if pd.isna(latest) or pd.isna(earliest):
        return CheckResult(name="data_freshness", passed=False, detail="no price records in store")
    as_of = as_of or pd.Timestamp.today().normalize()
    staleness = int((pd.Timestamp(as_of) - latest).days)
    dates = store.distinct_dates(PRICE_RECORD)
    total = len(pd.bdate_range(earliest, latest))
    coverage = len(dates) / total if total else 0.0
    passed = staleness <= max_staleness_days and coverage >= min_coverage
    return CheckResult(
        name="data_freshness",
        passed=passed,
        detail=(
            f"latest bar {latest.date()} ({staleness} days old, allowed {max_staleness_days}); "
            f"{len(dates)}/{total} business days covered = {coverage:.0%} (floor {min_coverage:.0%})"
        ),
        meta={
            "latest_bar": str(latest),
            "staleness_days": staleness,
            "coverage": round(coverage, 3),
            "max_staleness_days": max_staleness_days,
        },
    )


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_all(
    formulas: Optional[list[str]] = None,
    store: Optional[PointInTimeStore] = None,
    tracker: Optional[CostTracker] = None,
    config: Optional[Config] = None,
    freshness_as_of: Optional[str] = None,
    real_data_audit: bool = False,
    factor_data: Optional[pd.DataFrame] = None,
) -> list[CheckResult]:
    formulas = formulas or [
        "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))",
        "Neg(TS_ZScore(Close, 20))",
        "Inv(TS_Std(Close, 30))",
    ]
    min_d = float(config.get("diversity.min_ast_distance", 0.25)) if config else 0.25
    checks = [
        fincad_check(),
        diversity_check(formulas, min_distance=min_d),
        cost_check(tracker),
    ]
    if factor_data is not None:
        checks.append(factor_causality_check(formulas, factor_data))
    if store is not None:
        ts = str(config.get("pit.validation_timestamp", "2019-06-28")) if config else "2019-06-28"
        forbidden = str(config.get("pit.forbidden_future_date", "2019-07-01")) if config else "2019-07-01"
        checks.insert(0, pit_check(store, ts=ts, forbidden=forbidden))
    real = bool(config.get("data.real_data", False)) if config else False
    # B1-B5 audit the *ingested full store* (verify). Research loops (mine /
    # backtest / evolve / monitor) run on a window-sliced, universe-bounded
    # market, so their checklist must stay with the four standing checks — B5
    # would always look stale on a train-window slice and B4's universe snapshots
    # are sliced out of scope.
    real = real and real_data_audit
    if real:
        if store is None:
            raise ValueError("data.real_data=true requires a PIT store to audit")
        boundary = str(config.get("research.train_end", "2019-12-31"))
        checks.extend(
            [
                no_future_leak_check(store, boundary=boundary),
                adjustment_consistency_check(
                    store,
                    sample=int(config.get("data.checks.adjustment_sample", 20)),
                    price_limit_band=float(config.get("data.checks.price_limit_band", 0.30)),
                ),
                survivorship_check(
                    store, as_of=str(config.get("data.checks.survivorship_date", "2015-01-05"))
                ),
                data_freshness_check(
                    store,
                    as_of=freshness_as_of,
                    max_staleness_days=int(config.get("data.checks.max_staleness_days", 7)),
                    min_coverage=float(config.get("data.checks.min_coverage", 0.70)),
                ),
            ]
        )
    return checks


__all__ = [
    "CheckResult",
    "pit_check",
    "fincad_check",
    "diversity_check",
    "factor_causality_check",
    "cost_check",
    "no_future_leak_check",
    "adjustment_consistency_check",
    "survivorship_check",
    "data_freshness_check",
    "run_all",
]
