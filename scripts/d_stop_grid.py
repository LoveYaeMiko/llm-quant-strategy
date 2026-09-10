"""D-8c — stop-width re-tune under the CORRECTED code (production-isomorphic).

Why: the deployed `pb_atr_mult / pb_stop_lo / pb_stop_hi` were selected by a grid
run while the true-range computation was collapsed to a per-date Series, which
silently forced EVERY stop to the 2.5% floor (defect D-8b). The parameters were
therefore never validated against a real ATR stop. After the fix,
`outputs/d_atr_impact.json` showed ATR-adaptive 2.5–4% (Sharpe 0.58) is far worse
than a flat 2.5% (Sharpe 1.53) in the 2026 in-sample window — so this grid
brackets the stop WIDTH itself, both flat and ATR-adaptive, on the same
production assembly.

Every variant runs through `src.cli._shadow_cycle` → `_build_account_portfolio`
(the 15:10 job's own path) into a FRESH ledger, with the production market slice
passed in (`market_override`) so only ONE market/feature load is paid for.

2026-09-10 — the PIT pool defect VOIDED the first run of this grid. It had seen
only ~301 symbols per day in 2026 (the incremental daily ingest covered ~300
names) while the D track declares an 800-name (`hs300_500`) book, so its absolute
numbers were wrong and its relative order unverified. The 499 missing names were
backfilled the same evening (`scripts/backfill_price_gap.py`; monthly coverage is
now 800 for every 2026 month), and the grid is re-run on the corrected pool for
BOTH windows:

    python scripts/d_stop_grid.py 2026-01-01 2026-08-28 --label is_2026_800
    python scripts/d_stop_grid.py 2025-09-01 2025-12-31 --label oos_2025h2_800

The artifact carries, next to the numbers, the evidence that the run was
production-isomorphic and COMPLETE: per variant `params_hash` (must equal the
deployed `d_oos_is_2026_v5.json` hash for `flat_3p5`), `n_symbols` actually
traded-from and the panel's `n_bars`; plus a top-level `panel` block with
`panel_universe_health(price_panel, as_of=end)`
(`n_columns / n_with_price / n_warm_20 / n_warm_60 / effective_ratio`) and the
five-field provenance block. A grid whose cross-section silently shrank cannot
look like a full-universe run any more.

CLI (the historical form still works):

    python scripts/d_stop_grid.py <start> <end> [--only=a,b] [--label NAME] [--out PATH]
    python scripts/d_stop_grid.py --reuse outputs/d_stop_grid_is_2026_800.json [--out PATH]

``--reuse`` re-emits an existing artifact (re-shaped, re-labelled, re-hashed) with
NO run; it refuses an artifact that has no provenance block, and keeps that
artifact's own `window / convention / data_as_of / code_commit` — it never
fabricates provenance for numbers it did not produce.

Research only: never touches the production ledger, writes no status/report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.provenance import stamp_artifact  # noqa: E402


#: label -> account overrides on top of the deployed D_5W config.
VARIANTS: dict[str, dict] = {
    # --- flat widths (stop_lo == stop_hi makes the ATR channel inert) ---
    "flat_1p5": {"pb_stop_lo": 0.015, "pb_stop_hi": 0.015},
    "flat_2p0": {"pb_stop_lo": 0.020, "pb_stop_hi": 0.020},
    "flat_2p5": {"pb_stop_lo": 0.025, "pb_stop_hi": 0.025},   # previous deployment
    "flat_3p0": {"pb_stop_lo": 0.030, "pb_stop_hi": 0.030},
    "flat_3p5": {"pb_stop_lo": 0.035, "pb_stop_hi": 0.035},   # current deployment
    # --- ATR-adaptive bands ---
    "atr_1p0_25_35": {"pb_atr_mult": 1.0, "pb_stop_lo": 0.025, "pb_stop_hi": 0.035},
    "atr_1p0_25_40": {"pb_atr_mult": 1.0, "pb_stop_lo": 0.025, "pb_stop_hi": 0.040},
    "atr_1p5_25_40": {"pb_atr_mult": 1.5, "pb_stop_lo": 0.025, "pb_stop_hi": 0.040},
}

#: Top-level keys of an artifact — everything else at top level is a VARIANT
#: entry (the 2026-09-09 artifact was a flat ``{label: result}`` map).
ARTIFACT_KEYS: frozenset[str] = frozenset({
    "label", "window", "variants", "panel", "assembly", "kill_switch",
    "provenance", "convention", "data_as_of", "artifact_sha256", "code_commit",
    "code_dirty",
})

#: The historical default (still the destination when neither --out nor --label
#: is given, so the documented ``outputs/d_stop_grid.json`` path keeps working).
DEFAULT_OUT = ROOT / "outputs" / "d_stop_grid.json"


def _out_path(label: str, out: str) -> Path:
    """Artifact destination: explicit ``--out`` wins, else the label suffix."""
    if out:
        p = Path(out)
        return p if p.is_absolute() else ROOT / p
    if label:
        return ROOT / "outputs" / f"d_stop_grid_{label}.json"
    return DEFAULT_OUT


def _select_variants(only: str) -> dict[str, dict]:
    """The variants to RUN (``--only=a,b``); no filter means all of them."""
    if not only:
        return dict(VARIANTS)
    want = {x.strip() for x in only.split(",") if x.strip()}
    return {k: v for k, v in VARIANTS.items() if k in want}


def _load_variants(path: Path) -> dict[str, dict]:
    """Variant entries of an existing artifact, in either artifact shape.

    ``--only=...`` re-runs ONE variant; the others must survive from the previous
    artifact. Accepts the current ``{"variants": {...}}`` nesting and the legacy
    flat map, and never raises (an unreadable file just means "nothing to merge").
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    nested = data.get("variants")
    if isinstance(nested, dict):
        return dict(nested)
    return {k: v for k, v in data.items() if k not in ARTIFACT_KEYS and isinstance(v, dict)}


def _merge_existing(results: Mapping[str, Any], ran: Mapping[str, Any]) -> dict[str, dict]:
    """Variants kept from a previous artifact, marked as NOT re-run this time.

    ``--only=flat_3p5`` re-runs one variant; the others must survive in the
    artifact. They are flagged ``reused`` so a reader cannot mistake a stale row
    for one produced by this run (and so a stale row is easy to spot in the JSON).
    The input mapping is never mutated.
    """
    out: dict[str, dict] = {}
    for name, prev in results.items():
        if isinstance(prev, dict) and name not in ran:
            out[name] = {**prev, "reused": True}
        else:
            out[name] = prev
    return out


def _convention(account: Mapping[str, Any]) -> str:
    """The execution/price convention EVERY variant in this grid is on.

    Two artifacts with different conventions are not comparable, and a grid is
    only a grid if all variants share one market slice — both facts are stated
    here rather than assumed by the reader.
    """
    return (
        "adjusted-close price basis; daily close rebalance at the panel close; "
        f"intraday stops on minute bars below 15:00 "
        f"(trigger={account.get('pb_stop_trigger')}, "
        f"open_minutes={account.get('pb_stop_open_minutes')}); T+1; no leverage; "
        "commission+stamp duty per the configured cost model; "
        "ALL variants ran on the SAME market slice (one _build_market_for_paper per "
        "grid, handed to every variant through market_override), so they differ only "
        "in the stop-width override"
    )


def _artifact(
    *,
    label: str,
    start: str,
    end: str,
    variants_out: Mapping[str, Any],
    panel: Mapping[str, Any],
    assembly: Mapping[str, Any],
    kill_switch: Mapping[str, Any],
    convention: str,
    data_as_of: str,
) -> dict:
    """Assemble + stamp the artifact payload (pure except for the git stamp)."""
    window = {"start": str(start)[:10], "end": str(end)[:10]}
    payload = {
        "label": label,
        "window": window,
        "variants": dict(variants_out),
        "panel": dict(panel),
        "assembly": dict(assembly),
        "kill_switch": dict(kill_switch),
    }
    return stamp_artifact(
        payload, window=window, convention=convention, data_as_of=data_as_of
    )


def _reemit(path: Path, out_path: Path, *, label: str = "") -> int:
    """``--reuse``: re-emit an existing artifact without running the grid."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    block = dict(data.get("provenance") or {})
    missing = [k for k in ("window", "convention", "data_as_of", "code_commit") if not block.get(k)]
    if missing:
        print(
            f"ERROR: {path} has no usable provenance block (missing {missing}) — "
            "re-run the grid instead of reusing it",
            file=sys.stderr,
        )
        return 2
    payload = {k: v for k, v in data.items() if k != "provenance"}
    if label:
        payload["label"] = label
    # Keep the ORIGINAL code_commit: the numbers were produced by that commit, not
    # by whatever HEAD happens to be while re-emitting.
    stamped = stamp_artifact(
        payload,
        window=block["window"],
        convention=str(block["convention"]),
        data_as_of=str(block["data_as_of"]),
        code_commit=str(block["code_commit"]),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(stamped, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[grid] reuse: {path} -> {out_path} "
          f"(label={stamped.get('label')!r}, {len(stamped.get('variants') or {})} variants, "
          f"sha256={stamped['provenance']['artifact_sha256'][:16]}…)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    start, end, label = args.start, args.end, args.label
    out_path = _out_path(label, args.out)

    if args.reuse:
        src = Path(args.reuse)
        if not src.is_absolute():
            src = ROOT / src
        return _reemit(src, out_path, label=label)

    variants = _select_variants(args.only)
    if not variants:
        print(f"ERROR: --only={args.only!r} matched no variant "
              f"(known: {', '.join(VARIANTS)})", file=sys.stderr)
        return 2

    import pandas as pd

    from src.cli import _build_market_for_paper, _shadow_cycle
    from src.config import load_config
    from src.forward.risk_gate import panel_universe_health
    from src.paper.ledger import PaperLedger
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    base_account = next(
        (a for a in (cfg.get("shadow.accounts") or []) if str(a.get("name")) == "D_5W"), None
    )
    if base_account is None:
        print("ERROR: D_5W not in shadow.accounts", file=sys.stderr)
        return 2
    symbols_declared = resolve_shadow_universe(cfg, base_account.get("universe"))
    n_declared = len(symbols_declared)

    # Same kill-switch state as the production 15:10 run: once the gate leaves
    # `normal`, a research run that ignores it is NOT isomorphic (it would trade
    # at full size while production de-risks).
    from src.autopilot.state import ControlState

    state_path = str(
        ROOT / str(cfg.get("autopilot.state_file", "outputs/autopilot_state.json"))
    ).replace(".json", f"_{base_account['name']}.json")
    control = ControlState.load(state_path)

    print(f"[grid] window=[{start}, {end}] label={label!r} variants={list(variants)} "
          f"out={out_path.name}", flush=True)
    market = _build_market_for_paper(cfg, symbols_declared, start, None, seed=1)
    symbols = [s for s in symbols_declared if s in market.price_panel.columns]
    n_bars = int(len(market.price_panel.index))
    last_bar = str(pd.Timestamp(market.price_panel.index.max()).date())

    # Panel health on the window's last day: a run whose cross-section is not the
    # declared book must be visible in the artifact, not in a footnote.
    panel = dict(panel_universe_health(market.price_panel, as_of=end))
    panel.update({
        "n_symbols": len(symbols),
        "n_declared": n_declared,
        "n_bars": n_bars,
        "last_bar": last_bar,
    })
    print(f"[grid] market ready: {len(symbols)} symbols traded-from "
          f"(declared {n_declared}), {n_bars} bars, last bar {last_bar}", flush=True)
    print(f"[grid] panel: n_with_price={panel.get('n_with_price')} "
          f"n_warm_20={panel.get('n_warm_20')} n_warm_60={panel.get('n_warm_60')} "
          f"effective_ratio={panel.get('effective_ratio')}", flush=True)
    print(f"[grid] kill-switch: mode={control.mode} gross={control.gross_scale:g} "
          f"({Path(state_path).name})", flush=True)

    results: dict[str, dict] = (
        _merge_existing(_load_variants(out_path), variants) if args.only else {}
    )
    last_probe: dict = {}

    for vlabel, over in variants.items():
        account = {**base_account, **over}
        ledger_path = ROOT / "outputs" / f"_dstop_{vlabel}.sqlite"
        ledger_path.unlink(missing_ok=True)
        probe: dict = {}
        status, _ = _shadow_cycle(
            cfg, symbols, start, end, 1, skip_refresh=True,
            control_scale=control.gross_scale,
            account=account, ledger_override=str(ledger_path), write_artifacts=False,
            probe=probe, market_override=market,
        )
        last_probe = probe
        ledger = PaperLedger(str(ledger_path))
        fills = ledger.fills()
        by_source = ledger.fills_by_source()
        n_intraday = (
            int((fills.get("time", "").fillna("").astype(str) != "").sum()) if len(fills) else 0
        )
        ledger.close()
        ledger_path.unlink(missing_ok=True)

        eq = status.get("equity", {})
        stop_widths = _stop_width_stats(probe.get("params") or {})
        results[vlabel] = {
            "overrides": over,
            "cum_return": eq.get("total_return", 0.0),
            "ann_return": eq.get("annualized_return", 0.0),
            "sharpe": eq.get("sharpe", 0.0),
            "max_dd": eq.get("max_drawdown", 0.0),
            "n_days": eq.get("n_days", 0),
            "n_fills": eq.get("n_fills", 0),
            "n_intraday": n_intraday,
            "cost": eq.get("total_commission", 0.0),
            "fills_by_source": by_source,
            "stop_width": stop_widths,
            "params_hash": probe.get("params_hash"),
            # Completeness / isomorphism evidence: the cross-section this variant
            # actually traded from, and the panel it was built on.
            "n_symbols": len(symbols),
            "n_bars": n_bars,
            "gross_scale_wired": bool(probe.get("gross_scale_wired")),
        }
        r = results[vlabel]
        print(f"{vlabel:16s}: cum={r['cum_return']:+.2%} ann={r['ann_return']:+.2%} "
              f"sharpe={r['sharpe']:+.2f} maxDD={r['max_dd']:.2%} fills={r['n_fills']} "
              f"intraday={r['n_intraday']} cost={r['cost']:,.0f} "
              f"n_symbols={r['n_symbols']} bars={r['n_bars']} "
              f"hash={r['params_hash']}", flush=True)

    assembly = {
        "book_class": last_probe.get("book_class"),
        "execution": last_probe.get("execution"),
        "universe_size": last_probe.get("universe_size"),
        "has_intraday_frames": last_probe.get("has_intraday_frames"),
        "has_minute_provider": last_probe.get("has_minute_provider"),
        "always_rebalance": last_probe.get("always_rebalance"),
        "gross_scale_wired": last_probe.get("gross_scale_wired"),
        "live_intraday_from": last_probe.get("live_intraday_from"),
    }
    kill_switch = {"mode": control.mode, "gross_scale": control.gross_scale,
                   "state_file": str(state_path)}

    artifact = _artifact(
        label=label, start=start, end=end, variants_out=results, panel=panel,
        assembly=assembly, kill_switch=kill_switch,
        convention=_convention(base_account), data_as_of=last_bar,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    ranked = sorted(results, key=lambda k: results[k].get("sharpe", 0.0), reverse=True)
    best = ranked[0]
    print(f"\nbest by Sharpe: {best} "
          f"(sharpe={results[best]['sharpe']:.2f} ann={results[best]['ann_return']:+.2%})")
    ranking = " > ".join(f"{k}({results[k].get('sharpe', 0.0):.2f})" for k in ranked)
    print(f"ranking: {ranking}")
    print(f"[grid] provenance: data_as_of={last_bar} "
          f"commit={artifact['provenance']['code_commit'][:12]} "
          f"sha256={artifact['provenance']['artifact_sha256'][:16]}…")
    print(f"wrote {out_path}", flush=True)
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="D-8c stop-width grid (8 variants, production-isomorphic)",
    )
    ap.add_argument("start", nargs="?", default="2026-01-01", help="window start")
    ap.add_argument("end", nargs="?", default="2026-08-28", help="window end")
    ap.add_argument("--only", default="", help="comma-separated variant labels (default: all)")
    ap.add_argument("--label", default="", help="run label; also the output suffix when "
                                               "--out is not given")
    ap.add_argument("--out", default="", help="artifact path "
                                             "(default outputs/d_stop_grid[_<label>].json)")
    ap.add_argument("--reuse", default="", metavar="PATH",
                    help="re-emit an existing artifact (no run, no re-computation)")
    return ap.parse_args(argv)


def _stop_width_stats(params: dict) -> dict:
    """The stop width the parameters imply (flat vs ATR band)."""
    lo = float(params.get("stop_lo", 0.0) or 0.0)
    hi = float(params.get("stop_hi", 0.0) or 0.0)
    mult = float(params.get("atr_mult", 0.0) or 0.0)
    return {
        "stop_lo": lo,
        "stop_hi": hi,
        "atr_mult": mult,
        "flat": abs(hi - lo) < 1e-12,
        "band": [lo, hi],
    }


if __name__ == "__main__":
    raise SystemExit(main())
