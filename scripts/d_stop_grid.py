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

Research only: never touches the production ledger, writes no status/report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


#: label -> account overrides on top of the deployed D_5W config.
VARIANTS: dict[str, dict] = {
    # --- flat widths (stop_lo == stop_hi makes the ATR channel inert) ---
    "flat_1p5": {"pb_stop_lo": 0.015, "pb_stop_hi": 0.015},
    "flat_2p0": {"pb_stop_lo": 0.020, "pb_stop_hi": 0.020},
    "flat_2p5": {"pb_stop_lo": 0.025, "pb_stop_hi": 0.025},   # current deployment
    "flat_3p0": {"pb_stop_lo": 0.030, "pb_stop_hi": 0.030},
    "flat_3p5": {"pb_stop_lo": 0.035, "pb_stop_hi": 0.035},
    # --- ATR-adaptive bands ---
    "atr_1p0_25_35": {"pb_atr_mult": 1.0, "pb_stop_lo": 0.025, "pb_stop_hi": 0.035},
    "atr_1p0_25_40": {"pb_atr_mult": 1.0, "pb_stop_lo": 0.025, "pb_stop_hi": 0.040},
    "atr_1p5_25_40": {"pb_atr_mult": 1.5, "pb_stop_lo": 0.025, "pb_stop_hi": 0.040},
}


def main() -> int:
    from src.cli import _build_market_for_paper, _shadow_cycle
    from src.config import load_config
    from src.paper.ledger import PaperLedger
    from src.paper.shadow import resolve_shadow_universe

    start = sys.argv[1] if len(sys.argv) > 1 else "2026-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-28"
    only = ""
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = a.split("=", 1)[1]

    cfg = load_config()
    base_account = next(
        (a for a in (cfg.get("shadow.accounts") or []) if str(a.get("name")) == "D_5W"), None
    )
    if base_account is None:
        print("ERROR: D_5W not in shadow.accounts", file=sys.stderr)
        return 2
    symbols = resolve_shadow_universe(cfg, base_account.get("universe"))

    variants = VARIANTS
    if only:
        want = {x.strip() for x in only.split(",") if x.strip()}
        variants = {k: v for k, v in variants.items() if k in want}

    print(f"[grid] window=[{start}, {end}] variants={list(variants)}", flush=True)
    market = _build_market_for_paper(cfg, symbols, start, None, seed=1)
    symbols = [s for s in symbols if s in market.price_panel.columns]
    print(f"[grid] market ready: {len(symbols)} symbols, "
          f"{len(market.price_panel.index)} bars", flush=True)

    results: dict[str, dict] = {}
    out_path = ROOT / "outputs" / "d_stop_grid.json"
    if out_path.is_file() and only:
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
        except ValueError:
            results = {}

    for label, over in variants.items():
        account = {**base_account, **over}
        ledger_path = ROOT / "outputs" / f"_dstop_{label}.sqlite"
        ledger_path.unlink(missing_ok=True)
        probe: dict = {}
        status, _ = _shadow_cycle(
            cfg, symbols, start, end, 1, skip_refresh=True, control_scale=None,
            account=account, ledger_override=str(ledger_path), write_artifacts=False,
            probe=probe, market_override=market,
        )
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
        results[label] = {
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
        }
        r = results[label]
        print(f"{label:16s}: cum={r['cum_return']:+.2%} ann={r['ann_return']:+.2%} "
              f"sharpe={r['sharpe']:+.2f} maxDD={r['max_dd']:.2%} fills={r['n_fills']} "
              f"intraday={r['n_intraday']} cost={r['cost']:,.0f}", flush=True)

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    best = max(results, key=lambda k: results[k]["sharpe"])
    print(f"\nbest by Sharpe: {best} "
          f"(sharpe={results[best]['sharpe']:.2f} ann={results[best]['ann_return']:+.2%})")
    print(f"wrote {out_path}", flush=True)
    return 0


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
