"""D-track ATR defect impact — flat 2.5% stop vs ATR-adaptive stop (A/B).

Context (2026-09-08 audit, defect D-8b): ``PullbackPortfolio`` built its true
range with ``pd.concat([...], axis=1).max(axis=1)``, which collapses the
(date × symbol) frame to a per-DATE Series. ``_atr20 / close_wide`` then aligned
a date-indexed Series against symbol columns, so every ``_atr_pct.loc[d, symbol]``
lookup returned NaN and ``_stop_dist`` fell back to its 2.5% floor for the ENTIRE
track — the ATR-adaptive band [2.5%, 4%] never engaged. The fix keeps the frame
(``np.maximum.reduce``).

This script measures the impact on the DEPLOYED D configuration by running the
production cycle twice with fresh ledgers:

* ``flat_2p5``    — ``pb_stop_hi = 0.025``: exactly what the defect produced;
* ``atr_adaptive``— deployed ``pb_stop_lo=0.025 / pb_stop_hi=0.04``.

Both go through ``src.cli._shadow_cycle`` → ``_build_account_portfolio``, i.e.
the same assembly as the 15:10 production job (same ML scanner cache, same
intraday feature pack, same minute provider, same preclose gate). Research only —
it never touches the production ledger and writes no status/report artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from src.cli import _shadow_cycle
    from src.config import load_config
    from src.paper.ledger import PaperLedger
    from src.paper.shadow import resolve_shadow_universe

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = ""
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = a.split("=", 1)[1]
    start = args[0] if len(args) > 0 else "2026-01-01"
    end = args[1] if len(args) > 1 else None

    cfg = load_config()
    base_account = next(
        (a for a in (cfg.get("shadow.accounts") or []) if str(a.get("name")) == "D_5W"), None
    )
    if base_account is None:
        print("ERROR: D_5W not in shadow.accounts", file=sys.stderr)
        return 2
    symbols = resolve_shadow_universe(cfg, base_account.get("universe"))

    flat = {"pb_stop_lo": 0.025, "pb_stop_hi": 0.025}
    atr = {"pb_stop_lo": 0.025, "pb_stop_hi": 0.04}
    variants = {
        # control: no intraday stops at all (close-only), flat stop width — this
        # is grid7's D_close_only shape and must reproduce its ~+33% to prove the
        # harness is unchanged.
        "no_intraday": {**base_account, **flat, "pb_intraday_stops": False},
        # flat 2.5% stop: the width the buggy one-dimensional TR silently forced
        # on EVERY day, so this variant ≈ the pre-fix production behaviour.
        "flat_2p5": {**base_account, **flat},
        # deployed ATR-adaptive stop (2.5–4%) with the two-dimensional TR fix.
        "atr_adaptive": {**base_account, **atr},
    }
    if only:
        want = {x.strip() for x in only.split(",") if x.strip()}
        variants = {k: v for k, v in variants.items() if k in want}

    results: dict[str, dict] = {}
    out_path = ROOT / "outputs" / "d_atr_impact.json"
    if out_path.is_file() and only:
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
        except ValueError:
            results = {}
    for label, account in variants.items():
        ledger_path = ROOT / "outputs" / f"_datr_{label}.sqlite"
        ledger_path.unlink(missing_ok=True)
        print(f"\n=== {label} (stop_lo={account['pb_stop_lo']} stop_hi={account['pb_stop_hi']} "
              f"intraday={account.get('pb_intraday_stops', True)}) ===", flush=True)
        probe: dict = {}
        status, _ = _shadow_cycle(
            cfg, symbols, start, end, 1, skip_refresh=True, control_scale=None,
            account=account, ledger_override=str(ledger_path), write_artifacts=False,
            probe=probe,
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
        results[label] = {
            "stop_lo": account["pb_stop_lo"],
            "stop_hi": account["pb_stop_hi"],
            "intraday_stops": bool(account.get("pb_intraday_stops", True)),
            "cum_return": eq.get("total_return", 0.0),
            "ann_return": eq.get("annualized_return", 0.0),
            "sharpe": eq.get("sharpe", 0.0),
            "max_dd": eq.get("max_drawdown", 0.0),
            "n_days": eq.get("n_days", 0),
            "n_fills": eq.get("n_fills", 0),
            "n_intraday": n_intraday,
            "cost": eq.get("total_commission", 0.0),
            "fills_by_source": by_source,
            "fingerprint": probe,
        }
        r = results[label]
        print(f"{label:14s}: cum={r['cum_return']:+.2%} ann={r['ann_return']:+.2%} "
              f"sharpe={r['sharpe']:+.2f} maxDD={r['max_dd']:.2%} fills={r['n_fills']} "
              f"intraday={r['n_intraday']} cost={r['cost']:,.0f}", flush=True)

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
