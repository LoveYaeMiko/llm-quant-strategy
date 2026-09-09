"""Summarise every ``d_oos_*.json`` artifact in outputs/ (one line each).

Each row also reports the provenance contract status, so a number can never be
quoted without its slice/convention/commit being visible next to it (audit P-6).
``citable`` alone is not enough: an artifact whose payload no longer matches its
``artifact_sha256`` was edited after the fact and is flagged here.

    python scripts/oos_summary.py
    python scripts/oos_summary.py --json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.provenance import check_provenance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="d_oos_* artifact summary")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(str(ROOT / "outputs" / "d_oos_*.json"))):
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except ValueError:
            continue
        m = d.get("metrics", {})
        prov = check_provenance(d)
        rows.append(
            {
                "label": d.get("label"),
                "cum": m.get("total_return"),
                "ann": m.get("annualized_return"),
                "sharpe": m.get("sharpe"),
                "t": d.get("sharpe_t_stat"),
                "dd": m.get("max_drawdown"),
                "fills": m.get("n_fills"),
                "mincov": d.get("minute_min_symbol_coverage"),
                "symwin": d.get("minute_min_symbol_tradable_coverage"),
                "below90": d.get("n_symbols_below_90pct"),
                "hole": d.get("fills_in_low_coverage_days"),
                "citable": d.get("citable"),
                "candidate": d.get("is_candidate_run"),
                "prov": "ok" if prov["ok"] else "MISSING",
                "commit": (prov["values"].get("code_commit") or "")[:12],
                "data_as_of": prov["values"].get("data_as_of"),
            }
        )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return 0
    head = (f"{'label':30s} {'cum':>9s} {'ann':>9s} {'sharpe':>7s} {'t':>6s} {'dd':>7s} "
            f"{'fills':>6s} {'mincov':>7s} {'symwin':>7s} {'<90%':>5s} {'hole':>5s} "
            f"{'citable':>8s} {'prov':>8s} {'commit':>12s}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{str(r['label'])[:30]:30s} "
            f"{(r['cum'] or 0):+9.2%} {(r['ann'] or 0):+9.2%} "
            f"{(r['sharpe'] or 0):+7.2f} {(r['t'] or 0):+6.2f} "
            f"{(r['dd'] or 0):7.2%} {int(r['fills'] or 0):6d} "
            f"{(r['mincov'] or 0):7.2%} {(r['symwin'] or 0):7.2%} "
            f"{int(r['below90'] if r['below90'] is not None else -1):5d} "
            f"{int(r['hole'] or 0):5d} {str(r['citable']):>8s} {r['prov']:>8s} {r['commit']:>12s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
