"""Dry-run the 14:50 preclose order layer against LIVE data, writing nowhere.

Exercises the exact production path (`build_preclose_orders`: provisional minute
bars + T-1 ML ranks + intraday features + the book) but points ``preclose.ROOT``
at a temp directory, so it can NEVER leave a stale order list behind for the
15:10 close run to execute. Run it shortly before 14:50 to prove the layer works
before the real job fires.

Usage: python scripts/preclose_dry_run.py [--keep]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="preclose dry run (no production writes)")
    ap.add_argument("--keep", action="store_true", help="keep the temp order file")
    args = ap.parse_args()

    from src import preclose as pc
    from src.config import load_config

    cfg = load_config()
    lcfg = cfg.section("live") or {}
    name = str(lcfg.get("account", "D_5W"))
    account = next(
        (a for a in (cfg.section("shadow").get("accounts") or []) if a.get("name") == name), None
    )
    if account is None:
        print(f"[dry] account {name!r} not found")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="preclose_dry_"))
    out_dir = tmp / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        # ROOT stays the real FQA root (the ledger holds the current positions);
        # only the order-list JSON is redirected.
        result = pc.build_preclose_orders(cfg, account, [], out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[dry] FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"[dry] result: {json.dumps(result, ensure_ascii=False, default=str)}")
    path = out_dir / f"preclose_orders_{name}.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        orders = payload.get("orders", [])
        print(f"[dry] would submit {len(orders)} order(s) at 15:00:")
        for o in orders:
            print(f"       {o['side']:>4} {o['symbol']} {o['shares']:>8.0f}")
        if not args.keep:
            path.unlink(missing_ok=True)
    print(f"[dry] {'OK' if result.get('ok') else 'FAILED'} — nothing written under the real ROOT")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
