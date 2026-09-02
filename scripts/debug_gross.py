"""Reproduce the A-shadow June/July stretch with gross instrumentation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def main() -> int:
    from src.cli import _build_market_for_paper
    from src.config import load_config
    from src.paper.ledger import PaperLedger
    from src.paper.ml_book import MLBookPortfolio
    from src.paper.runner import PaperRunner
    from src.paper.shadow import paper_runner_kwargs, resolve_shadow_universe

    cfg = load_config()
    symbols = resolve_shadow_universe(cfg, "hs300")
    market = _build_market_for_paper(cfg, symbols, "2026-01-01", None, seed=1)
    portfolio = MLBookPortfolio(
        market, ["lgbm"], long_pct=0.10, short_pct=0.10,
        max_position_pct=0.05, trend_days=60, trend_gate=0.03, short_scale=0.5,
        cfg=cfg, n_jobs=4,
    )
    print("books built:", len(portfolio._books), flush=True)

    ledger_path = ROOT / "outputs" / "_dbg_A.sqlite"
    ledger_path.unlink(missing_ok=True)
    ledger = PaperLedger(str(ledger_path))
    kw = paper_runner_kwargs(cfg)
    kw.update(cash=2_000_000.0, notional_floor=0.0, band_frac=0.0, rebalance_days=10)
    runner = PaperRunner(portfolio, market, ledger, symbols=symbols, seed=1, **kw)
    try:
        result = runner.run(start="2026-01-01", end="2026-07-14")
        m = result["metrics"]
        print("run ok:", {k: m.get(k) for k in ("n_fills", "total_return", "total_commission")}, flush=True)
    except Exception as e:
        print(f"RUN RAISED: {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        ledger.close()
        ledger_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
