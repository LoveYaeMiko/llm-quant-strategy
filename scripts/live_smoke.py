"""Pre-open smoke test for the D-track real-time trader (read-only).

Replicates ``cli.py live`` up to (but NOT including) ``LiveTrader.run()`` so the
whole assembly can be verified before 09:25 without leaving a long-lived
process or writing anything:

1. config → account → universe → market (same ``_build_market_for_paper`` slice);
2. ledger + ``_build_account_portfolio`` (same ML scanner cache, intraday feature
   pack, minute provider, kill-switch wiring);
3. ``AlphaFeedAdapter`` reachability + latest minute print per held symbol;
4. the exact decision path: ``_filter_quotes`` (staleness / limit-down guards)
   then ``portfolio.live_check`` (stop breaches).

Pre-market the print is yesterday's close, so the staleness guard MUST block it —
that is the expected output and proves the guard works. During the session the
same call returns a fresh print and evaluates the stop.

Usage::

    python scripts/live_smoke.py            # read-only, never writes
    python scripts/live_smoke.py --account D_5W
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="D-track live pre-open smoke test")
    ap.add_argument("--account", default="")
    args = ap.parse_args()

    from src.cli import _build_account_portfolio, _build_market_for_paper
    from src.config import load_config
    from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter
    from src.live import LiveTrader
    from src.paper import PaperLedger
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    lcfg = cfg.section("live") or {}
    if not bool(lcfg.get("enabled", True)):
        print("[smoke] live.enabled=false — the trader would not run")
        return 1
    name = args.account or str(lcfg.get("account", "D_5W"))
    shadow = cfg.section("shadow")
    account = next((a for a in shadow.get("accounts", []) if a.get("name") == name), None)
    if account is None:
        print(f"[smoke] FAIL: account {name!r} not in shadow.accounts")
        return 1
    if str(account.get("alpha_source", "")) != "pullback":
        print(f"[smoke] FAIL: alpha_source={account.get('alpha_source')!r} (pullback required)")
        return 1

    symbols = resolve_shadow_universe(cfg, account.get("universe"))
    market = _build_market_for_paper(
        cfg, symbols, str(shadow.get("start_date", "2026-01-01")), None, seed=1
    )
    symbols = [s for s in symbols if s in market.price_panel.columns]
    base = str(shadow.get("ledger_db", "outputs/shadow_ledger.sqlite"))
    ledger = PaperLedger(str(ROOT / base.replace(".sqlite", f"_{name}.sqlite")))
    portfolio, _ = _build_account_portfolio(cfg, market, symbols, account, ledger=ledger)
    adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))
    trader = LiveTrader(cfg, portfolio, ledger, account, adapter)

    print(f"[smoke] account={name} universe={len(symbols)} "
          f"poll={trader.poll_seconds}s max_quote_age={trader.max_quote_age_minutes}min")
    print(f"[smoke] deployment={trader.deployment['mode']} "
          f"(real_money_enabled={trader.deployment['real_money_enabled']})")
    print(f"[smoke] positions={json.dumps(trader.positions)}")

    now = datetime.now()
    in_hours = trader._in_trading_hours(now) if hasattr(trader, "_in_trading_hours") else None
    from src.live.trader import _in_trading_hours

    print(f"[smoke] now={now:%Y-%m-%d %H:%M:%S} in_decision_window={_in_trading_hours(now)}")

    if not trader.positions:
        print("[smoke] no open position — nothing to poll (entry happens at the close)")
        return 0

    try:
        raw = trader._current_prices()
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] FAIL: minute poll raised {type(exc).__name__}: {exc}")
        return 1
    print(f"[smoke] minute prints: {json.dumps({k: [v[0], str(v[1])] for k, v in raw.items()})}")

    prices = trader._filter_quotes(raw, now)
    print(f"[smoke] decision prints={json.dumps(prices)} blocked={json.dumps(trader._quote_blocks)}")

    exits = portfolio.live_check(prices, now)
    print(f"[smoke] stop breaches={json.dumps(exits)}")
    for sym, lot in portfolio._open.items():
        print(f"[smoke] lot {sym}: entry={lot.entry_price:.2f} stop={lot.stop:.2f} "
              f"dist={lot.stop_dist:.4f} trail={lot.trail_active}")

    ok = bool(raw) and (bool(prices) or bool(trader._quote_blocks))
    print(f"[smoke] {'OK' if ok else 'FAIL'} — API reachable and the guard path executed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
