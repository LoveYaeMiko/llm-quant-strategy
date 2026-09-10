# Execution invariants

Scope: the two order entry points in `src/online/order_executor.py` —
`execute_orders` (14:50 pre-submitted list, filled at the 15:00 closing auction)
and `execute` (daily close rebalance from target weights, planned by
`_plan_delta`).

## The long-only invariant

**The sum of sells for a symbol never exceeds the shares held when the sell pass
started. A sell with nothing left to cover is skipped and reported, never filled
as a short.**

In `execute_orders` the availability is tracked *cumulatively* over the sell
pass: the first sell line consumes what it fills and every later line of the
same symbol sees only the remainder.

## Why it matters

This account type cannot short A-shares. An uncovered sell therefore does not
"fail" loudly — it silently creates a negative position, which corrupts the book
in three ways that are hard to see after the fact:

* the ledger records a position the account could never hold (and the short's
  mark-to-market then drives equity with an inverted sign);
* the sell's cash proceeds are booked as if they were real inventory sales;
* `gross_exposure` (and every report built on it) silently counts the phantom
  short as exposure.

The trigger is mundane: the 14:50 order list and the persisted account state
disagreeing — a duplicated sell line, a stale list executed against a fresh
ledger, a holding already reduced by an intraday stop.

## Where it is enforced

| Path | Enforcement |
| --- | --- |
| `execute_orders` (auction, `source="auction"`) | **Always.** Cumulative clip against `self.positions`, snapshotted once at the start of the sell pass. |
| `execute` (close rebalance, `source="close"`) | **Only when `OrderExecutor(long_only=True)`.** Off by default. |
| `_plan_delta` | Never clips. It can still construct a short leg for a long/short book. |

## What `skipped` means for a reviewer

`OrderResult.skipped` is a JSON-serialisable list of
`{"symbol", "reason", "wanted", "clipped"}` (signed share quantities) and is
exposed by `OrderResult.to_dict()`. It is empty on a clean execution.

* `reason="sell_without_holding"` — nothing was coverable: the order is a
  mismatch between the submitted list and the account state, not a trade.
* `reason="sell_exceeds_holding"` — partially clipped; the filled part is real,
  the `wanted - clipped` remainder is not.

A non-empty `skipped` list is never "just noise": it means the order generator
and the ledger disagreed, and it should be investigated before the next session.

## Known residual (deliberate)

`_plan_delta` can still build a short leg, and `execute` will fill it unless
`long_only=True`. That is intentional: the retired long/short factor books
(A/B/C tracks) depend on short semantics, and the D-track paper account is the
only live book. Enabling the clip is an explicit, per-instance opt-in so retiring
those books cannot silently change their historical behaviour.
