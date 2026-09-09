"""Paper-trading layer — the resumable daily loop that turns the gate-passing
three-layer portfolio into a paper-trading account.

* :class:`~src.paper.ledger.PaperLedger` — persistent, resumable account state;
* :class:`~src.paper.runner.PaperRunner` — the daily walk-forward loop;
* :func:`~src.paper.ledger.clone_ledger_before` — fork an account state for a
  candidate shadow / replay.
"""

from .ledger import PaperLedger, clone_ledger_before
from .runner import PaperRunner

__all__ = ["PaperLedger", "PaperRunner", "clone_ledger_before"]
