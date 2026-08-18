"""Paper-trading layer — the resumable daily loop that turns the gate-passing
three-layer portfolio into a paper-trading account.

* :class:`~src.paper.ledger.PaperLedger` — persistent, resumable account state;
* :class:`~src.paper.runner.PaperRunner` — the daily walk-forward loop.
"""

from .ledger import PaperLedger
from .runner import PaperRunner

__all__ = ["PaperLedger", "PaperRunner"]
