"""C2 — structured daily digest + webhook notification scheduling.

The factor-mining loop runs unattended for hours; the researcher should not have
to tail logs to learn what survived the gates today. This package turns the raw
run state (accepted pool, checklist verdicts, cost ledger) into a deterministic
four-section markdown digest, and pushes it to a webhook (飞书 / DingTalk
compatible) when one is configured.

Both halves are deliberately dependency-free (``urllib`` for the webhook, plain
string formatting for the digest) so reporting never becomes a setup burden.
"""

from .digest import WebhookNotifier, build_digest, build_notifier

__all__ = ["WebhookNotifier", "build_digest", "build_notifier"]
