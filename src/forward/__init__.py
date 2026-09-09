"""Forward-period tooling: pre-registration, the risk gate, candidate evaluation.

The forward period is not an alpha test (the window can never have the power for
one — see :mod:`src.forward.risk_gate` and ``docs/FORWARD_PROTOCOL.md``). It is a
*trust* test of the pipeline: does the deployed assembly do exactly what the
frozen specification says, at an acceptable cost, without operational failure.
"""

from . import prereg, risk_gate  # noqa: F401

__all__ = ["prereg", "risk_gate"]
