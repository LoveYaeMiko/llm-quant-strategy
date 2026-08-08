"""LLM-driven quantitative trading system.

Strict TiMi separation (review.md §3, blueprint §1):
  - **Offline R&D**   : LLM-heavy — factor mining, schema exploration, EvoQuant
                        self-evolution. Anything that runs a model lives here.
  - **Online execution**: deterministic — pure NumPy/Pandas signal computation,
                        PCA-neutralised portfolio construction, order execution.
                        Zero `torch` / `transformers` imports.

Bias-control is the cross-cutting concern: `point_in_time_loader` prevents
look-ahead at the data layer, and `bias_control.look_ahead_detector` suppresses
look-ahead at the *inference* layer (FinCAD context-aware decoding).
"""

__version__ = "0.1.0"
