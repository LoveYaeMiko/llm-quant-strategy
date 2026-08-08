"""Online execution — the deterministic side of the TiMi split.

The offline layer can be LLM-heavy; this package must be pure NumPy/Pandas.
:data:`NO_HEAVY_IMPORTS` and :func:`assert_no_heavy_imports` enforce that:
``torch`` / ``transformers`` are forbidden here.
"""
