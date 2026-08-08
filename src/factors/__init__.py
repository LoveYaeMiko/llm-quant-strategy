"""Semantic factor mining — AlphaSchema + AlphaMemo + operator library.

The *exploration* (schemas) is deliberately decoupled from the *implementation*
(formulas), because AlphaSchema shows factor quality is robust to which LLM does
the translation — so we can mine cheaply (review.md §2.2).
"""
