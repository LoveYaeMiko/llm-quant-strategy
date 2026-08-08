"""Bias control — look-ahead prevention at the *inference* layer.

FinCAD (blueprint Phase 1 / review.md §2.4): while the point-in-time loader
stops future data reaching the model's inputs, a pretrained LLM may *recall*
future events from memory. These modules penalise tokens that embed a date after
the current timestamp during generation, so reasoning stays grounded in
information available at T.
"""
