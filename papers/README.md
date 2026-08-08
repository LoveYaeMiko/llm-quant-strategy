# Papers

This folder holds the 2026 SOTA papers that drive the system design. The code in
`src/` reads these for the exact algorithms (FinCAD context-aware decoding,
AlphaSchema schema space, EvoQuant validation pipeline, FINSABER bias traps).

| Blueprint filename | Paper | Source |
|---|---|---|
| `AlphaSchema_arXiv2607.26642.pdf` | AlphaSchema — trading semantics space | https://arxiv.org/abs/2607.26642 |
| `FinCAD_arXiv2605.24564.pdf` | FinCAD — context-aware decoding (look-ahead suppression) | https://arxiv.org/abs/2605.24564 |
| `EvoQuant_arXiv2607.12455.pdf` | EvoQuant — verifier-guided strategy optimization | https://arxiv.org/abs/2607.12455 |
| `AlphaMemo_arXiv2606.20625.pdf` | AlphaMemo — structured search memory | https://arxiv.org/abs/2606.20625 |
| `CognitiveAlphaMining_ACL2026.pdf` | Cognitive Alpha Mining (CogAlpha) | https://aclanthology.org/2026.acl-long.538 |
| `AgenticAITA_arXiv2605.12532.pdf` | AgenticAITA — autonomous deliberative loop | https://arxiv.org/abs/2605.12532 |
| `FINSABER_KDD2026.pdf` | FINSABER — LLM investing pitfalls / bias traps | https://arxiv.org/abs/2505.07078 |

## Download

```bash
python papers/download_papers.py
```

The script maps each filename to its canonical URL, downloads into `papers/`, skips
files that already exist, and reports any that failed (some listings may be
unreachable). arXiv PDFs are also fetchable manually at
`https://arxiv.org/pdf/<id>`.

> **Note on reproducibility.** arXiv identifiers in this blueprint carry a `26xx`
> prefix (2026). If a given ID does not resolve, the paper may not be publicly
> indexed yet — place the PDF you obtained from the source link under the same
> filename so the loader finds it.
