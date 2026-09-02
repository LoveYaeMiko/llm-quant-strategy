"""Extract the alpha101 + gtja191 formula inventory from paper/repos/aurumq-rl.

Parses the per-factor markdown docs into ``paper/factor_zoo/inventory.json``:
``{"alpha101": [...], "gtja191": [...]}`` where each entry carries the id,
family (category), direction, quality flag, the original WorldQuant/GTJA
formula and the legacy AQML prefix expression (the closest thing to FQA's own
grammar — the input to the translation step).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "paper" / "repos" / "aurumq-rl"
OUT = Path(__file__).resolve().parents[1] / "paper" / "factor_zoo"

_TITLE = re.compile(r"^#\s+(\S+)\s*—\s*(.*)$", re.M)
_META = re.compile(r"\*\*(Category|Direction|Quality)\*\*:\s*([^*\n]+)", re.M)


def _section(text: str, heading: str) -> str | None:
    # `.*?\n` (not `\s*\n`) — headings may carry suffixes like "(deprecated)"
    m = re.search(rf"^##\s+{re.escape(heading)}.*?\n(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    if not m:
        return None
    body = m.group(1).strip()
    for fence in ("```",):
        body = body.replace(fence, "")
    # keep only the first paragraph: AQML sections repeat the expression once
    # as prose and once inside a code fence — the duplicate would parse as
    # trailing tokens
    body = body.split("\n\n")[0].strip()
    if not body or body.startswith("_") and body.endswith("_") and len(body) < 40:
        return None
    return body


_GTJA_FUNC = re.compile(r"def\s+(gtja_\d+)\(panel[^)]*\)\s*->[^:]*:\s*\"{3}(.*?)\"{3}", re.S)
_GTJA_FORMULA = re.compile(
    r"Guotai Junan Formula\s*\n\s*-+\s*\n"
    r"(?P<formula>(?:(?![ \t]{4}\S)[^\n]*\n?)*)",
    re.S,
)

#: Daic115/alpha191.py — one function per alpha, the ORIGINAL formula in the docstring
_A191_FUNC = re.compile(
    r"def\s+alpha191_(\d+)\(data[^)]*\)\s*:\s*\"{3}\s*\n"
    r"(?P<formula>.*?)\n\s*\"{3}",
    re.S,
)


def _parse_daic115(path: Path) -> dict[str, str]:
    """Extract ``{gtja_id: formula}`` from the Daic115 alpha191.py docstrings."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for m in _A191_FUNC.finditer(text):
        num, doc = m.group(1), m.group(2)
        lines = [ln.strip() for ln in doc.splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#")]
        if lines:
            out[f"gtja_{num}"] = " ".join(lines)
    return out


def _parse_gtja_source(path: Path) -> dict[str, str]:
    """Extract ``{gtja_id: formula}`` from the polars implementation docstrings.

    The formula block is the indented (8-space) run under the heading; capture
    stops at the first 4-space prose line ("Required panel columns:", the
    docstring body indent).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for m in _GTJA_FUNC.finditer(text):
        fid, doc = m.group(1), m.group(2)
        fm = _GTJA_FORMULA.search(doc)
        if not fm:
            continue
        lines = [ln.strip() for ln in fm.group("formula").splitlines()]
        lines = [ln for ln in lines if ln]
        if lines:
            out[fid] = " ".join(lines)
    return out


def _parse_doc(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    tm = _TITLE.search(text)
    if not tm:
        return None
    fid, desc = tm.group(1), tm.group(2).strip()
    meta = {m.group(1).lower(): m.group(2).strip().rstrip("|").strip() for m in _META.finditer(text)}
    original = _section(text, "Original WorldQuant Formula")
    aqml = _section(text, "Legacy AQML Expression")
    if aqml and aqml.lower().startswith("_pure-callable"):
        aqml = None
    return {
        "id": fid,
        "desc": desc,
        "family": meta.get("category", ""),
        "direction": meta.get("direction", ""),
        "quality": meta.get("quality", ""),
        "original": original,
        "aqml": aqml,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for family, pattern in (("alpha101", "alpha*.md"), ("gtja191", "gtja_*.md")):
        docs = sorted((REPO / "docs" / "factor_library" / family).glob(pattern))
        entries = [e for e in (_parse_doc(p) for p in docs) if e]
        # GTJA191 formulas live in the polars implementation docstrings, not the
        # markdown (whose "Original Formula" is "(not specified)")
        if family == "gtja191":
            src_formulas: dict[str, str] = {}
            for src in (REPO / "src" / "aurumq_rl" / "factors" / "gtja191").glob("batch_*.py"):
                src_formulas.update(_parse_gtja_source(src))
            for e in entries:
                if e["original"] is None and e["id"] in src_formulas:
                    e["original"] = src_formulas[e["id"]]
            # second source (Daic115/alpha191.py) fills the REMAINING gaps only —
            # never overwrites the aurumq originals (its dialect is rougher)
            ref = Path(__file__).resolve().parents[1] / "paper" / "repos" / "gtja191-reference" / "alpha191.py"
            if ref.is_file():
                daic = _parse_daic115(ref)
                for e in entries:
                    if e["original"] is None and e["id"] in daic:
                        e["original"] = daic[e["id"]]
        inventory[family] = entries
        summary[family] = {
            "docs": len(docs),
            "parsed": len(entries),
            "with_original": sum(1 for e in entries if e["original"]),
            "with_aqml": sum(1 for e in entries if e["aqml"]),
            "families": sorted({e["family"] for e in entries if e["family"]}),
        }
    out = OUT / "inventory.json"
    out.write_text(
        json.dumps({"summary": summary, **inventory}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
