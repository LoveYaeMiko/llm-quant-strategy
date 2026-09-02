"""Extract qlib Alpha158/Alpha360 default features → translate to FQA grammar.

Replicates ``qlib/contrib/data/loader.py``'s ``get_feature_config`` default
(kbar + price windows [0] + volume windows [0] + rolling windows
[5,10,20,30,60] with all 29 operators) WITHOUT importing qlib — the
expressions are pure string templates (source: microsoft/qlib, MIT; a copy is
kept under ``paper/repos/qlib_extract/loader.py``).

Output: appends an ``alpha158`` family to ``paper/factor_zoo/translated.json``
and updates ``paper/factor_zoo/inventory.json`` with the raw expressions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.exploration.translate import translate  # noqa: E402

_WINDOWS = [5, 10, 20, 30, 60]


def _build_default_fields() -> list[tuple[str, str]]:
    """(expression, name) pairs — mirrors Alpha158DL.get_feature_config defaults."""
    fields: list[tuple[str, str]] = []

    # kbar (9 hard-coded features)
    kbar = [
        "($close-$open)/$open", "($high-$low)/$open",
        "($close-$open)/($high-$low+1e-12)",
        "($high-Greater($open, $close))/$open",
        "($high-Greater($open, $close))/($high-$low+1e-12)",
        "(Less($open, $close)-$low)/$open",
        "(Less($open, $close)-$low)/($high-$low+1e-12)",
        "(2*$close-$high-$low)/$open",
        "(2*$close-$high-$low)/($high-$low+1e-12)",
    ]
    names = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
    fields += list(zip(kbar, names))

    # price windows [0] (OPEN/HIGH/LOW/VWAP relative to close)
    for f in ("open", "high", "low", "vwap"):
        fields.append((f"${f}/$close", f.upper() + "0"))
    # volume windows [0]
    fields.append(("$volume/($volume+1e-12)", "VOLUME0"))

    # rolling windows, all operators (include=None, exclude=[])
    d = "{%d}"  # noqa: F841  (readability only)
    for w in _WINDOWS:
        wf = [
            (f"Ref($close, {w})/$close", f"ROC{w}"),
            (f"Mean($close, {w})/$close", f"MA{w}"),
            (f"Std($close, {w})/$close", f"STD{w}"),
            (f"Slope($close, {w})/$close", f"BETA{w}"),
            (f"Rsquare($close, {w})", f"RSQR{w}"),
            (f"Resi($close, {w})/$close", f"RESI{w}"),
            (f"Max($high, {w})/$close", f"MAX{w}"),
            (f"Min($low, {w})/$close", f"MIN{w}"),
            (f"Quantile($close, {w}, 0.8)/$close", f"QTLU{w}"),
            (f"Quantile($close, {w}, 0.2)/$close", f"QTLD{w}"),
            (f"Rank($close, {w})", f"RANK{w}"),
            (f"($close-Min($low, {w}))/(Max($high, {w})-Min($low, {w})+1e-12)", f"RSV{w}"),
            (f"IdxMax($high, {w})/{w}", f"IMAX{w}"),
            (f"IdxMin($low, {w})/{w}", f"IMIN{w}"),
            (f"(IdxMax($high, {w})-IdxMin($low, {w}))/{w}", f"IMXD{w}"),
            (f"Corr($close, Log($volume+1), {w})", f"CORR{w}"),
            (f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {w})", f"CORD{w}"),
            (f"Mean($close>Ref($close, 1), {w})", f"CNTP{w}"),
            (f"Mean($close<Ref($close, 1), {w})", f"CNTN{w}"),
            (f"Mean($close>Ref($close, 1), {w})-Mean($close<Ref($close, 1), {w})", f"CNTD{w}"),
            (f"Sum(Greater($close-Ref($close, 1), 0), {w})/(Sum(Abs($close-Ref($close, 1)), {w})+1e-12)", f"SUMP{w}"),
            (f"Sum(Greater(Ref($close, 1)-$close, 0), {w})/(Sum(Abs($close-Ref($close, 1)), {w})+1e-12)", f"SUMN{w}"),
            (f"(Sum(Greater($close-Ref($close, 1), 0), {w})-Sum(Greater(Ref($close, 1)-$close, 0), {w}))"
             f"/(Sum(Abs($close-Ref($close, 1)), {w})+1e-12)", f"SUMD{w}"),
            (f"Mean($volume, {w})/($volume+1e-12)", f"VMA{w}"),
            (f"Std($volume, {w})/($volume+1e-12)", f"VSTD{w}"),
            (f"Std(Abs($close/Ref($close, 1)-1)*$volume, {w})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {w})+1e-12)", f"WVMA{w}"),
            (f"Sum(Greater($volume-Ref($volume, 1), 0), {w})/(Sum(Abs($volume-Ref($volume, 1)), {w})+1e-12)", f"VSUMP{w}"),
            (f"Sum(Greater(Ref($volume, 1)-$volume, 0), {w})/(Sum(Abs($volume-Ref($volume, 1)), {w})+1e-12)", f"VSUMN{w}"),
            (f"(Sum(Greater($volume-Ref($volume, 1), 0), {w})-Sum(Greater(Ref($volume, 1)-$volume, 0), {w}))"
             f"/(Sum(Abs($volume-Ref($volume, 1)), {w})+1e-12)", f"VSUMD{w}"),
        ]
        fields += wf
    return fields


def main() -> int:
    fields = _build_default_fields()
    print(f"alpha158 features: {len(fields)}")

    entries = [
        {"id": name, "family": "alpha158", "original": expr}
        for expr, name in fields
    ]
    inv_path = ROOT / "paper" / "factor_zoo" / "inventory.json"
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    inv["alpha158"] = entries
    inv_path.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")

    results = {}
    counts = {"ok": 0, "deferred": 0}
    for e in entries:
        t = translate(e["original"])
        if t.status == "ok":
            results[e["id"]] = {"status": "ok", "fqa": t.fqa}
            counts["ok"] += 1
        else:
            results[e["id"]] = {"status": "deferred", "reasons": t.reasons}
            counts["deferred"] += 1

    tr_path = ROOT / "paper" / "factor_zoo" / "translated.json"
    payload = json.loads(tr_path.read_text(encoding="utf-8"))
    payload["alpha158"] = results
    payload["summary"]["alpha158"] = counts
    tr_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("alpha158 translation:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
