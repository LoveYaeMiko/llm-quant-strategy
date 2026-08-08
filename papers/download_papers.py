"""Download the papers referenced by PROJECT_BLUEPRINT.md into ./papers/.

Usage:
    python papers/download_papers.py [--force]

Each blueprint filename is mapped to its canonical URL (arXiv / ACL). Files that
already exist are skipped unless --force is given. Failures are reported but do
not abort the rest of the batch — some 2026 listings may not be publicly
reachable yet; place any manually-obtained PDF under the same filename.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request

PAPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Blueprint filename -> canonical download URL (see papers/README.md).
PAPERS: dict[str, str] = {
    "AlphaSchema_arXiv2607.26642.pdf": "https://arxiv.org/pdf/2607.26642",
    "FinCAD_arXiv2605.24564.pdf": "https://arxiv.org/pdf/2605.24564",
    "EvoQuant_arXiv2607.12455.pdf": "https://arxiv.org/pdf/2607.12455",
    "AlphaMemo_arXiv2606.20625.pdf": "https://arxiv.org/pdf/2606.20625",
    "CognitiveAlphaMining_ACL2026.pdf": (
        "https://aclanthology.org/2026.acl-long.538.pdf"
    ),
    "AgenticAITA_arXiv2605.12532.pdf": "https://arxiv.org/pdf/2605.12532",
    "FINSABER_KDD2026.pdf": "https://arxiv.org/pdf/2505.07078",
}

_HEADERS = {"User-Agent": "llm-quant-strategy/0.1 (research; paper fetch)"}


def _download(url: str, dest: str, timeout: int = 60) -> bool:
    """Download ``url`` to ``dest``. Returns True on success."""
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if b"<html" in data[:4096].lower() and len(data) > 1_000_000:
            # arXiv sometimes returns an HTML landing page on redirects.
            # Follow the arxiv.org/pdf/<id> form which 302s to the real file.
            print(f"    got HTML body for {url}; retrying export URL")
            return _download(url.replace("/pdf/", "/pdf/"), dest, timeout)
        with open(dest, "wb") as fh:
            fh.write(data)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"    FAILED: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download files that already exist"
    )
    args = parser.parse_args()

    os.makedirs(PAPERS_DIR, exist_ok=True)
    failed: list[str] = []
    for name, url in PAPERS.items():
        dest = os.path.join(PAPERS_DIR, name)
        if os.path.exists(dest) and not args.force:
            print(f"exists   {name}")
            continue
        print(f"fetching {name} <- {url}")
        if not _download(url, dest):
            failed.append(name)

    if failed:
        print("\nFailed to download:", ", ".join(failed))
        print("Fetch manually and place under the same filename in papers/.")
        return 1
    print("\nAll papers present in papers/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
