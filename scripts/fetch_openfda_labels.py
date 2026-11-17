"""Download the drug labels locally (fallback if Databricks compute cannot reach openFDA).

Usage (from the project folder):
    python scripts/fetch_openfda_labels.py
Then upload data/drug_labels.jsonl to the Unity Catalog volume named in config.py.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DRUG_BRANDS  # noqa: E402
from drug_rag_core import fetch_labels  # noqa: E402


def main() -> None:
    labels, missing = fetch_labels(DRUG_BRANDS)
    out = ROOT / "data" / "drug_labels.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in labels:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(labels)} labels to {out}")
    if missing:
        print(f"No exact label found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
