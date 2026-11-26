#!/usr/bin/env python3
"""
experiments/table1_comparison.py
================================
Renders Table 1 of the paper (comparative analysis of prior DoH detection
work) plus our row, and writes both Markdown and CSV.

Usage
-----
    python experiments/table1_comparison.py --out artifacts/results/tables
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

TABLE1 = [
    {
        "Study": "Houser et al.",
        "Methodology": "Decision trees on flow-level features",
        "Key contribution": "Early classification model to detect DoH traffic",
        "Identified limitations": "Limited adaptability to emerging patterns; static feature set",
    },
    {
        "Study": "Böttger et al.",
        "Methodology": "Packet size and timing fingerprinting",
        "Key contribution": "Behavioral recognition of DoH sessions in controlled environments",
        "Identified limitations": "Susceptible to traffic padding and timing obfuscation",
    },
    {
        "Study": "Lin et al.",
        "Methodology": "CNN-based encrypted traffic classification",
        "Key contribution": "Deep learning applied to infer encrypted DNS queries",
        "Identified limitations": "Low interpretability; black-box; high compute",
    },
    {
        "Study": "Pearce et al.",
        "Methodology": "Entropy-based detection of DNS tunneling",
        "Key contribution": "Targeted identification of malicious behavior within DoH",
        "Identified limitations": "High false positives; centralised and privacy-invasive",
    },
    {
        "Study": "Wang et al.",
        "Methodology": "Temporal correlation with browsing activity",
        "Key contribution": "Sequence modeling of encrypted user behavior",
        "Identified limitations": "Privacy concerns; relies on labeled data; poor scalability",
    },
    {
        "Study": "Our Work (NQAL)",
        "Methodology": "SNN + federated learning + quantum entropy filters",
        "Key contribution": "First modular, privacy-preserving, scalable DoH threat framework",
        "Identified limitations": "N/A — addresses prior limitations",
    },
]


def to_markdown(rows) -> str:
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/results/tables")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    md = to_markdown(TABLE1)
    (Path(args.out) / "table1.md").write_text(md)

    with open(Path(args.out) / "table1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE1[0].keys())
        w.writeheader()
        w.writerows(TABLE1)

    print(md)
    print(f"\nSaved → {args.out}/table1.{{md,csv}}")


if __name__ == "__main__":
    main()
