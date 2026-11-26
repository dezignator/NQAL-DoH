#!/usr/bin/env python3
"""
experiments/figure1_architecture.py
===================================
Renders Figure 1 (logical flow / architecture) as a Graphviz SVG + PNG.

The rendered figure closely mirrors the paper's block diagram:
    datasets → hardware → NQAL defense core → secured output
with the defense core containing SNN/DSGAT, QNI, ZKP-Learn,
Swarm-FL, and Threat Immunization.
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None


def render(out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if Digraph is None:
        print("graphviz not installed; skipping figure 1 render.")
        return

    dot = Digraph(
        "NQAL", format="svg",
        graph_attr={"rankdir": "TB", "splines": "ortho", "nodesep": "0.4"},
        node_attr={"shape": "box", "style": "rounded,filled",
                   "fontname": "Helvetica", "fontsize": "10"},
    )

    # --- Datasets ---
    with dot.subgraph(name="cluster_datasets") as ds:
        ds.attr(label="Experimental Datasets", color="green", style="filled,filled",
                fillcolor="#e8f5e9")
        ds.node("d1", "CIRA-CIC-DoHBrw-2020")
        ds.node("d2", "Dahl DoH Gen-C DDD-2022")
        ds.node("d3", "doh_resolvers_2022")

    # --- Hardware ---
    with dot.subgraph(name="cluster_hw") as hw:
        hw.attr(label="Neuromorphic Hardware", color="gray",
                style="filled,filled", fillcolor="#e3f2fd")
        hw.node("hw1", "Intel Loihi 2")
        hw.node("hw2", "BrainChip Akida")
        hw.node("hw3", "Sub-1ms inference")

    # --- Defense core ---
    with dot.subgraph(name="cluster_core") as core:
        core.attr(label="NQAL Defense Core", color="#1b5e20",
                  style="filled,filled", fillcolor="#c8e6c9")
        core.node("snn", "Neuromorphic SNN\nDSGAT · STDP",
                  fillcolor="#81c784")
        core.node("qni", "Quantum Noise Injection\nAdversarial · QRW",
                  fillcolor="#90caf9")
        core.node("zkp", "ZKP-Learn\nPrivacy-preserving",
                  fillcolor="#ce93d8")
        core.node("fl", "Swarm Federated Learning\nBio-inspired · Decentralized",
                  fillcolor="#ef9a9a")
        core.node("gan", "Threat Immunization\nAdversarial GANs · Zero-day",
                  fillcolor="#ffcc80")

    # --- Output ---
    dot.node("out", "Secure DNS Output\n(99.18% acc / 0.5% FPR)",
             fillcolor="#66bb6a", fontcolor="white")

    # --- Edges ---
    for d in ("d1", "d2", "d3"):
        dot.edge(d, "hw1", style="dashed", color="gray")
        dot.edge(d, "hw2", style="dashed", color="gray")
    dot.edge("hw1", "snn", label="spikes")
    dot.edge("hw2", "snn", label="spikes")
    dot.edge("snn", "qni", label="features")
    dot.edge("qni", "zkp", label="QNI-hardened")
    dot.edge("snn", "fl", label="weights")
    dot.edge("fl", "snn", style="dashed", label="aggregated")
    dot.edge("zkp", "gan", label="anomalies")
    dot.edge("gan", "snn", style="dashed", label="immunize")
    dot.edge("zkp", "out", label="verified")
    dot.edge("snn", "out", style="dashed")

    svg_path = dot.render(
        filename="figure1_architecture",
        directory=out_dir,
        cleanup=True,
    )
    print(f"Figure 1 saved → {svg_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/results/figures")
    args = p.parse_args()
    render(args.out)
