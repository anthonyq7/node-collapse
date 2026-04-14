"""
Compares citation rates for seed papers vs. LLM-generated papers at each node.

seed_citation_rate  = total citations to SEED_ papers / total exposures of SEED_ papers
llm_citation_rate   = total citations to N*P* papers  / total exposures of N*P* papers

Uses node_N_exposure.jsonl, which records per-paper citations and exposures
within that node's generation.
"""

import json
import os
import matplotlib.pyplot as plt

BASE = "archived_synthetic/alpha_0.2.0_scaled"
MAX_NODES = 12
OUTPUT_ROOT = f"{BASE}/output"


def load_exposure(node: int, data_root: str) -> list[dict]:
    path = f"{data_root}/node_{node}/node_{node}_exposure.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_rates(entries: list[dict]) -> dict:
    seed_citations = seed_exposures = 0
    llm_citations = llm_exposures = 0

    for e in entries:
        pid = e["id"]
        c, ex = e["citations"], e["exposures"]
        if pid.startswith("SEED_"):
            seed_citations += c
            seed_exposures += ex
        else:
            llm_citations += c
            llm_exposures += ex

    return {
        "seed_citations": seed_citations,
        "seed_exposures": seed_exposures,
        "seed_rate": round(seed_citations / seed_exposures, 6) if seed_exposures > 0 else None,
        "llm_citations": llm_citations,
        "llm_exposures": llm_exposures,
        "llm_rate": round(llm_citations / llm_exposures, 6) if llm_exposures > 0 else None,
    }


def main():
    results = []
    for node in range(MAX_NODES):
        entries = load_exposure(node, OUTPUT_ROOT)
        rates = compute_rates(entries)
        results.append({"node": node, **rates})

    print(f"{'Node':>4}  {'Seed Rate':>10}  {'Seed C/E':>14}  {'LLM Rate':>10}  {'LLM C/E':>14}")
    print("-" * 60)
    for r in results:
        seed_str = f"{r['seed_citations']}/{r['seed_exposures']}"
        llm_str  = f"{r['llm_citations']}/{r['llm_exposures']}"
        seed_rate = f"{r['seed_rate']:.4f}" if r['seed_rate'] is not None else "N/A"
        llm_rate  = f"{r['llm_rate']:.4f}"  if r['llm_rate']  is not None else "N/A"
        print(f"{r['node']:>4}  {seed_rate:>10}  {seed_str:>14}  {llm_rate:>10}  {llm_str:>14}")

    out_path = f"{OUTPUT_ROOT}/master/seed_vs_llm_citation_rate.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {out_path}")

    # --- Plot ---
    nodes      = [r["node"] for r in results]
    seed_rates = [r["seed_rate"] for r in results]
    llm_rates  = [r["llm_rate"]  if r["llm_rate"] is not None else 0 for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(nodes, seed_rates, color="#E74C3C", marker="o", linewidth=2, label="Seed papers")
    ax.plot(nodes, llm_rates,  color="#3498DB", marker="s", linewidth=2, linestyle="--", label="LLM-generated papers")

    for i, (s, l) in enumerate(zip(seed_rates, llm_rates)):
        ax.annotate(f"{s:.3f}", (i, s), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="#E74C3C")
        ax.annotate(f"{l:.3f}", (i, l), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=7, color="#3498DB")

    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Citation Rate (citations / exposures)")
    ax.set_xticks(nodes)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    ax.legend()
    fig.suptitle("Citation Rate: Seed Papers vs. LLM-Generated Papers by Node", fontsize=12)
    fig.tight_layout()

    fig_path = f"{OUTPUT_ROOT}/master/seed_vs_llm_citation_rate.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
