"""
Distribution of total citations per seed paper aggregated across all nodes.
X axis: individual seed papers (sorted by total citations desc)
Y axis: total citations received across all nodes
"""

import json
from collections import defaultdict
import matplotlib.pyplot as plt

BASE = "archived_synthetic/alpha_10_cap"
MAX_NODES = 12
OUTPUT_ROOT = f"{BASE}/output"


def main():
    seed_citations = defaultdict(int)

    for node in range(MAX_NODES):
        path = f"{OUTPUT_ROOT}/node_{node}/node_{node}_exposure.jsonl"
        with open(path) as f:
            for line in f:
                e = json.loads(line.strip())
                if e["id"].startswith("SEED_"):
                    seed_citations[e["id"]] += e["citations"]

    sorted_papers = sorted(seed_citations.items(), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in sorted_papers]
    counts = [p[1] for p in sorted_papers]

    total = sum(counts)
    print(f"Total seed citations across all nodes: {total}")
    print(f"Seed papers with at least 1 citation: {sum(1 for c in counts if c > 0)} / {len(counts)}")
    print(f"Top 10 seed papers:")
    for label, count in sorted_papers[:10]:
        print(f"  {label}: {count} ({100*count/total:.2f}%)")

    out_path = f"{OUTPUT_ROOT}/master/seed_citation_distribution.jsonl"
    with open(out_path, "w") as f:
        for label, count in sorted_papers:
            f.write(json.dumps({"id": label, "total_citations": count}) + "\n")
    print(f"\nSaved {out_path}")

    # --- Bar chart ---
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(18, 5))
    bars = ax.bar(x, counts, color="#E74C3C", alpha=0.85, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Seed Paper (sorted by total citations, descending)")
    ax.set_ylabel("Total Citations Across All Nodes")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    fig.suptitle("Citation Distribution Across Seed Papers (All Nodes Combined)", fontsize=12)
    fig.tight_layout()

    fig_path = f"{OUTPUT_ROOT}/master/seed_citation_distribution.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
