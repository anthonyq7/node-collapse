import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = "alpha_0.2.0_scaled"
MAX_NODES = 12
OUTPUT_ROOT = f"{BASE}/output"
RANDOM_ROOT = f"{BASE}/random_output"

valid_ids = set()


def load_node_stats(node: int, data_root: str) -> dict:
    stats = {}
    path = f"{data_root}/node_{node}/node_{node}_stats.jsonl"
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            for k, v in data.items():
                if k in valid_ids:
                    stats[k] = v
    return stats


def compute_shannon_entropy(stats: dict) -> float:
    """Shannon entropy in bits: H = -sum(p_i * log2(p_i)), p_i = count_i / total."""
    total = sum(stats.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for count in stats.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return round(h, 4)


def analyze_node(node: int, data_root: str) -> dict:
    stats = load_node_stats(node, data_root)
    total_citations = sum(stats.values())
    shannon_entropy = compute_shannon_entropy(stats)
    return {
        "node": node,
        "shannon_entropy": shannon_entropy,
        "total_citations": total_citations,
        "unique_papers_cited": len(stats),
    }


def run_for_root(data_root: str, label: str) -> list:
    output_dir = f"{data_root}/master"
    os.makedirs(output_dir, exist_ok=True)

    valid_ids.clear()
    kv_path = f"{data_root}/master/kv_pairs.jsonl"
    with open(kv_path) as f:
        for line in f:
            data = json.loads(line.strip())
            valid_ids.add(data["id"])

    results = []
    for node in range(MAX_NODES):
        result = analyze_node(node, data_root)
        results.append(result)
        print(
            f"[{label}] Node {node}: shannon_entropy={result['shannon_entropy']}, "
            f"total_citations={result['total_citations']}, "
            f"unique_papers_cited={result['unique_papers_cited']}"
        )

    with open(f"{output_dir}/shannon_entropy.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {output_dir}/shannon_entropy.jsonl")

    nodes = [r["node"] for r in results]
    entropy = [r["shannon_entropy"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        np.arange(len(nodes)),
        entropy,
        color="#3498DB",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title(f"Shannon Entropy of Citation Distribution per Node — {label}")
    ax.set_xticks(np.arange(len(nodes)))
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    fig.tight_layout()
    fig.savefig(f"{output_dir}/shannon_entropy.png", dpi=150, bbox_inches="tight")
    print(f"Saved {output_dir}/shannon_entropy.png")

    return results


def main():
    exp_results = run_for_root(OUTPUT_ROOT, "Experiment")
    random_results = run_for_root(RANDOM_ROOT, "Random")

    # Comparison plot: Experiment vs Random per node
    output_dir = f"{OUTPUT_ROOT}/master"
    nodes = [r["node"] for r in exp_results]
    exp_entropy = [r["shannon_entropy"] for r in exp_results]
    random_entropy = [r["shannon_entropy"] for r in random_results]

    x = np.arange(len(nodes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(
        x - width / 2,
        exp_entropy,
        width,
        label="Experiment",
        color="#3498DB",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    b2 = ax.bar(
        x + width / 2,
        random_entropy,
        width,
        label="Random",
        color="#E67E22",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("Shannon Entropy: Experiment vs Random Citer per Node")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        f"{output_dir}/shannon_entropy_comparison.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved {output_dir}/shannon_entropy_comparison.png")


if __name__ == "__main__":
    main()
