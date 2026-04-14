import json
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = "archived_synthetic/alpha_0.2.0_scaled"
MAX_NODES = 12
SEED_COUNT = 120
PAPERS_PER_NODE = 120
OUTPUT_ROOT = f"{BASE}/output"
RANDOM_ROOT = f"{BASE}/random_output"

valid_ids = set()


def get_available_papers(node: int) -> int:
    return SEED_COUNT + (PAPERS_PER_NODE * node)


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


def top_n_share(sorted_stats: list, n: int, total_citations: int) -> float:
    if total_citations <= 0:
        return 0.0
    top = sorted_stats[:n]
    top_citations = sum(count for _, count in top)
    return round(100.0 * top_citations / total_citations, 2)


def top_percent_share(
    sorted_stats: list, percent: float, total_citations: int, available_papers: int
) -> tuple[float, int]:
    """
    Return (share_pct, k) where k is the number of papers in the top `percent`
    of the available pool, and share_pct is the percentage of total citations
    those k papers account for.
    """
    if total_citations <= 0 or available_papers <= 0 or percent <= 0.0:
        return 0.0, 0
    k = int(round(available_papers * percent))
    if k < 1:
        k = 1
    if k > len(sorted_stats):
        k = len(sorted_stats)
    share = top_n_share(sorted_stats, k, total_citations)
    return share, k


def analyze_node(node: int, data_root: str) -> dict:
    stats = load_node_stats(node, data_root)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    total_citations = sum(stats.values())
    available = get_available_papers(node)

    top_1pct_share, top_1pct_count = top_percent_share(
        sorted_stats, 0.01, total_citations, available
    )
    top_10pct_share, top_10pct_count = top_percent_share(
        sorted_stats, 0.10, total_citations, available
    )

    return {
        "node": node,
        "total_citations": total_citations,
        "unique_papers_cited": len(stats),
        "available_papers": available,
        "top_1pct_share": top_1pct_share,
        "top_10pct_share": top_10pct_share,
        "top_1pct_count": top_1pct_count,
        "top_10pct_count": top_10pct_count,
    }


def run_for_root(data_root: str, label: str):
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
            f"[{label}] Node {node}: total={result['total_citations']}, "
            f"top1pct={result['top_1pct_share']}% (n={result['top_1pct_count']}), "
            f"top10pct={result['top_10pct_share']}% (n={result['top_10pct_count']})"
        )

    with open(f"{output_dir}/concentration.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {output_dir}/concentration.jsonl")

    nodes = [r["node"] for r in results]
    top1pct = [r["top_1pct_share"] for r in results]
    top10pct = [r["top_10pct_share"] for r in results]

    x = np.arange(len(nodes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(
        x - width / 2,
        top1pct,
        width,
        label="Top 1% of papers",
        color="#E74C3C",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    b2 = ax.bar(
        x + width / 2,
        top10pct,
        width,
        label="Top 10% of papers",
        color="#E67E22",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar_label(b1, fmt="%.1f%%", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("% of Total Citations")
    ax.set_title(
        f"Citation Concentration: % of Citations in Top 1% / 10% of Papers per Node — {label}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/concentration.png", dpi=150, bbox_inches="tight")
    print(f"Saved {output_dir}/concentration.png")


def main():
    run_for_root(OUTPUT_ROOT, "Experiment")
    run_for_root(RANDOM_ROOT, "Random")


if __name__ == "__main__":
    main()
