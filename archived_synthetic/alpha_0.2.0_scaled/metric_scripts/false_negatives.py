import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE = "archived_synthetic/alpha_0.2.0_scaled"
MAX_NODES = 12
SEED_COUNT = 120
PAPERS_PER_NODE = 120
OUTPUT_ROOT = f"{BASE}/output"
RANDOM_ROOT = f"{BASE}/random_output"

valid_ids = set()


def get_available_papers(node: int) -> int:
    return SEED_COUNT + (PAPERS_PER_NODE * node)


def build_exposure_for_node(node: int, data_root: str) -> list:
    citations = defaultdict(int)
    exposures = defaultdict(int)

    stats_path = f"{data_root}/node_{node}/node_{node}_stats.jsonl"
    with open(stats_path) as f:
        for line in f:
            data = json.loads(line.strip())
            for paper_id, count in data.items():
                if paper_id in valid_ids:
                    citations[paper_id] += count

    node_path = f"{data_root}/node_{node}/node_{node}.jsonl"
    with open(node_path) as f:
        for line in f:
            data = json.loads(line.strip())
            papers_seen = data.get("papers_seen_id", [])
            for paper_id in papers_seen:
                if paper_id in valid_ids:
                    exposures[paper_id] += 1

    all_ids = set(citations.keys()) | set(exposures.keys())
    results = []
    for paper_id in all_ids:
        cite_count = citations[paper_id]
        expose_count = exposures[paper_id]
        rate = cite_count / expose_count if expose_count > 0 else 0.0
        results.append({
            "id": paper_id,
            "citations": cite_count,
            "exposures": expose_count,
            "rate": round(rate, 4),
        })
    results.sort(key=lambda x: x["rate"], reverse=True)
    return results


def write_exposure_jsonl(node: int, results: list, data_root: str) -> None:
    path = f"{data_root}/node_{node}/node_{node}_exposure.jsonl"
    with open(path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {path}")


def compute_exclusion(node: int, available: int, entries: list) -> dict:
    cited = sum(1 for e in entries if e["citations"] > 0)
    shown_ignored = sum(1 for e in entries if e["exposures"] > 0 and e["citations"] == 0)
    shown_papers = cited + shown_ignored
    shown_ignored_rate_of_shown = (
        round(shown_ignored / shown_papers, 4) if shown_papers > 0 else 0.0
    )
    never_shown = available - len(entries)
    uncited = shown_ignored + never_shown
    exclusion_rate = round(uncited / available, 4) if available > 0 else 0.0
    shown_ignored_rate = round(shown_ignored / available, 4) if available > 0 else 0.0
    never_shown_rate = round(never_shown / available, 4) if available > 0 else 0.0
    total_citations = sum(e["citations"] for e in entries)
    return {
        "node": node,
        "available": available,
        "cited": cited,
        "shown_ignored": shown_ignored,
        "shown_papers": shown_papers,
        "shown_ignored_rate_of_shown": shown_ignored_rate_of_shown,
        "never_shown": never_shown,
        "uncited": uncited,
        "exclusion_rate": exclusion_rate,
        "shown_ignored_rate": shown_ignored_rate,
        "never_shown_rate": never_shown_rate,
        "total_citations": total_citations,
    }


def run_for_root(data_root: str, label: str) -> list:
    output_dir = f"{data_root}/analysis_figures"
    os.makedirs(output_dir, exist_ok=True)

    valid_ids.clear()
    kv_path = f"{data_root}/master/kv_pairs.jsonl"
    with open(kv_path) as f:
        for line in f:
            data = json.loads(line.strip())
            valid_ids.add(data["id"])

    excl_results = []
    for node in range(MAX_NODES):
        entries = build_exposure_for_node(node, data_root)
        write_exposure_jsonl(node, entries, data_root)
        available = get_available_papers(node)
        row = compute_exclusion(node, available, entries)
        excl_results.append(row)
        print(f"[{label}] Node {node}: shown_ignored={row['shown_ignored']}, "
              f"shown_papers={row['shown_papers']}, "
              f"shown_ignored_rate_of_shown={row['shown_ignored_rate_of_shown']}, "
              f"cited={row['cited']}, total_citations={row['total_citations']}")

    with open(f"{output_dir}/exclusion_rate_by_node.jsonl", "w") as f:
        for r in excl_results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {output_dir}/exclusion_rate_by_node.jsonl")

    return excl_results


def main():
    exp_results = run_for_root(OUTPUT_ROOT, "Experiment")
    random_results = run_for_root(RANDOM_ROOT, "Random")

    nodes = [r["node"] for r in exp_results]
    exp_rates = [r["shown_ignored_rate_of_shown"] for r in exp_results]
    random_rates = [r["shown_ignored_rate_of_shown"] for r in random_results]

    x = np.arange(len(nodes))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, exp_rates, color="#E74C3C", marker="o", linewidth=2, label="Experiment")
    ax.plot(x, random_rates, color="#3498DB", marker="s", linewidth=2, label="Random")
    for i, rate in enumerate(exp_rates):
        ax.annotate(f"{rate:.1%}", (x[i], exp_rates[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color="#E74C3C")
    for i, rate in enumerate(random_rates):
        ax.annotate(f"{rate:.1%}", (x[i], random_rates[i]), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color="#3498DB")
    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Shown-but-Ignored Rate (of shown papers)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    all_rates = exp_rates + random_rates
    max_rate = max(all_rates) if all_rates else 0.05
    ax.set_ylim(0, max_rate * 1.25)
    ax.legend()
    fig.suptitle("False Negatives: Uncited Rate among Shown Papers by Node — Experiment vs Random", fontsize=12)
    fig.tight_layout()
    out_dir = f"{OUTPUT_ROOT}/analysis_figures"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{out_dir}/exclusion_rate_by_node.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir}/exclusion_rate_by_node.png")


if __name__ == "__main__":
    main()
