import json
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = "archived_synthetic/alpha_0.2.0_scaled"
MAX_NODES = 12
OUTPUT_ROOT = f"{BASE}/output"
RANDOM_ROOT = f"{BASE}/random_output"


def load_node_fp_counts(node: int, data_root: str) -> dict:
    total_raw = 0
    total_valid = 0
    path = f"{data_root}/node_{node}/node_{node}.jsonl"
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            raw = data.get("citations", data.get("citation_ids", []))
            valid = data.get("citation_ids", [])
            total_raw += len(raw)
            total_valid += len(valid)
    hallucinated = total_raw - total_valid
    if total_raw > 0:
        fp_rate = round(hallucinated / total_raw, 4)
    else:
        fp_rate = 0.0
    return {
        "node": node,
        "total_citations": total_raw,
        "valid": total_valid,
        "hallucinated": hallucinated,
        "fp_rate": fp_rate,
    }


def run_for_root(data_root: str, label: str):
    output_dir = f"{data_root}/master"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for node in range(MAX_NODES):
        row = load_node_fp_counts(node, data_root)
        results.append(row)
        print(f"[{label}] Node {row['node']}: total={row['total_citations']}, valid={row['valid']}, "
              f"hallucinated={row['hallucinated']}, fp_rate={row['fp_rate']}")

    with open(f"{output_dir}/false_positives.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {output_dir}/false_positives.jsonl")

    headers = ["Node", "Total Citations", "Valid", "Hallucinated", "FP Rate"]
    rows = []
    for r in results:
        rows.append([
            r["node"],
            r["total_citations"],
            r["valid"],
            r["hallucinated"],
            f"{r['fp_rate']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D9E2F3")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/false_positives_table.png", dpi=200, bbox_inches="tight")
    print(f"Saved {output_dir}/false_positives_table.png")

    nodes = [r["node"] for r in results]
    halluc = [r["hallucinated"] for r in results]
    fp_rates = [r["fp_rate"] for r in results]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    bars = ax1.bar(nodes, halluc, color="#E74C3C", alpha=0.8, edgecolor="black", linewidth=0.5,
                   label="Hallucinated Citations")
    ax1.bar_label(bars, padding=3, fontsize=9)
    ax1.set_xlabel("Node (Generation)")
    ax1.set_ylabel("Hallucinated Citations (count)")
    ax1.set_xticks(nodes)
    ax1.set_xticklabels([f"Node {n}" for n in nodes])

    ax2 = ax1.twinx()
    ax2.plot(nodes, fp_rates, color="#2C3E50", marker="o", linewidth=2, label="FP Rate")
    for i, rate in enumerate(fp_rates):
        ax2.annotate(f"{rate:.3f}", (nodes[i], fp_rates[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color="#2C3E50")
    ax2.set_ylabel("False Positive Rate", color="#2C3E50")
    max_rate = max(fp_rates) if fp_rates else 0.05
    if max_rate <= 0:
        max_rate = 0.05
    ax2.set_ylim(0, max_rate * 1.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.suptitle(f"False Positives (Hallucinations) by Node — {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/false_positives.png", dpi=150, bbox_inches="tight")
    print(f"Saved {output_dir}/false_positives.png")


def main():
    run_for_root(OUTPUT_ROOT, "Experiment")
    run_for_root(RANDOM_ROOT, "Random")


if __name__ == "__main__":
    main()
