import json
import os
import re
from collections import defaultdict
import math

import matplotlib.pyplot as plt


BASE = "archived_synthetic/alpha_0.2.0_scaled"
MAX_NODES = 12
OUTPUT_ROOT = f"{BASE}/output"
RANDOM_ROOT = f"{BASE}/random_output"

SEED_PREFIX = "SEED_"
ORIGIN_NODE_PATTERN = re.compile(r"^N(\d+)P")

valid_ids: set[str] = set()


def build_exposure_for_node(node: int, data_root: str) -> list[dict]:
    citations: dict[str, int] = defaultdict(int)
    exposures: dict[str, int] = defaultdict(int)

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
    results: list[dict] = []
    for paper_id in all_ids:
        cite_count = citations[paper_id]
        expose_count = exposures[paper_id]
        rate = cite_count / expose_count if expose_count > 0 else 0.0
        results.append(
            {
                "id": paper_id,
                "citations": cite_count,
                "exposures": expose_count,
                "rate": round(rate, 4),
            }
        )
    results.sort(key=lambda x: x["rate"], reverse=True)
    return results


def write_exposure_jsonl(node: int, results: list[dict], data_root: str) -> None:
    path = f"{data_root}/node_{node}/node_{node}_exposure.jsonl"
    with open(path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")


def load_exposure_entries(node: int, data_root: str) -> list[dict]:
    exposure_path = f"{data_root}/node_{node}/node_{node}_exposure.jsonl"
    if not os.path.exists(exposure_path):
        entries = build_exposure_for_node(node, data_root)
        write_exposure_jsonl(node, entries, data_root)
    entries: list[dict] = []
    with open(exposure_path) as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line.strip()))
    return entries


def parse_origin_node(paper_id: str) -> str:
    if paper_id.startswith(SEED_PREFIX):
        return "Seed"
    match = ORIGIN_NODE_PATTERN.match(paper_id)
    if match:
        node_idx = int(match.group(1))
        return f"Node {node_idx}"
    return "Unknown"


def compute_shown_ignored_by_node(data_root: str, label: str) -> list[dict]:
    output_dir = f"{data_root}/analysis_figures"
    os.makedirs(output_dir, exist_ok=True)

    results: list[dict] = []
    jsonl_path = f"{output_dir}/shown_ignored_makeup.jsonl"
    with open(jsonl_path, "w") as outf:
        for node in range(MAX_NODES):
            entries = load_exposure_entries(node, data_root)
            shown_ignored_entries = [
                e
                for e in entries
                if e.get("exposures", 0) > 0 and e.get("citations", 0) == 0
            ]
            shown_ignored_ids = [e["id"] for e in shown_ignored_entries]

            origin_counts: dict[str, int] = defaultdict(int)
            for pid in shown_ignored_ids:
                origin = parse_origin_node(pid)
                origin_counts[origin] += 1

            record = {
                "node": node,
                "experiment": label,
                "shown_ignored_ids": shown_ignored_ids,
                "origin_counts": dict(origin_counts),
                "total_shown_ignored": len(shown_ignored_ids),
            }
            results.append(record)
            outf.write(json.dumps(record) + "\n")
    return results


def plot_pies_for_root(records: list[dict], pies_dir: str, label: str) -> None:
    os.makedirs(pies_dir, exist_ok=True)

    for rec in records:
        node = rec["node"]
        origin_counts: dict = rec.get("origin_counts", {})
        if not origin_counts:
            continue

        labels: list[str] = []
        short_labels: list[str] = []
        sizes: list[int] = []
        for origin, count in origin_counts.items():
            if count <= 0:
                continue
            labels.append(origin)
            if origin == "Seed":
                short_labels.append("S")
            elif origin.startswith("Node "):
                try:
                    idx = int(origin.split(" ", 1)[1])
                    short_labels.append(f"N{idx}")
                except Exception:
                    short_labels.append(origin)
            else:
                short_labels.append(origin)
            sizes.append(count)

        if not sizes:
            continue

        total = sum(sizes)

        fig, ax = plt.subplots(figsize=(5, 5))
        wedges = ax.pie(
            sizes,
            startangle=90,
            counterclock=False,
        )[0]

        # Inside labels: short origin codes (e.g., N1, S)
        for wedge, short_label in zip(wedges, short_labels):
            theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
            r_in = 0.6
            x_in = r_in * math.cos(theta)
            y_in = r_in * math.sin(theta)
            ax.text(
                x_in,
                y_in,
                short_label,
                ha="center",
                va="center",
                fontsize=9,
                color="white",
            )

        # Outside labels: percentages with leader lines
        for wedge, size in zip(wedges, sizes):
            if total <= 0:
                continue
            pct = 100.0 * size / total
            theta = math.radians((wedge.theta1 + wedge.theta2) / 2.0)
            r_out = 1.15
            x_out = r_out * math.cos(theta)
            y_out = r_out * math.sin(theta)
            ha = "left" if x_out >= 0 else "right"
            ax.annotate(
                f"{pct:.1f}%",
                xy=(math.cos(theta), math.sin(theta)),
                xytext=(x_out, y_out),
                ha=ha,
                va="center",
                fontsize=8,
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle="angle3,angleA=0,angleB=90",
                    linewidth=0.6,
                    color="gray",
                ),
            )

        ax.axis("equal")
        fig.suptitle(
            f"Shown-but-Ignored Origin Makeup — Node {node} ({label})", fontsize=11
        )
        fig.tight_layout()

        out_path = os.path.join(
            pies_dir, f"node_{node}_{label.lower().replace(' ', '_')}.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def init_valid_ids(data_root: str) -> None:
    valid_ids.clear()
    kv_path = f"{data_root}/master/kv_pairs.jsonl"
    with open(kv_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            valid_ids.add(data["id"])


def run_for_root(data_root: str, label: str) -> None:
    init_valid_ids(data_root)
    records = compute_shown_ignored_by_node(data_root, label)
    pies_dir = os.path.join(data_root, "analysis_figures", "shown_ignored_pies")
    plot_pies_for_root(records, pies_dir, label)


def main() -> None:
    run_for_root(OUTPUT_ROOT, "Experiment")
    run_for_root(RANDOM_ROOT, "Random")


if __name__ == "__main__":
    main()

