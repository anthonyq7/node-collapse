import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

MAX_NODES = 6
OUTPUT_DIR = "alpha_0.2.0/output/paper_type_citations"
TYPES = ["seed paper", "position paper", "literature review"]

valid_ids = set()
with open("alpha_0.2.0/output/master/kv_pairs.jsonl") as f:
    for line in f:
        data = json.loads(line.strip())
        valid_ids.add(data["id"])

type_lookup = {}

for paper_id in valid_ids:
    if paper_id.startswith("SEED_"):
        type_lookup[paper_id] = "seed paper"

for node in range(MAX_NODES):
    node_path = f"alpha_0.2.0/output/node_{node}/node_{node}.jsonl"
    with open(node_path) as f:
        for line in f:
            data = json.loads(line.strip())
            paper_id = data["id"]
            if paper_id in valid_ids:
                type_lookup[paper_id] = data.get("type", "unknown")

node_results = []
totals = defaultdict(int)

for node in range(MAX_NODES):
    counts = defaultdict(int)
    node_path = f"alpha_0.2.0/output/node_{node}/node_{node}.jsonl"
    with open(node_path) as f:
        for line in f:
            data = json.loads(line.strip())
            citation_ids = data.get("citation_ids", [])
            for cited_id in citation_ids:
                if cited_id in valid_ids and cited_id in type_lookup:
                    counts[type_lookup[cited_id]] += 1

    total = sum(counts.values())
    entry = {
        "node": node,
        "position_paper": counts["position paper"],
        "literature_review": counts["literature review"],
        "seed_paper": counts["seed paper"],
        "total": total
    }
    node_results.append(entry)

    for t in TYPES:
        totals[t] += counts[t]

    print(f"\n{'='*40}")
    print(f"Node {node}")
    print(f"{'='*40}")
    for t in TYPES:
        pct = (counts[t] / total * 100) if total > 0 else 0
        print(f"  {t}: {counts[t]} ({pct:.1f}%)")
    print(f"  total: {total}")

all_total = sum(totals.values())
all_entry = {
    "node": "all",
    "position_paper": totals["position paper"],
    "literature_review": totals["literature review"],
    "seed_paper": totals["seed paper"],
    "total": all_total
}

print(f"\n{'='*40}")
print("Across all generations")
print(f"{'='*40}")
for t in TYPES:
    pct = (totals[t] / all_total * 100) if all_total > 0 else 0
    print(f"  {t}: {totals[t]} ({pct:.1f}%)")
print(f"  total: {all_total}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "paper_type_citations.jsonl"), "w") as f:
    for entry in node_results:
        f.write(json.dumps(entry) + "\n")
    f.write(json.dumps(all_entry) + "\n")

print(f"\nSaved to {OUTPUT_DIR}/paper_type_citations.jsonl")

nodes = [r["node"] for r in node_results]
x = np.arange(len(nodes))
width = 0.25

seed_vals = [r["seed_paper"] for r in node_results]
pos_vals = [r["position_paper"] for r in node_results]
lit_vals = [r["literature_review"] for r in node_results]

fig, ax = plt.subplots(figsize=(10, 6))
bars_seed = ax.bar(x - width, seed_vals, width, label="Seed Paper")
bars_pos = ax.bar(x, pos_vals, width, label="Position Paper")
bars_lit = ax.bar(x + width, lit_vals, width, label="Literature Review")
ax.bar_label(bars_seed, padding=2, fontsize=8)
ax.bar_label(bars_pos, padding=2, fontsize=8)
ax.bar_label(bars_lit, padding=2, fontsize=8)
ax.set_xlabel("Node (Generation)")
ax.set_ylabel("Citation Count")
ax.set_title("Citations by Paper Type per Node")
ax.set_xticks(x)
ax.set_xticklabels([f"Node {n}" for n in nodes])
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "paper_type_by_node.png"), dpi=150)
print(f"Saved figure to {OUTPUT_DIR}/paper_type_by_node.png")

fig2, ax2 = plt.subplots(figsize=(7, 7))
sizes = [totals[t] for t in TYPES]
labels = [f"{t}\n({totals[t]}, {totals[t]/all_total*100:.1f}%)" for t in TYPES]
ax2.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
ax2.set_title("Citation Distribution by Paper Type (All Generations)")
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "paper_type_across_generations.png"), dpi=150)
print(f"Saved figure to {OUTPUT_DIR}/paper_type_across_generations.png")
