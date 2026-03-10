import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

MAX_NODES = 6
YEARS = list(range(2017, 2026))
OUTPUT_DIR = "alpha_0.2.0/output/citations_by_year"

id_to_year = {}
with open("alpha_0.2.0/output/master/kv_pairs.jsonl") as f:
    for line in f:
        data = json.loads(line.strip())
        id_to_year[data["id"]] = data["year"]

totals = defaultdict(int)

for node in range(MAX_NODES):
    node_path = f"alpha_0.2.0/output/node_{node}/node_{node}.jsonl"
    with open(node_path) as f:
        for line in f:
            data = json.loads(line.strip())
            for cited_id in data.get("citation_ids", []):
                if cited_id in id_to_year:
                    totals[id_to_year[cited_id]] += 1

all_total = sum(totals.values())
entry = {"node": "all", "total": all_total}
for y in YEARS:
    entry[str(y)] = totals[y]

print(f"{'='*40}")
print("Citations by Year (All Generations)")
print(f"{'='*40}")
for y in YEARS:
    pct = (totals[y] / all_total * 100) if all_total > 0 else 0
    print(f"  {y}: {totals[y]} ({pct:.1f}%)")
print(f"  total: {all_total}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "citations_by_year.jsonl"), "w") as f:
    f.write(json.dumps(entry) + "\n")

print(f"\nSaved to {OUTPUT_DIR}/citations_by_year.jsonl")

# ── Bar chart ────────────────────────────────────────────────────────────────
vals = [totals[y] for y in YEARS]
x = np.arange(len(YEARS))
cmap = plt.cm.tab10

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, vals, color=[cmap(i / len(YEARS)) for i in range(len(YEARS))])
ax.bar_label(bars, padding=2, fontsize=9)

ax.set_xlabel("Paper Year")
ax.set_ylabel("Total Citations Received")
ax.set_title("Citations Received by Paper Year (All Generations)")
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in YEARS])
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "citations_by_year.png"), dpi=150)
print(f"Saved figure to {OUTPUT_DIR}/citations_by_year.png")
