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

papers_per_year = defaultdict(int)
for year in id_to_year.values():
    papers_per_year[year] += 1

citations_per_year = defaultdict(int)
for node in range(MAX_NODES):
    node_path = f"alpha_0.2.0/output/node_{node}/node_{node}.jsonl"
    with open(node_path) as f:
        for line in f:
            data = json.loads(line.strip())
            for cited_id in data.get("citation_ids", []):
                if cited_id in id_to_year:
                    citations_per_year[id_to_year[cited_id]] += 1

entry = {}
print(f"{'='*50}")
print("Average Citations Per Paper by Year")
print(f"{'='*50}")
for y in YEARS:
    cites = citations_per_year[y]
    papers = papers_per_year[y]
    avg = round(cites / papers, 2) if papers > 0 else 0.0
    entry[str(y)] = {"citations": cites, "papers": papers, "avg": avg}
    print(f"  {y}: {cites} citations / {papers} papers = {avg} avg")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "citations_per_paper_year.jsonl"), "w") as f:
    f.write(json.dumps(entry) + "\n")

print(f"\nSaved to {OUTPUT_DIR}/citations_per_paper_year.jsonl")

# ── Bar chart ────────────────────────────────────────────────────────────────
avgs = [entry[str(y)]["avg"] for y in YEARS]
x = np.arange(len(YEARS))
cmap = plt.cm.tab10

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, avgs, color=[cmap(i / len(YEARS)) for i in range(len(YEARS))])
ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)

ax.set_xlabel("Paper Year")
ax.set_ylabel("Average Citations Per Paper")
ax.set_title("Average Citations Per Paper by Year (All Generations)")
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in YEARS])
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "citations_per_paper_year.png"), dpi=150)
print(f"Saved figure to {OUTPUT_DIR}/citations_per_paper_year.png")
