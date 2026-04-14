import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

MAX_NODES = 6
OUTPUT_DIR = "archived_synthetic/alpha_0.2.0/output/analysis_figures"
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
BIN_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

gen_stats = []
with open("archived_synthetic/alpha_0.2.0/output/master/generation_stats.jsonl") as f:
    for line in f:
        gen_stats.append(json.loads(line.strip()))

# ── 1. Summary Table ─────────────────────────────────────────────────────────

headers = ["Node", "Gini", "Total Citations", "Unique Cited", "Available"]
rows = []
for s in gen_stats:
    rows.append([
        s["node"],
        s["gini"],
        s["total_citations"],
        f"{s['unique_papers_cited']}/{s['available_papers']}",
        s["available_papers"],
    ])

fig, ax = plt.subplots(figsize=(8, 3))
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
fig.savefig(os.path.join(OUTPUT_DIR, "summary_table.png"), dpi=200, bbox_inches="tight")
print("Saved summary_table.png")

# ── 1b. Citation Verification Table ──────────────────────────────────────────

cv_headers = ["Node", "Total Citations", "Valid", "Hallucinated", "Valid %", "Hallucinated %"]
cv_rows = []
with open("archived_synthetic/alpha_0.2.0/output/citation_verification_by_node.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cv_rows.append([
            row["node"],
            row["total_citations"],
            row["valid"],
            row["hallucinated"],
            f"{float(row['valid_pct']):.2f}",
            f"{float(row['hallucinated_pct']):.2f}",
        ])

fig, ax = plt.subplots(figsize=(9, 3))
ax.axis("off")
table = ax.table(
    cellText=cv_rows,
    colLabels=cv_headers,
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
fig.savefig(os.path.join(OUTPUT_DIR, "citation_verification_table.png"), dpi=200, bbox_inches="tight")
print("Saved citation_verification_table.png")

# ── 2. Citation/Exposure Rate Distributions Per Node ─────────────────────────

def load_exposure(node):
    entries = []
    path = f"archived_synthetic/alpha_0.2.0/output/node_{node}/node_{node}_exposure.jsonl"
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries

def bucket_rates(entries):
    counts = [0] * len(BINS)
    for e in entries:
        r = e["rate"]
        for i, (lo, hi) in enumerate(BINS):
            if lo <= r < hi or (i == len(BINS) - 1 and r == hi):
                counts[i] += 1
                break
    return counts

dist_results = []
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()

for node in range(MAX_NODES):
    entries = load_exposure(node)
    counts = bucket_rates(entries)
    dist_results.append({
        "node": node,
        "buckets": {label: c for label, c in zip(BIN_LABELS, counts)},
        "total_papers": len(entries),
    })

    ax = axes[node]
    bars = ax.bar(BIN_LABELS, counts, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_title(f"Node {node}")
    ax.set_xlabel("Citation/Exposure Rate")
    if node % 3 == 0:
        ax.set_ylabel("Paper Count")
    ax.tick_params(axis="x", rotation=45)

fig.suptitle("Citation/Exposure Rate Distribution Per Node", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "citation_exposure_rate_distributions.png"), dpi=150, bbox_inches="tight")
print("Saved citation_exposure_rate_distributions.png")

with open(os.path.join(OUTPUT_DIR, "citation_exposure_rate_distributions.jsonl"), "w") as f:
    for entry in dist_results:
        f.write(json.dumps(entry) + "\n")
print("Saved citation_exposure_rate_distributions.jsonl")

# ── 3. Average Citation/Exposure Rate By Node (Seed vs Generated) ────────────

avg_results = []
for node in range(MAX_NODES):
    entries = load_exposure(node)
    seed_rates = [e["rate"] for e in entries if e["id"].startswith("SEED_")]
    gen_rates = [e["rate"] for e in entries if not e["id"].startswith("SEED_")]
    all_rates = [e["rate"] for e in entries]

    seed_avg = round(np.mean(seed_rates), 4) if seed_rates else 0.0
    gen_avg = round(np.mean(gen_rates), 4) if gen_rates else 0.0
    overall_avg = round(np.mean(all_rates), 4) if all_rates else 0.0

    avg_results.append({
        "node": node,
        "seed_avg_rate": seed_avg,
        "generated_avg_rate": gen_avg,
        "overall_avg_rate": overall_avg,
        "seed_count": len(seed_rates),
        "generated_count": len(gen_rates),
    })

with open(os.path.join(OUTPUT_DIR, "avg_citation_exposure_rate_by_node.jsonl"), "w") as f:
    for entry in avg_results:
        f.write(json.dumps(entry) + "\n")
print("Saved avg_citation_exposure_rate_by_node.jsonl")

x = np.arange(MAX_NODES)
width = 0.28
seed_avgs = [r["seed_avg_rate"] for r in avg_results]
gen_avgs = [r["generated_avg_rate"] for r in avg_results]
overall_avgs = [r["overall_avg_rate"] for r in avg_results]

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width, seed_avgs, width, label="Seed Papers")
b2 = ax.bar(x, gen_avgs, width, label="Generated Papers")
b3 = ax.bar(x + width, overall_avgs, width, label="Overall")
ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=7)
ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=7)
ax.bar_label(b3, fmt="%.3f", padding=2, fontsize=7)
ax.set_xlabel("Node (Generation)")
ax.set_ylabel("Average Citation/Exposure Rate")
ax.set_title("Average Citation/Exposure Rate by Node (Seed vs Generated)")
ax.set_xticks(x)
ax.set_xticklabels([f"Node {n}" for n in range(MAX_NODES)])
ax.set_ylim(0, 1.15)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "avg_citation_exposure_rate_by_node.png"), dpi=150)
print("Saved avg_citation_exposure_rate_by_node.png")

# ── 4. Exclusion Rate By Node (Shown-but-Ignored vs Never-Shown) ─────────────

excl_results = []
for s in gen_stats:
    node = s["node"]
    available = s["available_papers"]
    entries = load_exposure(node)

    cited = sum(1 for e in entries if e["citations"] > 0)
    shown_ignored = sum(1 for e in entries if e["exposures"] > 0 and e["citations"] == 0)
    never_shown = available - len(entries)
    uncited = shown_ignored + never_shown

    excl_results.append({
        "node": node,
        "available": available,
        "cited": cited,
        "shown_ignored": shown_ignored,
        "never_shown": never_shown,
        "uncited": uncited,
        "exclusion_rate": round(uncited / available, 4) if available > 0 else 0.0,
        "shown_ignored_rate": round(shown_ignored / available, 4) if available > 0 else 0.0,
        "never_shown_rate": round(never_shown / available, 4) if available > 0 else 0.0,
    })

with open(os.path.join(OUTPUT_DIR, "exclusion_rate_by_node.jsonl"), "w") as f:
    for entry in excl_results:
        f.write(json.dumps(entry) + "\n")
print("Saved exclusion_rate_by_node.jsonl")

nodes = [r["node"] for r in excl_results]
shown_ignored = [r["shown_ignored"] for r in excl_results]
never_shown = [r["never_shown"] for r in excl_results]
excl_rates = [r["exclusion_rate"] for r in excl_results]

fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(nodes))
b1 = ax1.bar(x, shown_ignored, color="#E74C3C", alpha=0.8, label="Shown but Ignored")
b2 = ax1.bar(x, never_shown, bottom=shown_ignored, color="#95A5A6", alpha=0.7, label="Never Shown")
ax1.bar_label(b1, padding=2, fontsize=8)
ax1.bar_label(b2, padding=2, fontsize=8)
ax1.set_xlabel("Node (Generation)")
ax1.set_ylabel("Uncited Papers")
ax1.set_xticks(x)
ax1.set_xticklabels([f"Node {n}" for n in nodes])

ax2 = ax1.twinx()
ax2.plot(x, excl_rates, color="#2C3E50", marker="o", linewidth=2, label="Exclusion Rate")
for i, rate in enumerate(excl_rates):
    ax2.annotate(f"{rate:.1%}", (x[i], rate), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=8, color="#2C3E50")
ax2.set_ylabel("Exclusion Rate", color="#2C3E50")
max_rate = max(excl_rates) if max(excl_rates) > 0 else 0.05
ax2.set_ylim(-0.01, max_rate * 1.5 + 0.02)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig.suptitle("Exclusion Rate by Node (Shown-but-Ignored vs Never-Shown)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "exclusion_rate_by_node.png"), dpi=150)
print("Saved exclusion_rate_by_node.png")

# ── 5. Gini Coefficient Progression ──────────────────────────────────────────

gini_nodes = [s["node"] for s in gen_stats]
gini_vals = [s["gini"] for s in gen_stats]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(gini_nodes, gini_vals, marker="o", linewidth=2, markersize=8, color="#2E86C1")
ax.fill_between(gini_nodes, gini_vals, alpha=0.15, color="#2E86C1")
for i, val in enumerate(gini_vals):
    ax.annotate(f"{val:.4f}", (gini_nodes[i], val), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9)
ax.set_xlabel("Node (Generation)")
ax.set_ylabel("Gini Coefficient")
ax.set_title("Gini Coefficient Progression Across Generations")
ax.set_xticks(gini_nodes)
ax.set_xticklabels([f"Node {n}" for n in gini_nodes])
ax.set_ylim(0, max(gini_vals) * 1.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "gini_progression.png"), dpi=150)
print("Saved gini_progression.png")

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
