import json
import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

MAX_NODES = 6
SEED_COUNT = 30
PAPERS_PER_NODE = 30
BASE = "archived_synthetic/alpha_0.2.0"
OUTPUT_DIR = f"{BASE}/output/additional_evidence"

os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_ids = set()
with open(f"{BASE}/output/master/kv_pairs.jsonl") as f:
    for line in f:
        data = json.loads(line.strip())
        valid_ids.add(data["id"])


def get_origin(paper_id):
    if paper_id.startswith("SEED_"):
        return "Seed"
    m = re.match(r"N(\d+)P", paper_id)
    if m:
        return f"Gen {m.group(1)}"
    return "Unknown"


def gini(values):
    if not values or sum(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumsum) / (n * sum(sorted_vals)) - (n + 1) / n


def load_stats(base_dir, node):
    stats = {}
    path = f"{base_dir}/node_{node}/node_{node}_stats.jsonl"
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            for k, v in data.items():
                if k in valid_ids:
                    stats[k] = v
    return stats


def compute_gini_series(base_dir):
    results = []
    for node in range(MAX_NODES):
        stats = load_stats(base_dir, node)
        available = SEED_COUNT + PAPERS_PER_NODE * node
        uncited = available - len(stats)
        all_values = list(stats.values()) + [0] * uncited
        results.append(round(gini(all_values), 4))
    return results


# ── 1. Gini Comparison Against Baselines ─────────────────────────────────────

experiment_gini = compute_gini_series(f"{BASE}/output")
random_gini = compute_gini_series(f"{BASE}/random_output")
semantic_gini = compute_gini_series(f"{BASE}/semantic_output")

gini_data = []
for node in range(MAX_NODES):
    gini_data.append({
        "node": node,
        "experiment": experiment_gini[node],
        "random": random_gini[node],
        "semantic": semantic_gini[node],
    })

with open(os.path.join(OUTPUT_DIR, "gini_comparison.jsonl"), "w") as f:
    for entry in gini_data:
        f.write(json.dumps(entry) + "\n")

nodes = list(range(MAX_NODES))
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(nodes, experiment_gini, marker="o", linewidth=2, label="LLM Experiment")
ax.plot(nodes, random_gini, marker="s", linewidth=2, linestyle="--", label="Random Baseline")
ax.plot(nodes, semantic_gini, marker="^", linewidth=2, linestyle=":", label="Semantic Baseline")
for i in range(MAX_NODES):
    ax.annotate(f"{experiment_gini[i]:.4f}", (i, experiment_gini[i]),
                textcoords="offset points", xytext=(0, 12), ha="center", fontsize=7)
    ax.annotate(f"{random_gini[i]:.4f}", (i, random_gini[i]),
                textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7)
    ax.annotate(f"{semantic_gini[i]:.4f}", (i, semantic_gini[i]),
                textcoords="offset points", xytext=(0, 12), ha="center", fontsize=7, color="green")
ax.set_xlabel("Node (Generation)")
ax.set_ylabel("Gini Coefficient")
ax.set_title("Gini Coefficient: LLM vs Random vs Semantic Baselines")
ax.set_xticks(nodes)
ax.set_xticklabels([f"Node {n}" for n in nodes])
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "gini_comparison.png"), dpi=150)
print("Saved gini_comparison.png + gini_comparison.jsonl")

# ── 2. Citation Concentration by Paper Origin Generation ─────────────────────

ORIGIN_LABELS = ["Seed"] + [f"Gen {i}" for i in range(MAX_NODES)]

origin_data = []
for node in range(MAX_NODES):
    counts = defaultdict(int)
    node_path = f"{BASE}/output/node_{node}/node_{node}.jsonl"
    with open(node_path) as f:
        for line in f:
            data = json.loads(line.strip())
            for cid in data.get("citation_ids", []):
                if cid in valid_ids:
                    counts[get_origin(cid)] += 1
    total = sum(counts.values())
    entry = {"node": node, "total": total}
    for label in ORIGIN_LABELS:
        entry[label] = counts.get(label, 0)
    origin_data.append(entry)

with open(os.path.join(OUTPUT_DIR, "citation_by_origin.jsonl"), "w") as f:
    for entry in origin_data:
        f.write(json.dumps(entry) + "\n")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(MAX_NODES)
bottom = np.zeros(MAX_NODES)
for label in ORIGIN_LABELS:
    vals = []
    for entry in origin_data:
        total = entry["total"]
        vals.append(entry.get(label, 0) / total * 100 if total > 0 else 0)
    vals = np.array(vals)
    if vals.sum() > 0:
        ax.bar(x, vals, bottom=bottom, label=label)
        bottom += vals
ax.set_xlabel("Citing Node (Generation)")
ax.set_ylabel("% of Citations")
ax.set_title("Citation Concentration by Paper Origin Generation")
ax.set_xticks(x)
ax.set_xticklabels([f"Node {n}" for n in range(MAX_NODES)])
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "citation_by_origin.png"), dpi=150)
print("Saved citation_by_origin.png + citation_by_origin.jsonl")

# ── 3. Rich-Get-Richer (Cumulative Advantage) ───────────────────────────────

cumulative = defaultdict(int)
node_snapshots = []

for node in range(MAX_NODES):
    stats = load_stats(f"{BASE}/output", node)
    for pid, count in stats.items():
        cumulative[pid] += count
    node_snapshots.append(dict(cumulative))

corr_data = []
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for t in range(MAX_NODES - 1):
    snap_before = node_snapshots[t]
    snap_after = node_snapshots[t + 1]
    all_ids = set(snap_before.keys()) | set(snap_after.keys())

    xs = [snap_before.get(pid, 0) for pid in all_ids]
    ys = [snap_after.get(pid, 0) for pid in all_ids]

    r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 1 else 0.0
    corr_data.append({"transition": f"{t}->{t+1}", "pearson_r": round(r, 4), "n_papers": len(all_ids)})

    ax = axes[t]
    is_seed = [pid.startswith("SEED_") for pid in all_ids]
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    seed_mask = np.array(is_seed)
    ax.scatter(xs_arr[seed_mask], ys_arr[seed_mask], alpha=0.6, s=20, label="Seed", zorder=3)
    ax.scatter(xs_arr[~seed_mask], ys_arr[~seed_mask], alpha=0.6, s=20, label="Generated", zorder=2)
    max_val = max(max(xs), max(ys)) + 5
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel(f"Cumulative citations through Node {t}")
    ax.set_ylabel(f"Cumulative citations through Node {t+1}")
    ax.set_title(f"Node {t} -> {t+1}  (r={r:.4f})")
    ax.legend(fontsize=7)

axes[5].axis("off")
fig.suptitle("Cumulative Advantage (Rich-Get-Richer)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "cumulative_advantage.png"), dpi=150, bbox_inches="tight")

with open(os.path.join(OUTPUT_DIR, "cumulative_advantage.jsonl"), "w") as f:
    for entry in corr_data:
        f.write(json.dumps(entry) + "\n")
print("Saved cumulative_advantage.png + cumulative_advantage.jsonl")

# ── 4. Citation/Exposure Rate by Paper Origin Generation ─────────────────────

rate_data = []
for node in range(MAX_NODES):
    path = f"{BASE}/output/node_{node}/node_{node}_exposure.jsonl"
    by_origin = defaultdict(list)
    with open(path) as f:
        for line in f:
            e = json.loads(line.strip())
            origin = get_origin(e["id"])
            by_origin[origin].append(e["rate"])

    entry = {"node": node}
    for label in ORIGIN_LABELS:
        rates = by_origin.get(label, [])
        entry[label] = round(float(np.mean(rates)), 4) if rates else None
    rate_data.append(entry)

with open(os.path.join(OUTPUT_DIR, "rate_by_origin.jsonl"), "w") as f:
    for entry in rate_data:
        f.write(json.dumps(entry) + "\n")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(MAX_NODES)
present_labels = []
for label in ORIGIN_LABELS:
    if any(entry.get(label) is not None for entry in rate_data):
        present_labels.append(label)

n_bars = len(present_labels)
width = 0.8 / n_bars
offsets = np.arange(n_bars) - (n_bars - 1) / 2

for i, label in enumerate(present_labels):
    vals = []
    for entry in rate_data:
        v = entry.get(label)
        vals.append(v if v is not None else 0)
    bars = ax.bar(x + offsets[i] * width, vals, width, label=label)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=6)

ax.set_xlabel("Node (Generation)")
ax.set_ylabel("Avg Citation/Exposure Rate")
ax.set_title("Citation/Exposure Rate by Paper Origin Generation")
ax.set_xticks(x)
ax.set_xticklabels([f"Node {n}" for n in range(MAX_NODES)])
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "rate_by_origin.png"), dpi=150)
print("Saved rate_by_origin.png + rate_by_origin.jsonl")

# ── 5. Semantic Similarity vs. Citation Count ────────────────────────────────

paper_similarity = {}
paper_exp_citations = defaultdict(int)

for node in range(MAX_NODES):
    path = f"{BASE}/semantic_output/node_{node}/node_{node}_comparison.jsonl"
    with open(path) as f:
        for line in f:
            e = json.loads(line.strip())
            pid = e["id"]
            if pid not in paper_similarity:
                paper_similarity[pid] = e["similarity_to_topic"]
            paper_exp_citations[pid] += e["experiment_citations"]

sim_data = []
for pid in paper_similarity:
    sim_data.append({
        "id": pid,
        "similarity_to_topic": paper_similarity[pid],
        "total_experiment_citations": paper_exp_citations[pid],
        "origin": get_origin(pid),
    })

with open(os.path.join(OUTPUT_DIR, "similarity_vs_citations.jsonl"), "w") as f:
    for entry in sim_data:
        f.write(json.dumps(entry) + "\n")

sims = np.array([e["similarity_to_topic"] for e in sim_data])
cites = np.array([e["total_experiment_citations"] for e in sim_data])
is_seed = np.array([e["id"].startswith("SEED_") for e in sim_data])
r = float(np.corrcoef(sims, cites)[0, 1])

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(sims[is_seed], cites[is_seed], alpha=0.7, s=40, label="Seed Papers", zorder=3)
ax.scatter(sims[~is_seed], cites[~is_seed], alpha=0.5, s=25, label="Generated Papers", zorder=2)
z = np.polyfit(sims, cites, 1)
p = np.poly1d(z)
x_line = np.linspace(sims.min(), sims.max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f"Trend (r={r:.4f})")
ax.set_xlabel("Semantic Similarity to Topic")
ax.set_ylabel("Total Experiment Citations")
ax.set_title(f"Semantic Similarity vs Citation Count (Pearson r={r:.4f})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "similarity_vs_citations.png"), dpi=150)
print("Saved similarity_vs_citations.png + similarity_vs_citations.jsonl")

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
