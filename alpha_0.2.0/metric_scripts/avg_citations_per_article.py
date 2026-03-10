import json
import os

MAX_NODES = 6
OUTPUT_DIR = "alpha_0.2.0/output/avg_citations_per_article"

results = []
all_citations = 0
all_papers = 0

for node in range(MAX_NODES):
    node_path = f"alpha_0.2.0/output/node_{node}/node_{node}.jsonl"
    total = 0
    count = 0
    with open(node_path) as f:
        for line in f:
            data = json.loads(line.strip())
            total += len(data.get("citation_ids", []))
            count += 1

    avg = round(total / count, 2) if count > 0 else 0.0
    results.append({"node": node, "papers": count, "total_citations": total, "avg_citations": avg})
    all_citations += total
    all_papers += count

    print(f"Node {node}: {count} papers, {total} citations, avg {avg}")

all_avg = round(all_citations / all_papers, 2) if all_papers > 0 else 0.0
all_entry = {"node": "all", "papers": all_papers, "total_citations": all_citations, "avg_citations": all_avg}
print(f"\nOverall: {all_papers} papers, {all_citations} citations, avg {all_avg}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "avg_citations_per_article.jsonl"), "w") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")
    f.write(json.dumps(all_entry) + "\n")

print(f"\nSaved to {OUTPUT_DIR}/avg_citations_per_article.jsonl")
