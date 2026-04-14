import json
from collections import defaultdict

MAX_NODES = 6

for node in range(MAX_NODES):
    citations = defaultdict(int)
    exposures = defaultdict(int)

    stats_path = f"archived_synthetic/alpha/output/node_{node}/node_{node}_stats.jsonl"
    with open(stats_path) as f:
        for line in f:
            data = json.loads(line.strip())
            for paper_id, count in data.items():
                citations[paper_id] += count

    node_path = f"archived_synthetic/alpha/output/node_{node}/node_{node}.jsonl"
    with open(node_path) as f:
        for line in f:
            data = json.loads(line.strip())
            papers_seen = data.get("papers_seen", [])
            for paper_id in papers_seen:
                exposures[paper_id] += 1

    all_ids = set(citations.keys()) | set(exposures.keys())

    results = []
    for paper_id in all_ids:
        cite_count = citations[paper_id]
        expose_count = exposures[paper_id]

        if expose_count > 0:
            rate = cite_count / expose_count
        else:
            rate = 0.0

        entry = {
            "id": paper_id,
            "citations": cite_count,
            "exposures": expose_count,
            "rate": round(rate, 4)
        }
        results.append(entry)

    results.sort(key=lambda x: x["rate"], reverse=True)

    output_path = f"archived_synthetic/alpha/output/node_{node}/node_{node}_exposure.jsonl"
    with open(output_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\n=== Node {node} ===")
    print(f"Papers with exposure data: {len(results)}")
    for entry in results:
        print(f"  {entry['id']}: {entry['citations']} citations / {entry['exposures']} exposures = {entry['rate']}")
    print(f"Saved to {output_path}")
