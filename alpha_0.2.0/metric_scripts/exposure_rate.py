import json
import os
from collections import defaultdict

MAX_NODES = 6
OUTPUT_PATH = "alpha_0.2.0/output/master/exposure_rate.jsonl"

valid_ids = set()
with open("alpha_0.2.0/output/master/kv_pairs.jsonl") as f:
    for line in f:
        data = json.loads(line.strip())
        valid_ids.add(data["id"])

citations = defaultdict(int)
exposures = defaultdict(int)

for node in range(MAX_NODES):
    stats_path = f"alpha_0.2.0/output/node_{node}/node_{node}_stats.jsonl"
    with open(stats_path) as f:
        for line in f:
            data = json.loads(line.strip())
            for paper_id, count in data.items():
                if paper_id in valid_ids:
                    citations[paper_id] += count

    node_path = f"alpha_0.2.0/output/node_{node}/node_{node}.jsonl"
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

os.makedirs("alpha_0.2.0/output/master", exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    for entry in results:
        f.write(json.dumps(entry) + "\n")

print(f"Exposure rates for {len(results)} papers:\n")
for entry in results:
    print(f"  {entry['id']}: {entry['citations']} citations / {entry['exposures']} exposures = {entry['rate']}")

print(f"\nSaved to {OUTPUT_PATH}")
