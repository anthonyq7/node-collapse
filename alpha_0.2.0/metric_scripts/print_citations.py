import json
from collections import defaultdict

MAX_NODES = 6

valid_ids = set()
with open("alpha_0.2.0/output/master/kv_pairs.jsonl") as f:
    for line in f:
        data = json.loads(line.strip())
        valid_ids.add(data["id"])

citations = defaultdict(int)

for node in range(MAX_NODES):
    with open(f"alpha_0.2.0/output/node_{node}/node_{node}_stats.jsonl") as f:
        for line in f:
            data = json.loads(line.strip())
            for k, v in data.items():
                if k in valid_ids:
                    citations[k] += v

sorted_citations = sorted(citations.items(), key=lambda x: x[1], reverse=True)

for paper_id, count in sorted_citations:
    print(f"{paper_id}: {count}")
