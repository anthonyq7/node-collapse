import json
from collections import defaultdict

MAX_NODES = 5
citations = defaultdict(int)

for node in range(MAX_NODES):
    with open(f"output/node_{node}/node_{node}_stats.jsonl") as f:
        for line in f:
            data = json.loads(line.strip())
            for k, v in data.items():
                citations[k] += v

sorted_citations = sorted(citations.items(), key=lambda x: x[1], reverse=True)

for paper_id, count in sorted_citations:
    print(f"{paper_id}: {count}")
