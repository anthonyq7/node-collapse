import json
import os
import random
from collections import defaultdict

SEED = 42
BASE = "archived_synthetic/alpha_10_cap"
OUT = f"{BASE}/random_output"
MAX_NODES = 12
CITATION_COUNTS = defaultdict(int)


def main():
    random.seed(SEED)

    os.makedirs(f"{OUT}/citation_counts", exist_ok=True)
    os.makedirs(f"{OUT}/master", exist_ok=True)

    for node in range(MAX_NODES):
        os.makedirs(f"{OUT}/node_{node}", exist_ok=True)

        exp_path = f"{BASE}/output/node_{node}/node_{node}.jsonl"
        with open(exp_path) as f:
            papers = [json.loads(line) for line in f]

        node_citations = defaultdict(int)
        out_rows = []

        for paper in papers:
            papers_seen_id = paper.get("papers_seen_id", [])
            k = len(paper.get("citation_ids", []))

            cited_ids = random.sample(papers_seen_id, k)

            for cid in cited_ids:
                node_citations[cid] += 1
                CITATION_COUNTS[cid] += 1

            out_rows.append({
                "id": paper["id"],
                "author": paper.get("author"),
                "year": paper.get("year"),
                "type": paper.get("type"),
                "papers_seen_id": papers_seen_id,
                "citation_ids": cited_ids,
            })

        node_path = f"{OUT}/node_{node}/node_{node}.jsonl"
        with open(node_path, "w") as f:
            for row in out_rows:
                f.write(json.dumps(row) + "\n")

        stats_path = f"{OUT}/node_{node}/node_{node}_stats.jsonl"
        with open(stats_path, "w") as f:
            for pid, count in sorted(node_citations.items()):
                f.write(json.dumps({pid: count}) + "\n")

        print(f"Node {node}: wrote {node_path}, {stats_path}")

    with open(f"{OUT}/citation_counts/citation_counts.jsonl", "w") as f:
        for pid, count in sorted(CITATION_COUNTS.items()):
            f.write(json.dumps({pid: count}) + "\n")

    with open(f"{BASE}/output/master/kv_pairs.jsonl") as src:
        with open(f"{OUT}/master/kv_pairs.jsonl", "w") as dst:
            dst.write(src.read())


if __name__ == "__main__":
    main()
