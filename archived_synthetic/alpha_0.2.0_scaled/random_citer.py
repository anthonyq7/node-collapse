import json
import os
import random
from collections import defaultdict

SEED = 42
BASE = "archived_synthetic/alpha_0.2.0_scaled"
OUT = f"{BASE}/random_output"
MAX_NODES = 12
CITATION_COUNTS = defaultdict(int)


def load_all_ids() -> list:
    ids = []
    with open(f"{BASE}/output/master/kv_pairs.jsonl") as f:
        for line in f:
            ids.append(json.loads(line)["id"])
    return ids


def build_pool(all_ids: list, node: int) -> list:
    """All SEED_* papers + N0P* through N(node-1)P* papers."""
    return [pid for pid in all_ids
            if pid.startswith("SEED_") or
            any(pid.startswith(f"N{n}P") for n in range(node))]


def main():
    random.seed(SEED)

    os.makedirs(f"{OUT}/citation_counts", exist_ok=True)
    os.makedirs(f"{OUT}/master", exist_ok=True)

    all_ids = load_all_ids()

    for node in range(MAX_NODES):
        os.makedirs(f"{OUT}/node_{node}", exist_ok=True)

        pool = build_pool(all_ids, node)

        exp_path = f"{BASE}/output/node_{node}/node_{node}.jsonl"
        with open(exp_path) as f:
            papers = [json.loads(line) for line in f]

        node_citations = defaultdict(int)
        out_rows = []

        for paper in papers:
            papers_seen_id = paper.get("papers_seen_id", [])
            k = min(len(paper.get("citation_ids", [])), len(pool))

            cited_ids = random.sample(pool, k) if k > 0 else []

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

        print(f"Node {node} (pool={len(pool)}): wrote {node_path}, {stats_path}")

    with open(f"{OUT}/citation_counts/citation_counts.jsonl", "w") as f:
        for pid, count in sorted(CITATION_COUNTS.items()):
            f.write(json.dumps({pid: count}) + "\n")

    with open(f"{BASE}/output/master/kv_pairs.jsonl") as src:
        with open(f"{OUT}/master/kv_pairs.jsonl", "w") as dst:
            dst.write(src.read())


if __name__ == "__main__":
    main()
