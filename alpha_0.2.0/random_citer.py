import json, os, random
from collections import defaultdict

SEED = 42
TOTAL_NODES = 6
NODE_SIZE = 30
PER_NODE_K = {0: 26, 1: 25, 2: 23, 3: 23, 4: 22, 5: 23}
BASE = "alpha_0.2.0"
OUT = f"{BASE}/random_output"

CITATION_COUNTS = defaultdict(int)
AUTHOR_YEAR_TO_ID = {}

def build_lookup():
    for node in range(TOTAL_NODES):
        with open(f"{BASE}/prompts/N_{node}_inputs.jsonl") as f:
            for raw in f:
                line = json.loads(raw)
                AUTHOR_YEAR_TO_ID[(line["author"], line["year"])] = line["id"]

    with open(f"{BASE}/output/seed/seed.jsonl") as f:
        for raw in f:
            p = json.loads(raw)
            AUTHOR_YEAR_TO_ID[(p["author"], p["year"])] = p["id"]

def run_node(node: int):
    k = PER_NODE_K[node]
    node_citations = defaultdict(int)

    os.makedirs(f"{OUT}/node_{node}", exist_ok=True)

    with open(f"{BASE}/prompts/N_{node}_inputs.jsonl") as fin, \
         open(f"{OUT}/node_{node}/node_{node}.jsonl", "w") as fout:

        for raw in fin:
            line = json.loads(raw)
            cited_ids = random.sample(line["papers_seen_id"], k)

            for cid in cited_ids:
                node_citations[cid] += 1

            fout.write(json.dumps({
                "id": line["id"],
                "author": line["author"],
                "year": line["year"],
                "type": line["type"],
                "papers_seen_id": line["papers_seen_id"],
                "citation_ids": cited_ids
            }) + "\n")

    with open(f"{OUT}/node_{node}/node_{node}_stats.jsonl", "w") as f:
        for pid, count in node_citations.items():
            f.write(json.dumps({pid: count}) + "\n")
            print(f"{pid}: {count}")

    for pid, count in node_citations.items():
        CITATION_COUNTS[pid] += count

def run():
    random.seed(SEED)

    os.makedirs(f"{OUT}/citation_counts", exist_ok=True)
    os.makedirs(f"{OUT}/master", exist_ok=True)

    build_lookup()

    for node in range(TOTAL_NODES):
        print(f"\n{'='*30}\nNode {node}\n{'='*30}\n")
        run_node(node)

    with open(f"{OUT}/citation_counts/citation_counts.jsonl", "w") as f:
        for pid, count in CITATION_COUNTS.items():
            f.write(json.dumps({pid: count}) + "\n")

    with open(f"{OUT}/master/kv_pairs.jsonl", "w") as f:
        for (author, year), pid in AUTHOR_YEAR_TO_ID.items():
            f.write(json.dumps({"author": author, "year": year, "id": pid}) + "\n")

    print("Finished random citer")

if __name__ == "__main__":
    run()
