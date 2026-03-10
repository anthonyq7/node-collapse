import json
import os
import random
from collections import defaultdict

SEED = 42
BASE = "alpha_0.2.0_scaled"
OUT = f"{BASE}/random_output"
MAX_NODES = 12
NODE_SIZE = 120

CITATION_COUNTS = defaultdict(int)
AUTHOR_YEAR_TO_ID = {}


def build_lookup():
    for node in range(MAX_NODES):
        path = f"{BASE}/prompts/N_{node}_inputs.jsonl"
        with open(path) as f:
            for raw in f:
                line = json.loads(raw)
                AUTHOR_YEAR_TO_ID[(line["author"], line["year"])] = line["id"]

    seed_path = f"{BASE}/output/seed/seed.jsonl"
    if os.path.exists(seed_path):
        with open(seed_path) as f:
            for raw in f:
                p = json.loads(raw)
                AUTHOR_YEAR_TO_ID[(p["author"], p["year"])] = p["id"]


def load_main_k_per_paper(node: int) -> dict:
    """
    Load how many citations each paper made in the main experiment for this node.
    Returns mapping: paper_id -> k_main.
    """
    k_by_paper: dict = {}
    path = f"{BASE}/output/node_{node}/node_{node}.jsonl"
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            paper_id = data["id"]
            k_by_paper[paper_id] = len(data.get("citation_ids", []))
    return k_by_paper


def run_node(node: int):
    k_by_paper = load_main_k_per_paper(node)
    node_citations = defaultdict(int)

    os.makedirs(f"{OUT}/node_{node}", exist_ok=True)

    prompt_path = f"{BASE}/prompts/N_{node}_inputs.jsonl"
    out_path = f"{OUT}/node_{node}/node_{node}.jsonl"
    with open(prompt_path) as fin, open(out_path, "w") as fout:
        for raw in fin:
            line = json.loads(raw)
            paper_id = line["id"]
            seen = line["papers_seen_id"]

            k = k_by_paper.get(paper_id, 0)
            if k > len(seen):
                k = len(seen)
            if k <= 0 or len(seen) == 0:
                cited_ids = []
            else:
                cited_ids = random.sample(seen, k)

            for cid in cited_ids:
                node_citations[cid] += 1

            row = {
                "id": line["id"],
                "author": line["author"],
                "year": line["year"],
                "type": line["type"],
                "papers_seen_id": line["papers_seen_id"],
                "citation_ids": cited_ids,
            }
            fout.write(json.dumps(row) + "\n")

    with open(f"{OUT}/node_{node}/node_{node}_stats.jsonl", "w") as f:
        for pid, count in node_citations.items():
            f.write(json.dumps({pid: count}) + "\n")

    for pid, count in node_citations.items():
        CITATION_COUNTS[pid] += count


def run():
    random.seed(SEED)

    os.makedirs(f"{OUT}/citation_counts", exist_ok=True)
    os.makedirs(f"{OUT}/master", exist_ok=True)

    build_lookup()

    for node in range(MAX_NODES):
        print(f"Running random citer for node {node}")
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
