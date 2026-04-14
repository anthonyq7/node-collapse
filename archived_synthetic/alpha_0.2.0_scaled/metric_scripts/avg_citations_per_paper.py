import json

BASE = "archived_synthetic/alpha_0.2.0_scaled"
MAX_NODES = 12
PAPER_SET_LENGTH = 30


def avg_citations_for_node(node: int) -> dict:
    counts = []
    path = f"{BASE}/output/node_{node}/node_{node}.jsonl"
    with open(path) as f:
        for line in f:
            paper = json.loads(line)
            counts.append(len(paper.get("citation_ids", [])))
    if not counts:
        return {"node": node, "papers": 0, "avg": 0.0, "min": 0, "max": 0}
    return {
        "node": node,
        "papers": len(counts),
        "avg": round(sum(counts) / len(counts), 2),
        "min": min(counts),
        "max": max(counts),
    }


def main():
    print(f"{'Node':<6} {'Papers':<8} {'Avg Citations':<16} {'Min':<6} {'Max':<6} {'Avg/Shown':<12}")
    print("-" * 56)
    for node in range(MAX_NODES):
        r = avg_citations_for_node(node)
        ratio = round(r["avg"] / PAPER_SET_LENGTH, 3)
        print(
            f"{r['node']:<6} {r['papers']:<8} {r['avg']:<16} {r['min']:<6} {r['max']:<6} {ratio:<12}"
        )


if __name__ == "__main__":
    main()
