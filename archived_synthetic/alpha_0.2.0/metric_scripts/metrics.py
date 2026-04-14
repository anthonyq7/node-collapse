import json
import os

MAX_NODES = 6
SEED_COUNT = 30
PAPERS_PER_NODE = 30

valid_ids = set()
with open("archived_synthetic/alpha_0.2.0/output/master/kv_pairs.jsonl") as f:
    for line in f:
        data = json.loads(line.strip())
        valid_ids.add(data["id"])

def get_available_papers(node: int) -> int:
    return SEED_COUNT + (PAPERS_PER_NODE * node)

def load_node_stats(node: int) -> dict:
    stats = {}
    path = f"archived_synthetic/alpha_0.2.0/output/node_{node}/node_{node}_stats.jsonl"
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            for k, v in data.items():
                if k in valid_ids:
                    stats[k] = v
    return stats

def gini(values: list) -> float:
    if not values or sum(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumsum) / (n * sum(sorted_vals)) - (n + 1) / n

def analyze_node(node: int) -> dict:
    stats = load_node_stats(node)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    
    total_citations = sum(stats.values())
    top_5 = sorted_stats[:5]
    top_5_citations = sum(count for _, count in top_5)
    top_5_pct = (top_5_citations / total_citations * 100) if total_citations > 0 else 0
    
    available = get_available_papers(node)
    cited_count = len(stats)
    uncited_count = available - cited_count
    
    all_values = list(stats.values()) + [0] * uncited_count
    gini_coef = gini(all_values)
    
    return {
        "node": node,
        "total_citations": total_citations,
        "unique_papers_cited": len(stats),
        "available_papers": available,
        "top_5": [{"id": id, "count": count} for id, count in top_5],
        "top_5_citations": top_5_citations,
        "top_5_percentage": round(top_5_pct, 2),
        "gini": round(gini_coef, 4)
    }

def main():
    os.makedirs("archived_synthetic/alpha_0.2.0/output/master", exist_ok=True)
    
    results = []
    for node in range(MAX_NODES):
        result = analyze_node(node)
        results.append(result)
        
        print(f"\n{'='*40}")
        print(f"Node {node}")
        print(f"{'='*40}")
        print(f"Total citations: {result['total_citations']}")
        print(f"Unique papers cited: {result['unique_papers_cited']}")
        print(f"Available papers: {result['available_papers']}")
        print(f"Top 5 articles:")
        for item in result['top_5']:
            print(f"  {item['id']}: {item['count']}")
        print(f"Top 5 citations: {result['top_5_citations']} / {result['total_citations']}")
        print(f"Top 5 percentage: {result['top_5_percentage']}%")
        print(f"Gini coefficient: {result['gini']}")
    
    with open("archived_synthetic/alpha_0.2.0/output/master/generation_stats.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\n{'='*40}")
    print("Saved to alpha_0.2.0/output/master/generation_stats.jsonl")

if __name__ == "__main__":
    main()
