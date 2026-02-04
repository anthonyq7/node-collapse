import json
import os

MAX_NODES = 5  # 0 through 4
SEED_COUNT = 50
PAPERS_PER_NODE = 30

def get_available_papers(node: int) -> int:
    """Get total papers available at a given node."""
    # Node 0: 50 seed papers
    # Node 1: 50 + 30 = 80
    # Node N: 50 + 30*N
    return SEED_COUNT + (PAPERS_PER_NODE * node)

def load_node_stats(node: int) -> dict:
    """Load citation counts from a node's stats file."""
    stats = {}
    path = f"output/node_{node}/node_{node}_stats.jsonl"
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            for k, v in data.items():
                stats[k] = v
    return stats

def gini(values: list) -> float:
    """Calculate Gini coefficient for a list of values."""
    if not values or sum(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumsum) / (n * sum(sorted_vals)) - (n + 1) / n

def analyze_node(node: int) -> dict:
    """Analyze citation stats for a single node."""
    stats = load_node_stats(node)
    
    #sort by citation count descending
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    
    total_citations = sum(stats.values())
    top_5 = sorted_stats[:5]
    top_5_citations = sum(count for _, count in top_5)
    top_5_pct = (top_5_citations / total_citations * 100) if total_citations > 0 else 0
    
    # For Gini: include all available papers (those with 0 citations too)
    available = get_available_papers(node)
    cited_count = len(stats)
    uncited_count = available - cited_count
    
    # Build full values list including zeros for uncited papers
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
    os.makedirs("output/master", exist_ok=True)
    
    results = []
    for node in range(MAX_NODES):
        result = analyze_node(node)
        results.append(result)
        
        #print summary
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
    
    #save to JSONL
    with open("output/master/generation_stats.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\n{'='*40}")
    print("Saved to output/master/generation_stats.jsonl")

if __name__ == "__main__":
    main()
