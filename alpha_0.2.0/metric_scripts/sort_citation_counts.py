import json

INPUT_PATH = "alpha_0.2.0/output/citation_counts/citation_counts.jsonl"
OUTPUT_PATH = "alpha_0.2.0/output/citation_counts/citation_counts_ordered.jsonl"

valid_ids = set()
with open("alpha_0.2.0/output/master/kv_pairs.jsonl") as f:
    for line in f:
        data = json.loads(line.strip())
        valid_ids.add(data["id"])

citations = {}
with open(INPUT_PATH, "r") as f:
    for line in f:
        data = json.loads(line.strip())
        for paper_id, count in data.items():
            if paper_id in valid_ids:
                citations[paper_id] = count

sorted_citations = sorted(citations.items(), key=lambda x: x[1], reverse=True)

with open(OUTPUT_PATH, "w") as f:
    for paper_id, count in sorted_citations:
        f.write(json.dumps({"id": paper_id, "citations": count}) + "\n")

print(f"Sorted {len(sorted_citations)} papers by citation count:\n")
for paper_id, count in sorted_citations:
    print(f"  {paper_id}: {count}")

print(f"\nSaved to {OUTPUT_PATH}")
