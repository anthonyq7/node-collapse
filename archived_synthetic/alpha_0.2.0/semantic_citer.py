import asyncio, json, os, math
from dotenv import load_dotenv
from openai import AsyncOpenAI
from collections import defaultdict

load_dotenv()

TOTAL_NODES = 6
NODE_SIZE = 30
PER_NODE_K = {0: 26, 1: 25, 2: 23, 3: 23, 4: 22, 5: 23}
TOPIC = "Knowledge distillation or model compression in deep learning or NLP"
EMBED_MODEL = "text-embedding-3-small"
BASE = "archived_synthetic/alpha_0.2.0"
OUT = f"{BASE}/semantic_output"

client = AsyncOpenAI()
CITATION_COUNTS = defaultdict(int)
AUTHOR_YEAR_TO_ID = {}
EMBEDDINGS = {}  # (author, year) -> list[float]


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


def collect_unique_papers():
    """Gather all unique (author, year) -> text from prompt files."""
    papers = {}
    for node in range(TOTAL_NODES):
        with open(f"{BASE}/prompts/N_{node}_inputs.jsonl") as f:
            for raw in f:
                line = json.loads(raw)
                for p in line["papers_seen"]:
                    key = (p["author"], p["year"])
                    if key not in papers:
                        papers[key] = f"{p.get('title', '')} {p.get('abstract', '')}"
    return papers


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed via OpenAI, chunking at 2048 per request."""
    all_vecs = []
    for i in range(0, len(texts), 2048):
        resp = await client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+2048])
        all_vecs.extend(d.embedding for d in resp.data)
    return all_vecs


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def run_node(node: int, topic_emb: list[float]):
    k = PER_NODE_K[node]
    node_citations = defaultdict(int)

    os.makedirs(f"{OUT}/node_{node}", exist_ok=True)

    with open(f"{BASE}/prompts/N_{node}_inputs.jsonl") as fin, \
         open(f"{OUT}/node_{node}/node_{node}.jsonl", "w") as fout:

        for raw in fin:
            line = json.loads(raw)
            papers_seen = line["papers_seen"]
            papers_seen_id = line["papers_seen_id"]

            scored = []
            for i, p in enumerate(papers_seen):
                key = (p["author"], p["year"])
                emb = EMBEDDINGS.get(key)
                sim = cosine_similarity(topic_emb, emb) if emb is not None else 0.0
                scored.append((sim, papers_seen_id[i], key))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = scored[:k]

            cited_ids = [pid for _, pid, _ in top_k]
            citations = [(author, year) for _, _, (author, year) in top_k]

            for cid in cited_ids:
                node_citations[cid] += 1

            fout.write(json.dumps({
                "id": line["id"],
                "author": line["author"],
                "year": line["year"],
                "type": line["type"],
                "papers_seen_id": papers_seen_id,
                "citation_ids": cited_ids,
                "citations": citations
            }) + "\n")

    with open(f"{OUT}/node_{node}/node_{node}_stats.jsonl", "w") as f:
        for pid, count in node_citations.items():
            f.write(json.dumps({pid: count}) + "\n")
            print(f"  {pid}: {count}")

    for pid, count in node_citations.items():
        CITATION_COUNTS[pid] += count

    return node_citations


def write_comparison(node: int, semantic_counts: dict, topic_emb: list[float]):
    """Join semantic citation counts with real experiment counts and similarity scores."""
    experiment_counts = {}
    with open(f"{BASE}/output/node_{node}/node_{node}_stats.jsonl") as f:
        for raw in f:
            entry = json.loads(raw)
            for pid, count in entry.items():
                experiment_counts[pid] = count

    all_ids = set(semantic_counts.keys()) | set(experiment_counts.keys())

    rows = []
    for pid in sorted(all_ids):
        key = next(((a, y) for (a, y), v in AUTHOR_YEAR_TO_ID.items() if v == pid), None)
        sim = cosine_similarity(topic_emb, EMBEDDINGS[key]) if key and key in EMBEDDINGS else None
        rows.append({
            "id": pid,
            "semantic_citations": semantic_counts.get(pid, 0),
            "experiment_citations": experiment_counts.get(pid, 0),
            "similarity_to_topic": round(sim, 4) if sim is not None else None
        })

    rows.sort(key=lambda r: r["experiment_citations"], reverse=True)

    with open(f"{OUT}/node_{node}/node_{node}_comparison.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


async def run():
    os.makedirs(f"{OUT}/citation_counts", exist_ok=True)
    os.makedirs(f"{OUT}/master", exist_ok=True)

    build_lookup()

    print("Collecting unique papers...")
    unique_papers = collect_unique_papers()
    keys = list(unique_papers.keys())
    texts = [unique_papers[k] for k in keys]

    print(f"Embedding {len(texts)} unique papers + topic...")
    vecs = await embed_texts([TOPIC] + texts)
    topic_emb = vecs[0]
    for i, key in enumerate(keys):
        EMBEDDINGS[key] = vecs[i + 1]

    for node in range(TOTAL_NODES):
        print(f"\n{'='*30}\nNode {node}\n{'='*30}")
        semantic_counts = run_node(node, topic_emb)
        write_comparison(node, semantic_counts, topic_emb)

    with open(f"{OUT}/citation_counts/citation_counts.jsonl", "w") as f:
        for pid, count in CITATION_COUNTS.items():
            f.write(json.dumps({pid: count}) + "\n")

    with open(f"{OUT}/master/kv_pairs.jsonl", "w") as f:
        for (author, year), pid in AUTHOR_YEAR_TO_ID.items():
            f.write(json.dumps({"author": author, "year": year, "id": pid}) + "\n")

    print("\nFinished semantic citer")


if __name__ == "__main__":
    asyncio.run(run())
