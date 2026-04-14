# Random Citer Extension — Plan

## Goal

Extend the current [random_citer.py](random_citer.py) so that:

1. The pool grows after every node using `add_to_pool(node)`.
2. Citation counts per paper come from the experiment via `get_node_citation(node)` (list of k values).
3. Each node’s stats are written (`node_{n}_stats.jsonl`) so you can compare citation counts after.
4. No new Python imports (no importing other project modules); only read experiment JSONL to get k counts.

---

## Current Behavior (to extend)

- **TOTAL_POOL**: global list, built incrementally.
- **create_seed_pool()**: appends `SEED_0` … `SEED_{NODE_SIZE-1}` to `TOTAL_POOL`.
- **add_to_pool(node)**: appends `N{node}P0` … `N{node}P{NODE_SIZE-1}` to `TOTAL_POOL`.
- **get_node_citation(node)**: reads experiment `node_{node}.jsonl`, returns a list of 120 integers — the citation count (k) for each paper in that node.  
  - **Bug to fix**: path uses `node{node}`; should be `node_{node}`.  
  - **What to use for k**: use `len(paper["citation_ids"])` (or keep `len(paper["citations"])`) so the list is one k per paper in order.

So at node 0 the pool is seed only; after node 0 we call `add_to_pool(0)` so at node 1 the pool is seed + node 0 papers; and so on.

---

## Intended Flow

1. **Seed pool**  
   `create_seed_pool()` so `TOTAL_POOL` = [SEED_0, …, SEED_119].

2. **For each node** `node = 0 .. MAX_NODES-1`:
   - **Get k per paper**  
     `citation_counts = get_node_citation(node)` → list of 120 integers (same order as papers in experiment node).
   - **Load experiment papers for this node**  
     Read `BASE/output/node_{node}/node_{node}.jsonl` to get paper order and metadata (id, author, year, type, papers_seen_id). We need this to know paper ids (e.g. N0P0 … N0P119) and optionally papers_seen_id for the output.
   - **For each paper index** `i` in 0..119:
     - `k = citation_counts[i]`
     - Randomly sample k ids from `TOTAL_POOL` (use `random.choices(TOTAL_POOL, k=k)` to allow replacement when k &gt; pool size).
     - Assign these as this paper’s `citation_ids`.
   - **Write** `OUT/node_{node}/node_{node}.jsonl`: one JSON object per paper with at least `id`, `papers_seen_id`, `citation_ids` (and optionally author, year, type from experiment).
   - **Aggregate node stats**  
     For each paper_id in the sampled citation_ids, increment a counter. Write `OUT/node_{node}/node_{node}_stats.jsonl`: one line per paper_id, `{paper_id: count}` (same format as experiment).
   - **Grow pool**  
     `add_to_pool(node)` so the next node has a larger pool.

3. **Optional**  
   - Write `OUT/citation_counts/citation_counts.jsonl` (global counts across all nodes) and copy `OUT/master/kv_pairs.jsonl` from experiment if you want random_output self-contained for metric scripts.
   - Do **not** write any `node_{n}_exposure.jsonl`; remove existing ones under `random_output` so exposure is not used for the random baseline.

---

## Fixes and Conventions

| Item | Change |
|------|--------|
| Path in `get_node_citation` | Use `f"{BASE}/output/node_{node}/node_{node}.jsonl"` (add underscore). |
| k from experiment | Use `len(paper["citation_ids"])` (or keep `paper["citations"]`) so the list length matches experiment. |
| Sampling | `random.choices(TOTAL_POOL, k=k)` so sampling with replacement is valid when k is large. |
| Output dirs | Ensure `OUT/node_{node}/` exists before writing (e.g. `os.makedirs(..., exist_ok=True)`). |

---

## Summary

- Extend your existing loop: use `get_node_citation(node)` for the list of k values, sample k from `TOTAL_POOL` per paper, write node JSONL and node stats, then `add_to_pool(node)`.
- Fix the experiment path in `get_node_citation`.
- No new imports; only read experiment node JSONL to get k counts and paper metadata.
- After running, you get per-node stats and can compare citation count distributions to the experiment.
