import csv, json, re
from collections import defaultdict

TOTAL_NODES = 6
BASE = "archived_synthetic/alpha_0.2.0"

def get_citations(text: str):
    """Same regex as experiment.py to ensure parity."""
    results = set()
    for group in re.findall(r"\(([^)]+)\)", text):
        for name, year in re.findall(r"([A-Z][a-z]+),\s*(\d{4})", group):
            results.add((name, int(year)))
    return results


def load_ground_truth():
    mapping = {}
    with open(f"{BASE}/output/master/kv_pairs.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            key = (entry["author"], entry["year"])
            mapping[key] = entry["id"]
    return mapping


def verify():
    gt = load_ground_truth()
    print(f"Ground truth: {len(gt)} (author, year) -> id mappings\n")

    overall = {
        "total_citations": 0,
        "valid": 0,
        "known_but_unseen": 0,
        "hallucinated": 0,
        "extraction_mismatches": 0,
        "citation_id_mismatches": 0,
    }
    all_issues = []
    node_rows = []
    paper_rows = []

    for node in range(TOTAL_NODES):
        path = f"{BASE}/output/node_{node}/node_{node}.jsonl"
        node_stats = {
            "total_citations": 0,
            "valid": 0,
            "known_but_unseen": 0,
            "hallucinated": 0,
            "extraction_mismatches": 0,
            "citation_id_mismatches": 0,
            "papers": 0,
        }
        node_issues = []

        with open(path) as f:
            for raw in f:
                paper = json.loads(raw)
                pid = paper["id"]
                body = paper.get("body", "")
                stored_citations = {(c[0], c[1]) for c in paper["citations"]}
                stored_ids = set(paper["citation_ids"])
                seen_ids = set(paper["papers_seen_id"])

                node_stats["papers"] += 1

                extracted = get_citations(body)
                extraction_ok = extracted == stored_citations
                if not extraction_ok:
                    node_stats["extraction_mismatches"] += 1
                    only_extracted = extracted - stored_citations
                    only_stored = stored_citations - extracted
                    node_issues.append({
                        "paper": pid,
                        "type": "extraction_mismatch",
                        "only_in_body": sorted([list(c) for c in only_extracted]),
                        "only_in_stored": sorted([list(c) for c in only_stored]),
                    })

                p_valid = 0
                p_unseen = 0
                p_hallucinated = 0
                p_hallucinated_list = []

                resolved_ids = set()
                for cite in extracted:
                    node_stats["total_citations"] += 1
                    resolved_id = gt.get(cite)

                    if resolved_id is None:
                        node_stats["hallucinated"] += 1
                        p_hallucinated += 1
                        p_hallucinated_list.append(f"{cite[0]}, {cite[1]}")
                        node_issues.append({
                            "paper": pid,
                            "type": "hallucinated",
                            "citation": list(cite),
                        })
                    elif resolved_id not in seen_ids:
                        node_stats["known_but_unseen"] += 1
                        p_unseen += 1
                        resolved_ids.add(resolved_id)
                        node_issues.append({
                            "paper": pid,
                            "type": "known_but_unseen",
                            "citation": list(cite),
                            "resolved_id": resolved_id,
                        })
                    else:
                        node_stats["valid"] += 1
                        p_valid += 1
                        resolved_ids.add(resolved_id)

                expected_ids = set()
                for cite in stored_citations:
                    rid = gt.get(cite)
                    if rid is not None:
                        expected_ids.add(rid)

                ids_ok = expected_ids == stored_ids
                if not ids_ok:
                    node_stats["citation_id_mismatches"] += 1
                    node_issues.append({
                        "paper": pid,
                        "type": "citation_id_mismatch",
                        "expected_ids": sorted(expected_ids),
                        "stored_ids": sorted(stored_ids),
                        "missing": sorted(expected_ids - stored_ids),
                        "extra": sorted(stored_ids - expected_ids),
                    })

                p_total = p_valid + p_unseen + p_hallucinated
                paper_rows.append({
                    "node": node,
                    "paper_id": pid,
                    "total_citations": p_total,
                    "valid": p_valid,
                    "known_but_unseen": p_unseen,
                    "hallucinated": p_hallucinated,
                    "hallucinated_refs": "; ".join(p_hallucinated_list) if p_hallucinated_list else "",
                    "valid_pct": round(p_valid / p_total * 100, 2) if p_total else 0,
                    "hallucinated_pct": round(p_hallucinated / p_total * 100, 2) if p_total else 0,
                    "extraction_ok": extraction_ok,
                    "citation_ids_ok": ids_ok,
                })

        total = node_stats["total_citations"]
        valid_pct = (node_stats["valid"] / total * 100) if total else 0
        hall_pct = (node_stats["hallucinated"] / total * 100) if total else 0
        unseen_pct = (node_stats["known_but_unseen"] / total * 100) if total else 0

        node_rows.append({
            "node": node,
            "papers": node_stats["papers"],
            "total_citations": total,
            "valid": node_stats["valid"],
            "known_but_unseen": node_stats["known_but_unseen"],
            "hallucinated": node_stats["hallucinated"],
            "valid_pct": round(valid_pct, 2),
            "known_but_unseen_pct": round(unseen_pct, 2),
            "hallucinated_pct": round(hall_pct, 2),
            "extraction_mismatches": node_stats["extraction_mismatches"],
            "citation_id_mismatches": node_stats["citation_id_mismatches"],
        })

        print(f"{'='*60}")
        print(f"Node {node}  ({node_stats['papers']} papers, {total} citations)")
        print(f"{'='*60}")
        print(f"  Valid (in pool):        {node_stats['valid']:>5}  ({valid_pct:.1f}%)")
        print(f"  Known but not in pool:  {node_stats['known_but_unseen']:>5}  ({unseen_pct:.1f}%)")
        print(f"  Hallucinated:           {node_stats['hallucinated']:>5}  ({hall_pct:.1f}%)")
        print(f"  Extraction mismatches:  {node_stats['extraction_mismatches']:>5}")
        print(f"  Citation ID mismatches: {node_stats['citation_id_mismatches']:>5}")

        if node_issues:
            hall = [i for i in node_issues if i["type"] == "hallucinated"]
            unseen = [i for i in node_issues if i["type"] == "known_but_unseen"]
            ext = [i for i in node_issues if i["type"] == "extraction_mismatch"]
            cid = [i for i in node_issues if i["type"] == "citation_id_mismatch"]

            if hall:
                print(f"\n  Hallucinated citations:")
                for h in hall:
                    print(f"    {h['paper']} cited ({h['citation'][0]}, {h['citation'][1]}) -- NOT in ground truth")
            if unseen:
                print(f"\n  Known-but-unseen citations:")
                for u in unseen:
                    print(f"    {u['paper']} cited ({u['citation'][0]}, {u['citation'][1]}) -> {u['resolved_id']} -- not in papers_seen_id")
            if ext:
                print(f"\n  Extraction mismatches:")
                for e in ext:
                    if e["only_in_body"]:
                        print(f"    {e['paper']} body has extra: {e['only_in_body']}")
                    if e["only_in_stored"]:
                        print(f"    {e['paper']} stored has extra: {e['only_in_stored']}")
            if cid:
                print(f"\n  Citation ID mismatches:")
                for c in cid:
                    if c["missing"]:
                        print(f"    {c['paper']} missing from stored: {c['missing']}")
                    if c["extra"]:
                        print(f"    {c['paper']} extra in stored: {c['extra']}")
        print()

        for k in overall:
            overall[k] += node_stats[k]
        all_issues.extend(node_issues)

    # Overall summary
    total = overall["total_citations"]
    print(f"{'='*60}")
    print(f"OVERALL ({total} total citations across {TOTAL_NODES} nodes)")
    print(f"{'='*60}")
    print(f"  Valid:                  {overall['valid']:>5}  ({overall['valid']/total*100:.1f}%)")
    print(f"  Known but not in pool:  {overall['known_but_unseen']:>5}  ({overall['known_but_unseen']/total*100:.1f}%)")
    print(f"  Hallucinated:           {overall['hallucinated']:>5}  ({overall['hallucinated']/total*100:.1f}%)")
    print(f"  Extraction mismatches:  {overall['extraction_mismatches']:>5} papers")
    print(f"  Citation ID mismatches: {overall['citation_id_mismatches']:>5} papers")

    # Write detailed JSONL
    out_path = f"{BASE}/output/citation_verification.jsonl"
    with open(out_path, "w") as f:
        f.write(json.dumps({"type": "summary", **overall}) + "\n")
        for issue in all_issues:
            f.write(json.dumps(issue) + "\n")
    print(f"\nDetailed issues written to {out_path}")

    # Write per-node summary CSV
    node_csv = f"{BASE}/output/citation_verification_by_node.csv"
    with open(node_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=node_rows[0].keys())
        writer.writeheader()
        writer.writerows(node_rows)
    print(f"Per-node CSV written to  {node_csv}")

    # Write per-paper detail CSV
    paper_csv = f"{BASE}/output/citation_verification_by_paper.csv"
    with open(paper_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=paper_rows[0].keys())
        writer.writeheader()
        writer.writerows(paper_rows)
    print(f"Per-paper CSV written to {paper_csv}")


if __name__ == "__main__":
    verify()
