import csv
import json
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

# =========================
# Config
# =========================
PAPER_CSV = "arxiv_to_s2_mapping.csv"

RAW_REFERENCE_CSV = "raw_reference.csv"
NEW_PAPERS_CSV = "new_papers_from_references.csv"
CITATION_CSV = "citation.csv"

PROGRESS_JSON = "reference_pull_progress.json"

api_key = open("api_key.txt", "r").readline().strip()

API = "https://api.semanticscholar.org/graph/v1"
HEADERS = {
    "x-api-key": api_key
}

BATCH_SIZE = 100
SLEEP_SECONDS = 1.1
MAX_RETRIES = 6
TIMEOUT = 30
FLUSH_EVERY = 10

# Keep fields lean.
FIELDS = ",".join([
    "paperId",
    "title",
    "year",
    "venue",
    "externalIds",
    "publicationDate",
    "references.paperId",
    "references.title",
    "references.year",
    "references.venue",
    "references.externalIds",
    "referenceCount",
])

# =========================
# Helpers
# =========================
def chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def request_with_backoff(
    method: str,
    url: str,
    *,
    params=None,
    headers=None,
    json_body=None,
    timeout=30,
    max_retries=6,
):
    delay = 2.0
    last_error = None

    for _ in range(max_retries):
        try:
            r = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json=json_body,
                timeout=timeout,
            )

            if r.status_code == 200:
                return r

            if r.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(delay)
                delay *= 2
                continue

            r.raise_for_status()

        except requests.RequestException as e:
            last_error = str(e)
            time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Request failed after retries: {last_error}")


def load_progress(progress_file: str) -> set[str]:
    path = Path(progress_file)
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_progress(done_ids: set[str], progress_file: str) -> None:
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(sorted(done_ids), f, ensure_ascii=False, indent=2)


def write_csv_header_if_needed(path: str, header: list[str]) -> None:
    if Path(path).exists():
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv_rows(path: str, rows: list[list]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def fetch_paper_batch(paper_ids: list[str]) -> list[dict | None]:
    """
    Input: Semantic Scholar paperIds
    Output: list aligned to input order
    """
    if not paper_ids:
        return []

    url = f"{API}/paper/batch"
    params = {"fields": FIELDS}
    payload = {"ids": paper_ids}

    r = request_with_backoff(
        method="POST",
        url=url,
        params=params,
        headers=HEADERS,
        json_body=payload,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
    )
    data = r.json()

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response shape: {type(data)} | {str(data)[:300]}")
    if len(data) != len(paper_ids):
        raise RuntimeError(f"Length mismatch: got {len(data)}, expected {len(paper_ids)}")

    time.sleep(SLEEP_SECONDS)
    return data


# =========================
# Main
# =========================
def pull_references_from_seed_papers():
    paper_df = pd.read_csv(PAPER_CSV)

    # Keep only matched seed papers that have s2 ids
    seed_df = paper_df[
        (paper_df["s2_found"] == 1) &
        (paper_df["s2_paperId"].notna()) &
        (paper_df["s2_paperId"].astype(str).str.strip() != "")
    ].copy()

    # Create internal paper_id if you do not yet have one
    if "paper_id" not in seed_df.columns:
        seed_df = seed_df.reset_index(drop=True)
        seed_df["paper_id"] = seed_df.index + 1

    # Map S2 id -> internal paper_id for seed papers
    seed_s2_to_internal = dict(zip(seed_df["s2_paperId"].astype(str), seed_df["paper_id"]))

    done_seed_s2_ids = load_progress(PROGRESS_JSON)

    pending_seed_s2_ids = [
        str(x) for x in seed_df["s2_paperId"].astype(str).tolist()
        if str(x) not in done_seed_s2_ids
    ]

    print(f"Seed papers total: {len(seed_df)}")
    print(f"Pending seed papers: {len(pending_seed_s2_ids)}")

    write_csv_header_if_needed(RAW_REFERENCE_CSV, [
        "raw_ref_id",
        "source_paper_id",
        "source_s2_paperId",
        "matched_paper_s2_id",
        "raw_string",
        "extracted_title",
        "extracted_doi",
        "extracted_arxiv_id",
        "ref_year",
        "ref_venue",
        "match_confidence",
        "match_method",
    ])

    write_csv_header_if_needed(NEW_PAPERS_CSV, [
        "s2_paperId",
        "title",
        "year",
        "venue",
        "doi",
        "arxiv_id",
        "publicationDate",
        "externalIds_json",
    ])

    write_csv_header_if_needed(CITATION_CSV, [
        "source_paper_id",
        "source_s2_paperId",
        "target_s2_paperId",
    ])

    # Track target papers seen so they are not duplicated in NEW_PAPERS_CSV
    seen_target_s2_ids = set()
    if Path(NEW_PAPERS_CSV).exists() and Path(NEW_PAPERS_CSV).stat().st_size > 0:
        try:
            existing_new = pd.read_csv(NEW_PAPERS_CSV)
            if "s2_paperId" in existing_new.columns:
                seen_target_s2_ids.update(existing_new["s2_paperId"].dropna().astype(str).tolist())
        except Exception:
            pass

    # Also avoid re-outputting seed papers as “new”
    seen_target_s2_ids.update(seed_df["s2_paperId"].astype(str).tolist())

    raw_ref_id = 1
    if Path(RAW_REFERENCE_CSV).exists() and Path(RAW_REFERENCE_CSV).stat().st_size > 0:
        try:
            existing_raw = pd.read_csv(RAW_REFERENCE_CSV)
            if len(existing_raw) > 0 and "raw_ref_id" in existing_raw.columns:
                raw_ref_id = int(existing_raw["raw_ref_id"].max()) + 1
        except Exception:
            pass

    processed_since_flush = 0
    processed_total = 0
    missing_ref_count = 0

    for batch_num, batch_seed_s2_ids in enumerate(chunked(pending_seed_s2_ids, BATCH_SIZE), start=1):
        try:
            batch_results = fetch_paper_batch(batch_seed_s2_ids)

            raw_reference_rows = []
            new_paper_rows = []
            citation_rows = []

            for seed_s2_id, item in zip(batch_seed_s2_ids, batch_results):
                source_paper_id = seed_s2_to_internal.get(seed_s2_id)

                if not item:
                    print(f"Missing item: {seed_s2_id}")
                    processed_total += 1
                    processed_since_flush += 1
                    continue

                refs = item.get("references", []) or []

                for ref in refs:
                    ext = ref.get("externalIds") or {}
                    target_s2_id = ref.get("paperId")

                    raw_reference_rows.append([
                        raw_ref_id,
                        source_paper_id,
                        seed_s2_id,
                        target_s2_id if target_s2_id else "",
                        "",  # raw_string not provided by this endpoint
                        ref.get("title", ""),
                        ext.get("DOI", ""),
                        ext.get("ArXiv", ""),
                        ref.get("year", ""),
                        ref.get("venue", ""),
                        1.0 if target_s2_id else "",
                        "semantic_scholar_references",
                    ])
                    raw_ref_id += 1

                    if target_s2_id:
                        citation_rows.append([
                            source_paper_id,
                            seed_s2_id,
                            target_s2_id,
                        ])

                        if str(target_s2_id) not in seen_target_s2_ids:
                            new_paper_rows.append([
                                target_s2_id,
                                ref.get("title", ""),
                                ref.get("year", ""),
                                ref.get("venue", ""),
                                ext.get("DOI", ""),
                                ext.get("ArXiv", ""),
                                "",  # publicationDate not included inside reference subobject
                                safe_json(ext),
                            ])
                            seen_target_s2_ids.add(str(target_s2_id))

                done_seed_s2_ids.add(seed_s2_id)
                processed_total += 1
                processed_since_flush += 1

                expected = item.get("referenceCount", None)
                observed = len(refs)

                if expected is not None:
                    missing_ref_count += max(expected - observed, 0)
                    
                    print(f"TRUNCATED: {seed_s2_id} | {observed}/{expected}")

            append_csv_rows(RAW_REFERENCE_CSV, raw_reference_rows)
            append_csv_rows(NEW_PAPERS_CSV, new_paper_rows)
            append_csv_rows(CITATION_CSV, citation_rows)

            if processed_since_flush >= FLUSH_EVERY:
                save_progress(done_seed_s2_ids, PROGRESS_JSON)
                processed_since_flush = 0

            print(
                f"[batch {batch_num}] done | "
                f"seed processed: {processed_total}/{len(pending_seed_s2_ids)} | "
                f"missing_ref_count: {missing_ref_count} | "
                f"raw refs written: {len(raw_reference_rows)} | "
                f"new papers: {len(new_paper_rows)} | "
                f"citations: {len(citation_rows)}"
            )

        except Exception as e:
            print(f"[batch {batch_num}] FAILED: {e}")
            # leave this batch unmarked so rerun can retry

    save_progress(done_seed_s2_ids, PROGRESS_JSON)
    print("Done.")


if __name__ == "__main__":
    pull_references_from_seed_papers()