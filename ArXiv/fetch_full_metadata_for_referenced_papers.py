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
INPUT_CSV = "new_papers_from_references.csv"
OUTPUT_CSV = "referenced_papers_full.csv"
PROGRESS_JSON = "referenced_papers_progress.json"

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

FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "year",
    "venue",
    "publicationDate",
    "publicationTypes",
    "journal",
    "authors",
    "externalIds",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    "openAccessPdf",
    "fieldsOfStudy",
    "s2FieldsOfStudy",
    "url",
])

# =========================
# Helpers
# =========================
def chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def normalize_authors(authors_field) -> str:
    if not authors_field:
        return ""
    if isinstance(authors_field, list):
        parts = []
        for a in authors_field:
            if isinstance(a, dict):
                name = a.get("name", "")
                author_id = a.get("authorId", "")
                if name and author_id:
                    parts.append({"name": name, "authorId": author_id})
                elif name:
                    parts.append({"name": name})
        return safe_json(parts)
    return safe_json(authors_field)


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


def build_output_row(item: dict | None, requested_s2_id: str) -> list:
    if not item:
        return [
            requested_s2_id,
            0,  # found
            "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "No record returned"
        ]

    ext = item.get("externalIds") or {}
    journal = item.get("journal") or {}
    open_access_pdf = item.get("openAccessPdf") or {}

    return [
        requested_s2_id,
        1,  # found
        item.get("paperId", ""),
        item.get("title", ""),
        item.get("abstract", ""),
        item.get("year", ""),
        item.get("venue", ""),
        item.get("publicationDate", ""),
        safe_json(item.get("publicationTypes")),
        journal.get("name", ""),
        journal.get("pages", ""),
        journal.get("volume", ""),
        safe_json(item.get("authors")),
        ext.get("DOI", ""),
        ext.get("ArXiv", ""),
        safe_json(ext),
        item.get("referenceCount", ""),
        item.get("citationCount", ""),
        item.get("influentialCitationCount", ""),
        item.get("isOpenAccess", ""),
        open_access_pdf.get("url", ""),
        safe_json(item.get("fieldsOfStudy")),
        safe_json(item.get("s2FieldsOfStudy")),
        item.get("url", ""),
        "",
    ]


# =========================
# Main
# =========================
def fetch_full_metadata_for_referenced_papers():
    df = pd.read_csv(INPUT_CSV)

    if "s2_paperId" not in df.columns:
        raise ValueError("INPUT_CSV must contain column 's2_paperId'")

    target_ids = (
        df["s2_paperId"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    target_ids = [x for x in target_ids.tolist() if x]

    # De-duplicate while preserving order
    seen = set()
    unique_target_ids = []
    for x in target_ids:
        if x not in seen:
            seen.add(x)
            unique_target_ids.append(x)

    done_ids = load_progress(PROGRESS_JSON)
    pending_ids = [x for x in unique_target_ids if x not in done_ids]

    print(f"Referenced papers total: {len(unique_target_ids)}")
    print(f"Pending referenced papers: {len(pending_ids)}")

    write_csv_header_if_needed(OUTPUT_CSV, [
        "requested_s2_paperId",
        "found",
        "s2_paperId",
        "title",
        "abstract",
        "year",
        "venue",
        "publicationDate",
        "publicationTypes_json",
        "journal_name",
        "journal_pages",
        "journal_volume",
        "authors_json",
        "doi",
        "arxiv_id",
        "externalIds_json",
        "referenceCount",
        "citationCount",
        "influentialCitationCount",
        "isOpenAccess",
        "openAccessPdf_url",
        "fieldsOfStudy_json",
        "s2FieldsOfStudy_json",
        "s2_url",
        "error",
    ])

    processed_since_flush = 0
    processed_total = 0

    for batch_num, batch_ids in enumerate(chunked(pending_ids, BATCH_SIZE), start=1):
        try:
            batch_results = fetch_paper_batch(batch_ids)
            out_rows = []

            for requested_s2_id, item in zip(batch_ids, batch_results):
                out_rows.append(build_output_row(item, requested_s2_id))
                done_ids.add(requested_s2_id)
                processed_total += 1
                processed_since_flush += 1

            append_csv_rows(OUTPUT_CSV, out_rows)

            if processed_since_flush >= FLUSH_EVERY:
                save_progress(done_ids, PROGRESS_JSON)
                processed_since_flush = 0

            print(
                f"[batch {batch_num}] done | "
                f"processed: {processed_total}/{len(pending_ids)} | "
                f"rows written: {len(out_rows)}"
            )

        except Exception as e:
            print(f"[batch {batch_num}] FAILED: {e}")
            # leave batch unmarked so rerun retries it

    save_progress(done_ids, PROGRESS_JSON)
    print("Done.")


if __name__ == "__main__":
    fetch_full_metadata_for_referenced_papers()