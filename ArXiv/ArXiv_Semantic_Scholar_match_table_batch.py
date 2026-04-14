import csv
import json
import time
from pathlib import Path
from typing import Iterable

import requests

# =========================
# Config
# =========================
ARXIV_CSV = "arxiv_kd_metadata.csv"
OUT_CSV = "arxiv_to_s2_mapping.csv"
PROGRESS_JSON = "arxiv_to_s2_progress.json"

api_key = open("api_key.txt", "r").readline().strip()

API = "https://api.semanticscholar.org/graph/v1"
HEADERS = {
    "x-api-key": api_key
}

# Batch size for /paper/batch
BATCH_SIZE = 100

# Sleep between batch requests, not per paper
SLEEP_SECONDS = 1.1

MAX_RETRIES = 6
TIMEOUT = 30
FLUSH_EVERY = 20

FIELDS = [
    "paperId",
    "title",
    "year",
    "venue",
    "authors",
    "externalIds",
    "referenceCount",
    "citationCount",
    "fieldsOfStudy",
    "s2FieldsOfStudy",
    "publicationDate",
]


# =========================
# Helpers
# =========================
def strip_arxiv_version(arxiv_id: str) -> str:
    """
    Examples:
        2603.13131v1 -> 2603.13131
        cs/0112017v2 -> cs/0112017
    """
    if not arxiv_id:
        return ""
    if "v" in arxiv_id:
        base, suffix = arxiv_id.rsplit("v", 1)
        if suffix.isdigit():
            return base
    return arxiv_id


def chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


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


def load_arxiv_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def normalize_authors(authors_field) -> str:
    if not authors_field:
        return ""
    if isinstance(authors_field, list):
        names = []
        for a in authors_field:
            if isinstance(a, dict):
                name = a.get("name")
                if name:
                    names.append(name)
        return "; ".join(names)
    return str(authors_field)


def safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def write_header_if_needed(out_csv: str) -> None:
    path = Path(out_csv)
    if path.exists():
        return

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "arxiv_id_raw",
            "arxiv_id_no_version",
            "arxiv_title",
            "arxiv_published",
            "arxiv_primary_category",
            "s2_found",
            "s2_paperId",
            "s2_title",
            "s2_year",
            "s2_venue",
            "s2_authors",
            "s2_referenceCount",
            "s2_citationCount",
            "s2_externalIds_json",
            "s2_fieldsOfStudy_json",
            "s2_s2FieldsOfStudy_json",
            "s2_publicationDate",
            "error",
        ])


def append_result(out_csv: str, row: list) -> None:
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def build_output_row(row: dict, s2: dict | None, error: str = "") -> list:
    arxiv_id_raw = row.get("arxiv_id", "").strip()
    arxiv_id_no_version = strip_arxiv_version(arxiv_id_raw)
    arxiv_title = row.get("title", "")
    arxiv_published = row.get("published", "")
    arxiv_primary_category = row.get("primary_category", "")

    if s2:
        return [
            arxiv_id_raw,
            arxiv_id_no_version,
            arxiv_title,
            arxiv_published,
            arxiv_primary_category,
            1,
            s2.get("paperId", ""),
            s2.get("title", ""),
            s2.get("year", ""),
            s2.get("venue", ""),
            normalize_authors(s2.get("authors")),
            s2.get("referenceCount", ""),
            s2.get("citationCount", ""),
            safe_json(s2.get("externalIds")),
            safe_json(s2.get("fieldsOfStudy")),
            safe_json(s2.get("s2FieldsOfStudy")),
            s2.get("publicationDate", ""),
            "",
        ]

    return [
        arxiv_id_raw,
        arxiv_id_no_version,
        arxiv_title,
        arxiv_published,
        arxiv_primary_category,
        0,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        error,
    ]


# =========================
# Batch lookup
# =========================
def lookup_s2_batch_by_arxiv_ids(arxiv_ids_no_version: list[str]) -> dict[str, dict]:
    """
    Uses Semantic Scholar batch paper details endpoint.

    Input IDs are sent as:
        ARXIV:<id>

    Returns:
        dict mapping arxiv_id_no_version -> s2 paper dict
    """
    if not arxiv_ids_no_version:
        return {}

    url = f"{API}/paper/batch"
    params = {"fields": ",".join(FIELDS)}

    ids = [f"ARXIV:{aid}" for aid in arxiv_ids_no_version]
    payload = {"ids": ids}

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

    # Expecting list aligned with request order.
    # We map it back by input position.
    result = {}
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected batch response shape: {type(data)} | {str(data)[:300]}")

    if len(data) != len(arxiv_ids_no_version):
        raise RuntimeError(
            f"Batch response length mismatch: got {len(data)}, expected {len(arxiv_ids_no_version)}"
        )

    for arxiv_id_no_version, item in zip(arxiv_ids_no_version, data):
        if isinstance(item, dict) and item.get("paperId"):
            result[arxiv_id_no_version] = item

    time.sleep(SLEEP_SECONDS)
    return result


# =========================
# Main bridge
# =========================
def bridge_arxiv_to_s2_batched(
    arxiv_csv: str,
    out_csv: str,
    progress_json: str,
    batch_size: int = 100,
    flush_every: int = 20,
):
    rows = load_arxiv_rows(arxiv_csv)
    done_ids = load_progress(progress_json)
    write_header_if_needed(out_csv)

    pending_rows = []
    for row in rows:
        arxiv_id_raw = row.get("arxiv_id", "").strip()
        if not arxiv_id_raw:
            continue
        if arxiv_id_raw in done_ids:
            continue
        pending_rows.append(row)

    total_pending = len(pending_rows)
    print(f"Pending rows: {total_pending}")

    processed_since_flush = 0
    processed_total = 0

    for batch_num, batch_rows in enumerate(chunked(pending_rows, batch_size), start=1):
        batch_ids = [
            strip_arxiv_version(row.get("arxiv_id", "").strip())
            for row in batch_rows
        ]

        try:
            matches = lookup_s2_batch_by_arxiv_ids(batch_ids)

            for row in batch_rows:
                arxiv_id_raw = row.get("arxiv_id", "").strip()
                arxiv_id_no_version = strip_arxiv_version(arxiv_id_raw)
                s2 = matches.get(arxiv_id_no_version)

                if s2:
                    append_result(out_csv, build_output_row(row, s2=s2))
                    print(
                        f"[batch {batch_num}] matched: "
                        f"{arxiv_id_raw} -> {s2.get('paperId', '')}"
                    )
                else:
                    append_result(
                        out_csv,
                        build_output_row(
                            row,
                            s2=None,
                            error="No match returned from /paper/batch",
                        ),
                    )
                    print(f"[batch {batch_num}] no match: {arxiv_id_raw}")

                done_ids.add(arxiv_id_raw)
                processed_since_flush += 1
                processed_total += 1

                if processed_since_flush >= flush_every:
                    save_progress(done_ids, progress_json)
                    processed_since_flush = 0

        except Exception as e:
            batch_error = f"Batch request failed: {e}"

            # If one whole batch fails, write failures row-by-row so resume still works cleanly.
            for row in batch_rows:
                arxiv_id_raw = row.get("arxiv_id", "").strip()
                append_result(out_csv, build_output_row(row, s2=None, error=batch_error))
                print(f"[batch {batch_num}] failed: {arxiv_id_raw} | {batch_error}")
                done_ids.add(arxiv_id_raw)
                processed_since_flush += 1
                processed_total += 1

                if processed_since_flush >= flush_every:
                    save_progress(done_ids, progress_json)
                    processed_since_flush = 0

        print(f"Processed {processed_total}/{total_pending} pending rows")

    save_progress(done_ids, progress_json)
    print("Done.")


if __name__ == "__main__":
    bridge_arxiv_to_s2_batched(
        arxiv_csv=ARXIV_CSV,
        out_csv=OUT_CSV,
        progress_json=PROGRESS_JSON,
        batch_size=BATCH_SIZE,
        flush_every=FLUSH_EVERY,
    )