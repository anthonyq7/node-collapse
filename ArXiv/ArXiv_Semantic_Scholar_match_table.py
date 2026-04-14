import csv
import json
import time
from pathlib import Path

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
    # Uncomment if you have an API key:
    "x-api-key": api_key
}

SLEEP_SECONDS = 1.1
MAX_RETRIES = 6
TIMEOUT = 30
FLUSH_EVERY = 20


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


def request_with_backoff(url: str, params=None, headers=None, timeout=30, max_retries=6):
    delay = 2.0
    last_error = None

    for _ in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)

            if r.status_code == 200:
                return r

            if r.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
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


def lookup_s2_by_arxiv_id(arxiv_id_no_version: str) -> dict:
    """
    Semantic Scholar lookup using:
        /paper/ARXIV:<id>
    """
    fields = [
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
    url = f"{API}/paper/ARXIV:{arxiv_id_no_version}"
    params = {"fields": ",".join(fields)}

    r = request_with_backoff(
        url=url,
        params=params,
        headers=HEADERS,
        timeout=TIMEOUT,
        max_retries=MAX_RETRIES,
    )
    time.sleep(SLEEP_SECONDS)
    return r.json()


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


# =========================
# Main bridge
# =========================
def bridge_arxiv_to_s2(
    arxiv_csv: str,
    out_csv: str,
    progress_json: str,
    flush_every: int = 20,
):
    rows = load_arxiv_rows(arxiv_csv)
    done_ids = load_progress(progress_json)
    write_header_if_needed(out_csv)

    processed_since_flush = 0
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        arxiv_id_raw = row.get("arxiv_id", "").strip()
        if not arxiv_id_raw:
            continue

        arxiv_id_no_version = strip_arxiv_version(arxiv_id_raw)

        if arxiv_id_raw in done_ids:
            continue

        arxiv_title = row.get("title", "")
        arxiv_published = row.get("published", "")
        arxiv_primary_category = row.get("primary_category", "")

        try:
            s2 = lookup_s2_by_arxiv_id(arxiv_id_no_version)

            append_result(out_csv, [
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
            ])

            print(f"[{i}/{total}] matched: {arxiv_id_raw} -> {s2.get('paperId', '')}")

        except Exception as e:
            append_result(out_csv, [
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
                str(e),
            ])
            print(f"[{i}/{total}] failed: {arxiv_id_raw} | {e}")

        done_ids.add(arxiv_id_raw)
        processed_since_flush += 1

        if processed_since_flush >= flush_every:
            save_progress(done_ids, progress_json)
            processed_since_flush = 0

    save_progress(done_ids, progress_json)
    print("Done.")


if __name__ == "__main__":
    bridge_arxiv_to_s2(
        arxiv_csv=ARXIV_CSV,
        out_csv=OUT_CSV,
        progress_json=PROGRESS_JSON,
        flush_every=FLUSH_EVERY,
    )