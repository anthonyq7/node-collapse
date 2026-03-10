# Bulk downloads with audit. 
# Please change the "User-Agent" to your contact when you are crawling. 

import os
import re
import csv
import time
import random
import urllib.request
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError

import xml.etree.ElementTree as ET
import urllib.parse

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# ---------- arXiv API paging ----------

def fetch_atom(search_query: str, start: int, max_results: int = 100) -> str:
    base = "http://export.arxiv.org/api/query?"
    params = {"search_query": search_query, "start": start, "max_results": max_results}
    url = base + urllib.parse.urlencode(params)

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "bulk-arxiv-downloader/1.0 (contact: bolong.tang@utexas.edu)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")

# Directly append pdf url or turn abstract url to pdf url
def extract_pdf_urls(atom_xml: str) -> list[str]:
    root = ET.fromstring(atom_xml)
    pdf_urls = []
    for entry in root.findall("atom:entry", ATOM_NS):
        link = entry.find("atom:link[@type='application/pdf']", ATOM_NS)
        if link is not None and "href" in link.attrib:
            pdf_urls.append(link.attrib["href"])
        else:
            abs_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
            if "/abs/" in abs_id:
                pdf_urls.append(abs_id.replace("/abs/", "/pdf/"))
    return pdf_urls

# A generator for pdf urls
def iter_pdf_urls(search_query: str, total: int, page_size: int = 100, delay_s: float = 3.0):
    got = 0
    start = 0
    while got < total:
        batch = min(page_size, total - got)
        atom_xml = fetch_atom(search_query, start=start, max_results=batch)
        urls = extract_pdf_urls(atom_xml)

        for u in urls:
            yield u

        got += len(urls)
        start += batch
        time.sleep(delay_s)

# ---------- audit log helpers ----------

AUDIT_FIELDS = ["timestamp_utc", "url", "status", "filepath", "bytes", "error"]

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_completed_urls(audit_csv_path: str) -> set[str]:
    """Treat downloaded + skipped_exists as completed (so reruns skip them)."""
    completed = set()
    if not os.path.exists(audit_csv_path):
        return completed

    with open(audit_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") in {"downloaded", "skipped_exists"}:
                u = row.get("url")
                if u:
                    completed.add(u)
    return completed

def append_audit_row(audit_csv_path: str, row: dict) -> None:
    file_exists = os.path.exists(audit_csv_path)
    with open(audit_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in AUDIT_FIELDS})

# ---------- download ----------

def safe_filename_from_url(pdf_url: str) -> str:
    tail = pdf_url.rstrip("/").split("/")[-1]
    tail = re.sub(r"[^A-Za-z0-9._-]+", "_", tail)
    return tail + ".pdf"

def download_pdf_with_audit(pdf_url: str, out_dir: str, audit_csv_path: str,
                            max_tries: int = 6, base_delay: float = 3.0) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, safe_filename_from_url(pdf_url))

    # Resume: if file exists and looks non-trivial, record skip and return
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
        append_audit_row(audit_csv_path, {
            "timestamp_utc": utc_now_iso(),
            "url": pdf_url,
            "status": "skipped_exists",
            "filepath": out_path,
            "bytes": os.path.getsize(out_path),
            "error": "",
        })
        return

    headers = {"User-Agent": "bulk-arxiv-downloader/1.0 (contact: your_email@domain.com)"}

    for attempt in range(1, max_tries + 1):
        try:
            req = urllib.request.Request(pdf_url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read()

            with open(out_path, "wb") as f:
                f.write(content)

            append_audit_row(audit_csv_path, {
                "timestamp_utc": utc_now_iso(),
                "url": pdf_url,
                "status": "downloaded",
                "filepath": out_path,
                "bytes": len(content),
                "error": "",
            })
            return

        except (HTTPError, URLError, TimeoutError) as e:
            err = f"{type(e).__name__}: {e}"
            if attempt == max_tries:
                append_audit_row(audit_csv_path, {
                    "timestamp_utc": utc_now_iso(),
                    "url": pdf_url,
                    "status": "failed",
                    "filepath": out_path,
                    "bytes": 0,
                    "error": err,
                })
                return

            sleep_s = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1.0)
            print(f"[{attempt}/{max_tries}] failed {pdf_url} -> {err}; sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)

# ---------- main bulk loop ----------

def bulk_download_with_audit(search_query: str, total: int, out_dir: str, audit_csv_path: str):
    completed = load_completed_urls(audit_csv_path)
    print(f"Loaded {len(completed)} completed URLs from audit log.")

    n_done = 0
    for pdf_url in iter_pdf_urls(search_query, total=total, page_size=100, delay_s=3.0):
        if pdf_url in completed:
            continue

        download_pdf_with_audit(pdf_url, out_dir=out_dir, audit_csv_path=audit_csv_path)
        completed.add(pdf_url)  # prevent duplicate work in same run
        n_done += 1

        # extra politeness delay between PDFs
        time.sleep(1.0)

    print(f"Finished. New processed URLs this run: {n_done}")

# Example:
bulk_download_with_audit(
    search_query="all:knowledge distillation",
    total=50,
    out_dir="electron_pdfs",
    audit_csv_path="arxiv_download_audit.csv",
)