import requests
import feedparser
import csv
import time

BASE_URL = "http://export.arxiv.org/api/query"

def query_arxiv(search_query, start=0, max_results=100, sort_by="submittedDate", sort_order="descending"):
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return feedparser.parse(response.text)

def extract_entry(entry):
    arxiv_id = entry.id.split("/abs/")[-1]
    pdf_url = None
    for link in entry.links:
        if getattr(link, "title", "") == "pdf":
            pdf_url = link.href
            break

    categories = [tag["term"] for tag in entry.tags] if "tags" in entry else []

    return {
        "arxiv_id": arxiv_id,
        "title": entry.title.strip().replace("\n", " "),
        "abstract": entry.summary.strip().replace("\n", " "),
        "published": entry.published,
        "updated": entry.updated,
        "authors": "; ".join(author.name for author in entry.authors),
        "primary_category": getattr(entry, "arxiv_primary_category", {}).get("term", None),
        "categories": "; ".join(categories),
        "abs_url": entry.id,
        "pdf_url": pdf_url,
    }

def crawl_arxiv_metadata(search_query, total_results=500, batch_size=100, sleep_sec=3.0):
    rows = []
    for start in range(0, total_results, batch_size):
        feed = query_arxiv(search_query, start=start, max_results=batch_size)
        if not feed.entries:
            break
        rows.extend(extract_entry(entry) for entry in feed.entries)
        time.sleep(sleep_sec)  # be polite
    return rows

def save_csv(rows, filename):
    if not rows:
        return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    q = '(ti:"knowledge distillation" OR abs:"knowledge distillation") AND (cat:cs.LG OR cat:cs.CV OR cat:cs.AI OR cat:cs.CL)'
    rows = crawl_arxiv_metadata(q, total_results=10000, batch_size=100)
    save_csv(rows, "arxiv_kd_metadata_10000.csv")
    print(f"Saved {len(rows)} rows.")