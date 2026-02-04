"""
arXiv Paper Collector for Attention Mechanisms in NLP

This script searches arXiv for papers about attention mechanisms, verifies them
using GPT-4o, fetches citation counts from Semantic Scholar, and organizes
results into stratified buckets based on citation count.

Workflow:
    1. Search arXiv API for papers
    2. Verify each paper with LLM (is it really about attention mechanisms?)
    3. Get citation count from Semantic Scholar
    4. Place paper in appropriate bucket based on citations
    5. Save results to JSON
"""

import json
import os
import ssl
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from openai import OpenAI

# --- SSL Configuration ---
# macOS sometimes has issues with SSL certificate verification.
# We create an SSL context that handles this gracefully.
# For production code, you'd want proper certificate handling.
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Load environment variables from .env file (contains OPENAI_API_KEY)
load_dotenv()

# --- arXiv API Configuration ---
# The arXiv API uses Atom XML format and supports boolean queries
# Documentation: https://info.arxiv.org/help/api/basics.html
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# --- Semantic Scholar API Configuration ---
# Semantic Scholar provides citation data for academic papers
# We can query by arXiv ID using the format: arXiv:{id}
# Documentation: https://api.semanticscholar.org/api-docs/
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper"

# Rate limiting: Semantic Scholar has strict rate limits (~100 requests/5 min for unauthenticated)
# We add a delay between requests to avoid 429 (Too Many Requests) errors
SEMANTIC_SCHOLAR_DELAY_SECONDS = 5.0  # Wait 5 seconds between requests

# Retry configuration for failed API requests
SEMANTIC_SCHOLAR_MAX_RETRIES = 3  # Number of retry attempts before giving up
SEMANTIC_SCHOLAR_RETRY_DELAY = 10.0  # Initial delay between retries (doubles each attempt)

# --- arXiv Pagination Configuration ---
# arXiv returns results in batches. We fetch multiple batches until buckets are full.
ARXIV_BATCH_SIZE = 500  # Papers per batch (arXiv max is typically 2000)
ARXIV_MAX_BATCHES = 20  # Safety limit to prevent infinite loops

# --- Bucket Configuration ---
# Papers are categorized into buckets based on citation count
# Each bucket has a maximum capacity to ensure diversity in the dataset
BUCKET_CONFIG = {
    "bucket_20_100": {
        "min_citations": 20,
        "max_citations": 100,
        "capacity": 50,
    }
}

# Output file for collected papers
OUTPUT_FILE = "arxiv_papers.json"


# =============================================================================
# ARXIV API FUNCTIONS
# =============================================================================

def search_arxiv(query, max_results=500, start=0):
    """
    Query the arXiv API and return the raw XML response.
    
    The arXiv API accepts search queries in a specific format:
    - search_query: The search terms (supports boolean operators)
    - max_results: Maximum number of results to return per batch
    - start: Offset for pagination (0 for first batch, 500 for second, etc.)
    
    Args:
        query: Search string (e.g., "attention mechanisms NLP")
        max_results: Maximum number of papers to fetch in this batch
        start: Offset for pagination (use for fetching additional batches)
    
    Returns:
        str: Raw XML response from arXiv API
    
    Example API URL:
        http://export.arxiv.org/api/query?search_query=all:attention+AND+all:NLP&max_results=500&start=0
    """
    # Build the search query - 'all:' searches across all fields (title, abstract, etc.)
    # We use AND to require all terms to be present
    search_terms = query.split()
    formatted_query = "+AND+".join(f"all:{term}" for term in search_terms)
    
    # Note: We manually build the URL because arXiv expects specific formatting
    # The 'start' parameter enables pagination through large result sets
    url = f"{ARXIV_API_URL}?search_query={formatted_query}&max_results={max_results}&start={start}"
    
    print(f"Searching arXiv (start={start}): {url}")
    
    # Make the HTTP request (with SSL context to handle certificate issues)
    with urllib.request.urlopen(url, context=SSL_CONTEXT) as response:
        xml_content = response.read().decode('utf-8')
    
    return xml_content


def parse_arxiv_xml(xml_content):
    """
    Parse arXiv API XML response to extract paper metadata.
    
    The arXiv API returns data in Atom XML format. Each paper is an <entry>
    element containing:
    - <title>: Paper title
    - <summary>: Abstract
    - <id>: Full arXiv URL (we extract the ID from this)
    - <link>: Links including PDF URL
    
    Important: arXiv uses XML namespaces, so we must handle them when parsing.
    The default namespace is Atom: http://www.w3.org/2005/Atom
    
    Args:
        xml_content: Raw XML string from arXiv API
    
    Returns:
        list: List of dicts with keys: title, abstract, arxiv_id, pdf_url
    """
    # Define XML namespaces used by arXiv
    # The Atom namespace is the default for feed structure
    namespaces = {
        'atom': 'http://www.w3.org/2005/Atom',
    }
    
    # Parse the XML content
    root = ET.fromstring(xml_content)
    
    papers = []
    
    # Find all <entry> elements (each represents one paper)
    for entry in root.findall('atom:entry', namespaces):
        # Extract title - remove extra whitespace and newlines
        title_elem = entry.find('atom:title', namespaces)
        title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""
        
        # Extract abstract (called 'summary' in Atom format)
        summary_elem = entry.find('atom:summary', namespaces)
        abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""
        
        # Extract arXiv ID from the <id> element
        # Format: http://arxiv.org/abs/1234.56789v1 -> we want "1234.56789"
        id_elem = entry.find('atom:id', namespaces)
        arxiv_url = id_elem.text if id_elem is not None else ""
        # Extract just the ID part (remove version number like 'v1')
        arxiv_id = arxiv_url.split('/abs/')[-1].split('v')[0] if arxiv_url else ""
        
        # Find the PDF link - look for link with title="pdf"
        pdf_url = ""
        for link in entry.findall('atom:link', namespaces):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href', '')
                break
        
        # Extract authors - each author has a <name> child element
        authors = []
        for author in entry.findall('atom:author', namespaces):
            name_elem = author.find('atom:name', namespaces)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
        
        # Extract publish date (ISO 8601 format: 2017-06-12T17:57:34Z)
        published_elem = entry.find('atom:published', namespaces)
        published_date = published_elem.text.strip() if published_elem is not None else ""
        
        # Only add papers with valid data
        if title and abstract and arxiv_id:
            papers.append({
                'title': title,
                'abstract': abstract,
                'arxiv_id': arxiv_id,
                'pdf_url': pdf_url,
                'authors': authors,
                'published_date': published_date,
            })
    
    print(f"Parsed {len(papers)} papers from arXiv response")
    return papers


# =============================================================================
# LLM VERIFICATION FUNCTION
# =============================================================================

def verify_with_llm(title, abstract, client):
    """
    Use GPT-4o to verify if a paper is actually about attention mechanisms.
    
    Sometimes search results include papers that mention "attention" in a
    different context. We use an LLM to verify relevance.
    
    The OpenAI API call structure:
    - model: The model to use (gpt-4o)
    - messages: List of message objects with 'role' and 'content'
    - max_tokens: Limit response length (we only need yes/no)
    
    Args:
        title: Paper title
        abstract: Paper abstract
        client: OpenAI client instance
    
    Returns:
        bool: True if paper is about attention mechanisms, False otherwise
    """
    # Construct a clear prompt asking for yes/no verification
    prompt = f"""Determine if this academic paper is specifically about attention mechanisms in NLP/deep learning.

Title: {title}

Abstract: {abstract}

Answer with only 'yes' or 'no'. The paper should be about attention mechanisms as a core topic (like self-attention, transformer attention, etc.), not just mentioning attention casually."""

    # Make the API call to GPT-4o
    # The messages list follows the chat format: system, user, assistant
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that classifies academic papers. Respond only with 'yes' or 'no'."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        max_tokens=10,  # We only need a short response
        temperature=0,  # Use deterministic output for consistency
    )
    
    # Extract the response text and check for 'yes'
    answer = response.choices[0].message.content.strip().lower()
    
    return 'yes' in answer


# =============================================================================
# SEMANTIC SCHOLAR API FUNCTION
# =============================================================================

def get_citation_count(arxiv_id):
    """
    Fetch citation count for a paper from Semantic Scholar API.
    
    Semantic Scholar allows querying papers by various IDs including arXiv IDs.
    The API endpoint format is:
        /paper/arXiv:{arxiv_id}?fields=citationCount
    
    The 'fields' parameter specifies which data to return (we only need citations).
    
    Response format:
        {"paperId": "...", "citationCount": 123}
    
    Note: This function includes a delay to respect Semantic Scholar's rate limits.
    Without throttling, you'll quickly hit 429 (Too Many Requests) errors.
    
    If a request fails, the function will retry with exponential backoff.
    
    Args:
        arxiv_id: The arXiv ID (e.g., "1706.03762")
    
    Returns:
        int: Number of citations, or 0 if not found
    
    Raises:
        Exception: If all retry attempts fail
    """
    # Build the API URL with arXiv ID prefix
    # The format arXiv:{id} tells Semantic Scholar to look up by arXiv ID
    url = f"{SEMANTIC_SCHOLAR_API_URL}/arXiv:{arxiv_id}?fields=citationCount"
    
    # Create request with a user agent (some APIs require this)
    request = urllib.request.Request(
        url,
        headers={'User-Agent': 'ArxivPaperCollector/1.0'}
    )
    
    last_exception = None
    
    for attempt in range(SEMANTIC_SCHOLAR_MAX_RETRIES + 1):
        # Throttle requests to avoid 429 rate limit errors
        # Semantic Scholar allows ~100 requests per 5 minutes for unauthenticated users
        time.sleep(SEMANTIC_SCHOLAR_DELAY_SECONDS)
        
        try:
            # Make the request and parse JSON response (with SSL context)
            with urllib.request.urlopen(request, context=SSL_CONTEXT) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            # Extract citation count (default to 0 if not present)
            citation_count = data.get('citationCount', 0)
            
            # Handle None values (paper exists but no citation data)
            if citation_count is None:
                citation_count = 0
            
            return citation_count
            
        except Exception as e:
            last_exception = e
            
            if attempt < SEMANTIC_SCHOLAR_MAX_RETRIES:
                # Calculate exponential backoff delay
                retry_delay = SEMANTIC_SCHOLAR_RETRY_DELAY * (2 ** attempt)
                print(f"  Citation request failed (attempt {attempt + 1}/{SEMANTIC_SCHOLAR_MAX_RETRIES + 1}): {e}")
                print(f"  Retrying in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
    
    # All retries exhausted, raise the last exception
    raise last_exception


# =============================================================================
# BUCKETING FUNCTIONS
# =============================================================================

def get_bucket_name(citation_count):
    """
    Determine which bucket a paper belongs to based on its citation count.
    
    Buckets are defined by citation ranges:
    - bucket_0_9: 0-9 citations (emerging/new papers)
    - bucket_10_999: 10-999 citations (established papers)
    - bucket_1000_9999: 1000-9999 citations (influential papers)
    - bucket_10000_plus: 10000+ citations (landmark papers)
    
    Args:
        citation_count: Number of citations
    
    Returns:
        str: Bucket name (e.g., "bucket_0_9")
    """
    for bucket_name, config in BUCKET_CONFIG.items():
        if config['min_citations'] <= citation_count <= config['max_citations']:
            return bucket_name
    
    # Fallback (should never happen with our config)
    return None


def add_to_bucket(paper, citation_count, buckets):
    """
    Add a paper to the appropriate bucket if there's space.
    
    Each bucket has a maximum capacity to ensure we collect a diverse set
    of papers across different citation ranges.
    
    Args:
        paper: Dict with paper metadata (title, abstract, arxiv_id, pdf_url, authors, published_date)
        citation_count: Number of citations for the paper
        buckets: Dict of bucket_name -> list of papers
    
    Returns:
        bool: True if paper was added, False if bucket was full
    """
    bucket_name = get_bucket_name(citation_count)
    bucket_capacity = BUCKET_CONFIG[bucket_name]['capacity']
    
    # Check if bucket has space
    if len(buckets[bucket_name]) >= bucket_capacity:
        print(f"  Bucket '{bucket_name}' is full, discarding paper")
        return False
    
    # Add paper with citation count to the bucket
    paper_with_citations = {
        'title': paper['title'],
        'abstract': paper['abstract'],
        'arxiv_id': paper['arxiv_id'],
        'citation_count': citation_count,
        'url': paper['pdf_url'],
        'authors': paper['authors'],
        'published_date': paper['published_date'],
    }
    
    buckets[bucket_name].append(paper_with_citations)
    print(f"  Added to '{bucket_name}' ({len(buckets[bucket_name])}/{bucket_capacity})")
    print(f"Title: {paper['title']}")
    print(f"Authors: {paper['authors']}")
    print(f"Date: {paper['published_date']}")
    
    return True


def buckets_are_full(buckets):
    """
    Check if all buckets have reached their capacity.
    
    This is used to stop processing early if we've collected enough papers.
    
    Args:
        buckets: Dict of bucket_name -> list of papers
    
    Returns:
        bool: True if all buckets are full
    """
    for bucket_name, config in BUCKET_CONFIG.items():
        if len(buckets[bucket_name]) < config['capacity']:
            return False
    return True


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def save_results(buckets, filename):
    """
    Save collected papers to a JSON file.
    
    Args:
        buckets: Dict of bucket_name -> list of papers
        filename: Output file path
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(buckets, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {filename}")


def print_summary(buckets):
    """
    Print a summary of collected papers.
    
    Args:
        buckets: Dict of bucket_name -> list of papers
    """
    print("\n" + "=" * 50)
    print("COLLECTION SUMMARY")
    print("=" * 50)
    
    total = 0
    for bucket_name, papers in buckets.items():
        capacity = BUCKET_CONFIG[bucket_name]['capacity']
        count = len(papers)
        total += count
        print(f"  {bucket_name}: {count}/{capacity} papers")
    
    print(f"\nTotal papers collected: {total}")


def process_papers(papers, buckets, client, start_index=0):
    """
    Process a batch of papers: verify with LLM, get citations, add to buckets.
    
    Args:
        papers: List of paper dicts from arXiv
        buckets: Dict of bucket_name -> list of papers
        client: OpenAI client instance
        start_index: Starting index for display purposes (for multi-batch runs)
    
    Returns:
        bool: True if all buckets are full, False otherwise
    """
    for i, paper in enumerate(papers):
        global_index = start_index + i + 1
        print(f"\n[Paper {global_index}] {paper['title'][:60]}...")
        
        # Check if all buckets are full
        if buckets_are_full(buckets):
            print("All buckets are full!")
            return True
        
        # Step 2: Get citation count from Semantic Scholar (check before LLM to save API costs)
        try:
            citation_count = get_citation_count(paper['arxiv_id'])
            print(f"  Citations: {citation_count}")
        except Exception as e:
            print(f"  Skipping: Could not get citation count - {e}")
            continue
        
        # Step 3: Check if bucket for this citation range needs more papers
        bucket_name = get_bucket_name(citation_count)
        if not bucket_name:
            continue

        bucket_capacity = BUCKET_CONFIG[bucket_name]['capacity']
        if len(buckets[bucket_name]) >= bucket_capacity:
            print(f"  Skipping: Bucket '{bucket_name}' already full")
            continue
        
        # Step 4: Verify with LLM (only if we need this paper)
        try:
            is_relevant = verify_with_llm(paper['title'], paper['abstract'], client)
            if not is_relevant:
                print("  Skipping: Not about attention mechanisms (LLM)")
                continue
            print("  LLM verification: Passed")
        except Exception as e:
            print(f"  Skipping: LLM verification failed - {e}")
            continue
        
        # Step 5: Add to bucket
        add_to_bucket(paper, citation_count, buckets)
    
    return False


def main():
    """
    Main workflow: search -> verify -> get citations -> bucket -> save.
    
    This function orchestrates the entire paper collection process:
    1. Initialize OpenAI client and empty buckets
    2. Search arXiv for papers (with pagination - keeps fetching until buckets full)
    3. For each paper:
       a. Verify with LLM that it's about attention mechanisms
       b. Get citation count from Semantic Scholar (throttled to avoid 429s)
       c. Add to appropriate bucket (if space available)
    4. Save results to JSON
    
    The script continues fetching batches from arXiv until:
    - All buckets are full, OR
    - No more papers are returned, OR
    - Maximum batch limit is reached (safety limit)
    
    Error handling: If any step fails for a paper, we skip it and continue.
    """
    print("=" * 50)
    print("arXiv Paper Collector - Attention Mechanisms in NLP")
    print("=" * 50)
    
    # Initialize OpenAI client (uses OPENAI_API_KEY from environment)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return
    
    client = OpenAI(api_key=openai_api_key)
    
    # Initialize empty buckets
    buckets = {name: [] for name in BUCKET_CONFIG.keys()}
    
    # Track how many papers we've processed across all batches
    total_papers_processed = 0
    batch_number = 0
    
    # Keep fetching batches until all buckets are full
    while not buckets_are_full(buckets) and batch_number < ARXIV_MAX_BATCHES:
        batch_number += 1
        start_offset = (batch_number - 1) * ARXIV_BATCH_SIZE
        
        print(f"\n{'='*50}")
        print(f"[Batch {batch_number}] Fetching papers {start_offset + 1} to {start_offset + ARXIV_BATCH_SIZE}")
        print("=" * 50)
        
        # Fetch next batch from arXiv
        try:
            xml_content = search_arxiv(
                "attention mechanisms NLP",
                max_results=ARXIV_BATCH_SIZE,
                start=start_offset
            )
            papers = parse_arxiv_xml(xml_content)
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            break
        
        # Check if we got any papers
        if not papers:
            print("No more papers found from arXiv")
            break
        
        print(f"Processing {len(papers)} papers from this batch...")
        
        # Process this batch of papers
        all_full = process_papers(papers, buckets, client, start_index=total_papers_processed)
        total_papers_processed += len(papers)
        
        if all_full:
            break
        
        # Print intermediate progress
        print_summary(buckets)
    
    # Final save
    print("\n" + "=" * 50)
    print("SAVING FINAL RESULTS")
    print("=" * 50)
    save_results(buckets, OUTPUT_FILE)
    
    # Print final summary
    print_summary(buckets)
    print(f"\nTotal papers processed: {total_papers_processed}")
    print(f"Batches fetched: {batch_number}")


if __name__ == "__main__":
    main()
