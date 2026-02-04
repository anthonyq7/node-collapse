import asyncio, json, os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict
from collections import defaultdict

load_dotenv()

MIN_CITATIONS = 5
MAX_CITATIONS = 10
POOLED_PAPERS = []
TOTAL_NODES = 5
MODEL = "gpt-5-mini"
MAX_CONCURRENT = 5
TARGET_LENGTH = 500
TOPIC = "Attention mechanisms in natural language processing/deep learning"

SYSTEM_PROMPT = """
You are an expert academic researcher specializing in writing comprehensive literature reviews.

You must output your response as valid JSON with this exact structure:

{
    "title": "Your literature review title here",
    "review": "Your literature review text here with inline citations...",
    "cited_ids": ["SEED_0", "N1P2", etc.]
}

IMPORTANT:
- Output ONLY valid JSON, no other text
- The "review" field contains the full literature review text
- The "cited_ids" field is an array of article IDs you referenced
- Include each unique ID only once in the array
- ONLY cite from the available pool of articles
- YOU MUST cite only using the IDs, NO authors or publish dates
"""

client = AsyncOpenAI()
CITATION_COUNTS = defaultdict(int)
POSSIBLE_PAPER_IDS = set()


async def generate_paper(paper_id: str, semaphore = asyncio.Semaphore(MAX_CONCURRENT)):

    USER_PROMPT = f"""I need you to write a comprehensive literature review. 

    LITERATURE REVIEW REQUIREMENTS:
    Topic: {TOPIC}

    Writing Guidelines:
    - Synthesize findings across papers thematically, don't just summarize each paper individually
    - Identify key trends, patterns, and developments in the field
    - Highlight consensus areas and points of debate or contradiction
    - Note any significant gaps in the literature
    - Organize by themes/topics, not chronologically or by paper
    - Use inline citations in the format: (Author et al., Year)
    - Target length: {TARGET_LENGTH} words

    CITATION REQUIREMENTS:
    - Only cite papers from the provided list below
    - Track every article you cite using its "id" field
    - You must return all cited IDs in the "cited_ids" array in your JSON response
    - It's okay to cite a paper multiple times, but only include its ID once in the final list
    - Only cite between {MIN_CITATIONS} to {MAX_CITATIONS} different papers. 

    AVAILABLE ARTICLES:
    Below is a list of research articles in JSON format. Each article has an "id" field - this is what you'll use to track citations.

    {POOLED_PAPERS}
    """

    PROMPT = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": USER_PROMPT
        }
    ]
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model = MODEL,
                messages = PROMPT,
                max_completion_tokens=5000,
                response_format={"type": "json_object"}
            )

            
            if not response:
                print(f"Empty response{paper_id}")
                print(f"Retrying....")
                await asyncio.sleep(5)

                response = await client.chat.completions.create(
                    model = MODEL,
                    messages = PROMPT,
                    max_completion_tokens=10000,
                    response_format={"type": "json_object"}
                )

                if not response:
                    return paper_id, None
            
            # Print token usage
            if response.usage:
                print(f"[{paper_id}] Input: {response.usage.prompt_tokens} tokens | Output: {response.usage.completion_tokens} tokens | Total: {response.usage.total_tokens} tokens")
            
            return paper_id, response.choices[0].message.content

        except Exception as e:
            print(f"API Error: {e}")
            print(f"Paper: {paper_id}")
            await asyncio.sleep(5)
            return paper_id, None

async def generate_node(node: int):

    print("\n")
    print(30*"=")
    print(f"Starting node {node}...")
    print(30*"=")
    print("\n")

    wrapped_tasks = []
    id_list = []
    node_citations = defaultdict(int)
    for i in range(30):
        wrapped_tasks.append(generate_paper(f"N{node}P{i}"))
        id_list.append(f"N{node}P{i}")

    with open(f"output/node_{node}/node_{node}.jsonl", "w") as f:
        for task in asyncio.as_completed(wrapped_tasks):
            paper_id, json_response = await task

            if not json_response:
                print(f"{paper_id} was empty...")
                continue

            print(f"Finished {paper_id}...")

            json_response = json.loads(json_response)

            title = json_response.get("title", "").strip()
            review = json_response.get("review", "").strip()
            citations = json_response.get("cited_ids")

            if citations:
                for c in citations:
                    if c in POSSIBLE_PAPER_IDS:
                        CITATION_COUNTS[c] += 1
                
            
            toDump = {
                "id": paper_id,
                "title": title,
                "review": review,
                "citations": citations
            }

            for cite in citations:
                node_citations[cite] += 1

            new_paper = {
                "id": paper_id,
                "title": title,
                "review": review
            }

            POOLED_PAPERS.append(new_paper)
            
            f.write(json.dumps(toDump) + "\n")
            f.flush()

    print("\n")
    print(30*"=")
    print(f"Node {node} Statistics")
    print(30*"=")
    print("\n")

    with open(f"output/node_{node}/node_{node}_stats.jsonl", "w") as f:
        for k, v in node_citations.items():
            f.write(json.dumps({k:v}) + "\n")
            print(f"{k}: {v}")
        
        f.flush()

    for id in id_list:
        POSSIBLE_PAPER_IDS.add(id)

def get_arxiv():
    try:
        with open("arxiv_papers.json") as f:
            pooled = json.load(f)
            return pooled
    except FileNotFoundError as e:
        print(f"arxiv_papers.json not found...")
        return None

def standardize_arxiv() -> List[Dict]:
    arxiv = get_arxiv()
    if not arxiv:
        return False

    papers = arxiv.get("bucket_20_100")
    arxiv_list = []
    arxiv_citation_count = []

    for i, paper in enumerate(papers):
        paper_id = f"SEED_{i}"
        title = paper.get("title")
        abstract = paper.get("abstract")
        citation_count = paper.get("citation_count")

        output_object = {
            "id": paper_id,
            "title": title,
            "abstract": abstract
        }

        citation_object = {
            "id": paper_id,
            "citation_count": citation_count
        }

        arxiv_list.append(output_object)
        arxiv_citation_count.append(citation_object)
        POSSIBLE_PAPER_IDS.add(paper_id)
        CITATION_COUNTS[paper_id] = citation_count
    
    with open("output/seed/seed_initial.jsonl", "w") as f:
        for k, v in CITATION_COUNTS.items():
            item = {k : v}
            f.write(json.dumps(item) + "\n")

    with open("output/seed/seed.jsonl", "w") as f:
        for item in arxiv_list:
            f.write(json.dumps(item) + "\n")   

    return arxiv_list 

async def run_experiment():

    #Make output directories 
    os.makedirs("output/seed", exist_ok=True)
    os.makedirs("output/master", exist_ok=True)
    os.makedirs("output/citation_counts", exist_ok=True)

    #standardizes arXiv papers and adds them to the pool
    #Additionally, saves the initial citation counts + add another running citation count to citation_counts
    POOLED_PAPERS.extend(standardize_arxiv())

    for i in range(TOTAL_NODES):
        os.makedirs(f"output/node_{i}", exist_ok=True)
    
    for i in range(TOTAL_NODES):
        await generate_node(i)

    with open("output/citation_counts/citation_counts.jsonl", "w") as f:
        for item in CITATION_COUNTS:
            f.write(json.dumps(item) + "\n")

    #save citation counts
    print("Finished generating nodes")

if __name__ == "__main__":
    asyncio.run(run_experiment())