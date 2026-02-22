import asyncio, json, os, random, re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict
from collections import defaultdict

load_dotenv()

MIN_CITATIONS = 5
MAX_CITATIONS = 10
POOLED_PAPERS = []
TOTAL_NODES = 6
MODEL = "gpt-5-mini"
MAX_CONCURRENT = 5
TARGET_LENGTH = 500
SEED = 42
PAPER_SET_LENGTH = 30
TOPIC = "Knowledge distillation or model compression in deep learning or NLP"
client = AsyncOpenAI()
CITATION_COUNTS = defaultdict(int)
POSSIBLE_PAPER_IDS = set()

SYSTEM_PROMPT = """
    You are a researcher writing about a topic using a provided set of articles.
    When referencing an article, cite it inline using only its ID (e.g., SEED_0, N1P2).
    Do not use author names, publication dates, or any other identifying information.
    Output valid JSON only:
    {
        "title": "...",
        "review": "... inline citations only ..."
    }
    """

def create_set():
    return random.sample(POOLED_PAPERS, 30)

async def generate_paper(paper_id: str, semaphore = asyncio.Semaphore(MAX_CONCURRENT)):

    random_papers = create_set()

    USER_PROMPT = f"""
    Write about the following topic using only the articles provided below.
    Synthesize what these articles say about the topic. 
    Support claims by citing relevant articles by ID.

    Topic: {TOPIC}

    Articles:
    {random_papers}
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
                    return paper_id, None, random_papers
            
            # Print token usage
            if response.usage:
                print(f"[{paper_id}] Input: {response.usage.prompt_tokens} tokens | Output: {response.usage.completion_tokens} tokens | Total: {response.usage.total_tokens} tokens")
                print(f"Number of input papers: {len(random_papers)}")
            
            return paper_id, response.choices[0].message.content, random_papers

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

    with open(f"alpha/output/node_{node}/node_{node}.jsonl", "w") as f, open(f"alpha/output/node_{node}/node_{node}_inputs.jsonl", "w") as f_i:
        for task in asyncio.as_completed(wrapped_tasks):
            paper_id, json_response, papers_seen = await task

            if not json_response:
                print(f"{paper_id} was empty...")
                continue

            print(f"Finished {paper_id}...")

            json_response = json.loads(json_response)

            title = json_response.get("title", "").strip()
            review = json_response.get("review", "").strip()
            
            citations = list(get_citations(review))

            papers_id_seen = []
            for p in papers_seen:
                papers_id_seen.append(p.get("id"))
            
            toDump = {
                "id": paper_id,
                "title": title,
                "content": review,
                "citations": citations, 
                "papers_seen": papers_id_seen
            }

            for cite in citations:
                node_citations[cite] += 1

            new_paper = {
                "id": paper_id,
                "title": title,
                "content": review
            }

            input_papers = {
                "id": paper_id,
                "papers_seen": papers_seen
            }

            POOLED_PAPERS.append(new_paper)
            
            f.write(json.dumps(toDump) + "\n")
            f_i.write(json.dumps(input_papers) + "\n")
            f.flush()
            f_i.flush()

    print("\n")
    print(30*"=")
    print(f"Node {node} Statistics")
    print(30*"=")
    print("\n")

    with open(f"alpha/output/node_{node}/node_{node}_stats.jsonl", "w") as f:
        for k, v in node_citations.items():
            f.write(json.dumps({k:v}) + "\n")
            print(f"{k}: {v}")
        
        f.flush()

    for id in id_list:
        POSSIBLE_PAPER_IDS.add(id)

def get_seed():
    try:
        pooled = []
        with open("alpha/buckets/bucket_100_249.jsonl") as f:
            for paper in f:
                pooled.append(json.loads(paper))
            
        return pooled
    except FileNotFoundError as e:
        print(f"alpha/buckets/bucket_100_249.jsonl not found...")
        return None

def standardize_seed() -> List[Dict]:
    seed = get_seed()
    if not seed:
        return False

    papers = seed

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
            "content": abstract
        }

        citation_object = {
            "id": paper_id,
            "citation_count": citation_count
        }

        arxiv_list.append(output_object)
        arxiv_citation_count.append(citation_object)
        POSSIBLE_PAPER_IDS.add(paper_id)
        CITATION_COUNTS[paper_id] = citation_count
    
    with open("alpha/output/seed/seed_initial.jsonl", "w") as f:
        for k, v in CITATION_COUNTS.items():
            item = {k : v}
            f.write(json.dumps(item) + "\n")

    with open("alpha/output/seed/seed.jsonl", "w") as f:
        for item in arxiv_list:
            f.write(json.dumps(item) + "\n")   

    return arxiv_list 

def get_citations(text: str):
    pattern = r"\b(SEED_\d+|N\d+P\d+)\b"
    return set(re.findall(pattern, text))


async def run_experiment():

    random.seed(SEED)

    #Make output directories 
    os.makedirs("alpha/output/seed", exist_ok=True)
    os.makedirs("alpha/output/master", exist_ok=True)
    os.makedirs("alpha/output/citation_counts", exist_ok=True)

    #standardizes arXiv papers and adds them to the pool
    #Additionally, saves the initial citation counts + add another running citation count to citation_counts
    POOLED_PAPERS.extend(standardize_seed())

    for i in range(TOTAL_NODES):
        os.makedirs(f"alpha/output/node_{i}", exist_ok=True)
    
    for i in range(TOTAL_NODES):
        await generate_node(i)

    with open("alpha/output/citation_counts/citation_counts.jsonl", "w") as f:
        for item in CITATION_COUNTS:
            f.write(json.dumps(item) + "\n")

    #save citation counts
    print("Finished generating nodes")

if __name__ == "__main__":
    asyncio.run(run_experiment())