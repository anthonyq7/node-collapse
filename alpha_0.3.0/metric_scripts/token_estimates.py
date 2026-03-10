"""
Compute token estimates per paper from experiment run data.
Input token metrics exclude system and user prompt overhead (papers only).
Token counts come from OpenAI API calls (3 brief calls), matching experiment.py.
"""

import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PAPER_SET_LENGTH = 30
TOTAL_NODES = 3
TOPIC = "Knowledge distillation or model compression in deep learning or NLP"
MODEL = "gpt-5-mini"

# Prompts from experiment.py (must stay in sync)
SYSTEM_PROMPT = """
    You are a researcher writing about a topic using a provided set of articles.
    Output valid JSON only:
    {
        "title": "...",
        "abstract": "... no citations ...",
        "body": "... support claims by citing relevant articles inline using parenthetical citation format, e.g. ([Surname], [Year]) ..."
    }
    Do not include citations in the abstract.
    Only cite articles from the provided list using their exact author and year via inline citations.
    """

PP_USER_PROMPT = """
    Using only the articles provided, argue a position on the following topic.
    Support claims by citing relevant articles inline.
    Only cite articles from the provided list using their exact author and year via inline citations.

    Topic: """ + TOPIC + """

    Articles:
    []
"""

LR_USER_PROMPT = """
    Using only the articles provided, synthesize what is known about the following topic.
    Support claims by citing relevant articles inline.
    Only cite articles from the provided list using their exact author and year via inline citations.

    Topic: """ + TOPIC + """

    Articles:
    []
"""

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
client = OpenAI()


def get_prompt_tokens(messages: List[Dict]) -> int:
    """Make a brief API call and return prompt_tokens from response.usage."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=10,
    )
    return response.usage.prompt_tokens


def load_node_token_usage(node: int) -> List[Dict]:
    path = OUTPUT_DIR / f"node_{node}" / f"node_{node}_token_usage.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_node_papers_seen(node: int) -> Dict[str, List[str]]:
    path = OUTPUT_DIR / f"node_{node}" / f"node_{node}.jsonl"
    id_to_papers_seen = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            id_to_papers_seen[record["id"]] = record.get("papers_seen_id", [])
    return id_to_papers_seen


def count_seed_papers(papers_seen_id: List[str]) -> int:
    count = 0
    for paper_id in papers_seen_id:
        if paper_id.startswith("SEED_"):
            count += 1
    return count


def main():
    # Brief API calls to get token counts (matches experiment.py usage)
    pp_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PP_USER_PROMPT},
    ]
    lr_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": LR_USER_PROMPT},
    ]

    pp_prompt_tokens = get_prompt_tokens(pp_messages)
    lr_prompt_tokens = get_prompt_tokens(lr_messages)

    system_prompt_user_prompt_tokens = (pp_prompt_tokens + lr_prompt_tokens) / 2

    # Third call: baseline for system-only tokens
    baseline_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ""},
    ]
    system_prompt_tokens = get_prompt_tokens(baseline_messages)
    averaged_user_prompt_tokens = system_prompt_user_prompt_tokens - system_prompt_tokens

    # 1. Mean input tokens per seed paper (Node 0), papers only
    node_0_records = load_node_token_usage(0)
    node_0_paper_tokens = []
    for record in node_0_records:
        total_prompt = record["prompt_tokens"]
        paper_tokens = total_prompt - system_prompt_user_prompt_tokens
        if paper_tokens < 0:
            paper_tokens = 0
        node_0_paper_tokens.append(paper_tokens)

    total_node_0_paper = sum(node_0_paper_tokens)
    count_node_0_prompts = len(node_0_paper_tokens)
    mean_paper_tokens_per_node_0_prompt = total_node_0_paper / count_node_0_prompts
    mean_input_tokens_seed = mean_paper_tokens_per_node_0_prompt / PAPER_SET_LENGTH

    # 2. Mean input tokens per LLM-generated paper (Node 1+), papers only
    mean_tokens_per_seed_paper = mean_input_tokens_seed

    llm_tokens_total = 0
    llm_papers_total = 0

    for node in range(1, TOTAL_NODES):
        token_records = load_node_token_usage(node)
        papers_seen_by_id = load_node_papers_seen(node)

        for record in token_records:
            paper_id = record["id"]
            total_prompt = record["prompt_tokens"]
            paper_tokens = total_prompt - system_prompt_user_prompt_tokens
            if paper_tokens < 0:
                paper_tokens = 0

            papers_seen_id = papers_seen_by_id.get(paper_id, [])
            seed_count = count_seed_papers(papers_seen_id)
            seed_tokens = mean_tokens_per_seed_paper * seed_count
            llm_tokens = paper_tokens - seed_tokens

            if llm_tokens < 0:
                llm_tokens = 0

            llm_papers_in_prompt = PAPER_SET_LENGTH - seed_count
            llm_tokens_total += llm_tokens
            llm_papers_total += llm_papers_in_prompt

    if llm_papers_total > 0:
        mean_input_tokens_llm = llm_tokens_total / llm_papers_total
    else:
        mean_input_tokens_llm = 0.0

    # 3. Mean output tokens per generated paper
    completion_tokens_total = 0
    completion_count = 0

    for node in range(TOTAL_NODES):
        token_records = load_node_token_usage(node)
        for record in token_records:
            completion_tokens_total += record["completion_tokens"]
            completion_count += 1

    mean_output_tokens = completion_tokens_total / completion_count

    estimates = {
        "mean_input_tokens_seed": round(mean_input_tokens_seed, 3),
        "mean_input_tokens_llm": round(mean_input_tokens_llm, 3),
        "mean_output_tokens": round(mean_output_tokens, 3),
        "system_prompt_tokens": round(system_prompt_tokens, 3),
        "averaged_user_prompt_tokens": round(averaged_user_prompt_tokens, 3),
        "system_prompt_user_prompt_tokens": round(system_prompt_user_prompt_tokens, 3),
    }

    output_path = OUTPUT_DIR / "token_estimates.json"
    with open(output_path, "w") as f:
        json.dump(estimates, f, indent=2)

    print(f"Saved token estimates to {output_path}")
    for key, value in estimates.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the experiment has been run and output files exist.")
        raise
