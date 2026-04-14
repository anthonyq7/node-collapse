import json
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = "archived_synthetic/alpha_0.2.0_scaled"
MAX_NODES = 12
OUTPUT_ROOT = f"{BASE}/output"


def load_token_usage(node: int) -> tuple[float, float]:
    path = f"{OUTPUT_ROOT}/node_{node}/node_{node}_token_usage.jsonl"
    prompt_tokens = []
    completion_tokens = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            prompt_tokens.append(data["prompt_tokens"])
            completion_tokens.append(data["completion_tokens"])
    mean_prompt = np.mean(prompt_tokens) if prompt_tokens else 0.0
    mean_completion = np.mean(completion_tokens) if completion_tokens else 0.0
    return mean_prompt, mean_completion


def main():
    output_dir = f"{OUTPUT_ROOT}/master"
    os.makedirs(output_dir, exist_ok=True)

    nodes = list(range(MAX_NODES))
    mean_prompts = []
    mean_completions = []

    for node in nodes:
        mean_prompt, mean_completion = load_token_usage(node)
        mean_prompts.append(mean_prompt)
        mean_completions.append(mean_completion)
        print(
            f"Node {node}: mean input={mean_prompt:.1f}, mean output={mean_completion:.1f}"
        )

    x = np.arange(len(nodes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(
        x - width / 2,
        mean_prompts,
        width,
        label="Mean Input Tokens",
        color="#2980B9",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    b2 = ax.bar(
        x + width / 2,
        mean_completions,
        width,
        label="Mean Output Tokens",
        color="#27AE60",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=8)
    ax.set_xlabel("Node (Generation)")
    ax.set_ylabel("Mean Token Count")
    ax.set_title("Mean Input & Output Tokens per Node — Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in nodes])
    ax.legend()
    fig.tight_layout()

    out_path = f"{output_dir}/mean_token_usage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
