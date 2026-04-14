"""
Three right-tailed 2-proportion z-tests, Bonferroni-corrected at alpha = 0.01.

  Test 1 – FN Rate:
    H0: p_LLM == p_random   H1: p_LLM > p_random
    p = shown_ignored / shown_papers

  Test 2 – Seed vs LLM Citation Rate:
    H0: p_seed == p_llm     H1: p_seed > p_llm
    p = citations / exposures  (node 0 skipped: no LLM exposures)

  Test 3 – Top-10% Concentration (LLM vs Random):
    H0: p_LLM == p_random   H1: p_LLM > p_random
    p = citations_to_top10% / total_citations
"""

import json
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

BASE = "archived_synthetic/alpha_10_cap"
ALPHA = 0.05

EXP_EXCL_PATH = f"{BASE}/output/analysis_figures/exclusion_rate_by_node.jsonl"
RND_EXCL_PATH = f"{BASE}/random_output/analysis_figures/exclusion_rate_by_node.jsonl"
SEED_LLM_PATH = f"{BASE}/output/master/seed_vs_llm_citation_rate.jsonl"
LLM_CONC_PATH = f"{BASE}/output/master/concentration.jsonl"
RND_CONC_PATH = f"{BASE}/random_output/master/concentration.jsonl"


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def two_prop_ztest_right(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Right-tailed z-test: H1: p1 > p2. Returns (z_stat, p_value)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    p = norm.sf(z)
    return z, p


def save_table(results: list[dict], headers: list[str], row_fn, title: str, path: str):
    rows = [row_fn(r) for r in results]
    reject_col = len(headers) - 1

    n_rows = len(rows) + 1  # +1 for header
    fig_height = n_rows * 0.38 + 0.7  # 0.38 per row + room for suptitle
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#D9E2F3")
        if r > 0 and c == reject_col:
            val = rows[r - 1][c]
            cell.set_facecolor("#C6EFCE" if val == "Reject H0" else "#FFC7CE")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Test 1: FN rate  (LLM shown-but-ignored rate > random)
# ---------------------------------------------------------------------------

def run_fn_rate_test():
    exp = load_jsonl(EXP_EXCL_PATH)
    rnd = load_jsonl(RND_EXCL_PATH)
    assert len(exp) == len(rnd)

    n_tests = len(exp)
    alpha_bonf = round(ALPHA / n_tests, 6)

    results = []
    for e, r in zip(exp, rnd):
        node = e["node"]
        x1, n1 = e["shown_ignored"], e["shown_papers"]
        x2, n2 = r["shown_ignored"], r["shown_papers"]
        p1 = x1 / n1 if n1 > 0 else float("nan")
        p2 = x2 / n2 if n2 > 0 else float("nan")
        z, p = two_prop_ztest_right(x1, n1, x2, n2)
        results.append({
            "node": node,
            "fn_llm": round(p1, 6),
            "fn_rnd": round(p2, 6),
            "z": round(z, 4),
            "p_value": round(p, 8),
            "alpha_bonferroni": alpha_bonf,
            "reject_h0_bonferroni": int(p < alpha_bonf),
        })

    print(f"\n{'Node':>4}  {'FN LLM':>8}  {'FN Rnd':>8}  {'z':>7}  {'p-value':>10}  {'α (Bonf)':>10}  {'Reject':>7}")
    print("-" * 70)
    for row in results:
        print(
            f"{row['node']:>4}  {row['fn_llm']:>8.4f}  {row['fn_rnd']:>8.4f}  "
            f"{row['z']:>7.3f}  {row['p_value']:>10.8f}  {alpha_bonf:>10.6f}  "
            f"{'Yes' if row['reject_h0_bonferroni'] else 'No':>7}"
        )
    print(f"\nBonferroni: {ALPHA} / {n_tests} = {alpha_bonf}")
    print("H1 (right-tailed): FN_LLM > FN_RND")

    out_path = f"{BASE}/output/master/fn_ztest.jsonl"
    with open(out_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {out_path}")

    headers = ["Node", "FN Rate (LLM)", "FN Rate (Random)", "z", "p-value", f"α (Bonf) = {alpha_bonf}"]
    save_table(
        results, headers,
        lambda r: [
            r["node"], f"{r['fn_llm']:.4f}", f"{r['fn_rnd']:.4f}",
            f"{r['z']:.3f}", f"{r['p_value']:.8f}",
            "Reject H0" if r["reject_h0_bonferroni"] else "Fail to Reject",
        ],
        f"2-Prop Z-Test: False Negative Rate (LLM vs. Random)\n"
        f"H\u2081 (right-tailed): FN_LLM > FN_RND  |  Bonferroni: {ALPHA} / {n_tests} = {alpha_bonf}",
        f"{BASE}/output/master/fn_ztest_table.png",
    )


# ---------------------------------------------------------------------------
# Test 2: Seed vs LLM citation rate  (seed_rate > llm_rate)
# ---------------------------------------------------------------------------

def run_seed_vs_llm_test():
    rows_raw = load_jsonl(SEED_LLM_PATH)
    # skip nodes where llm_exposures == 0
    valid = [r for r in rows_raw if r["llm_exposures"] > 0]
    n_tests = len(valid)
    alpha_bonf = round(ALPHA / n_tests, 6)

    results = []
    for r in valid:
        node = r["node"]
        x1, n1 = r["seed_citations"], r["seed_exposures"]
        x2, n2 = r["llm_citations"], r["llm_exposures"]
        p1 = x1 / n1 if n1 > 0 else float("nan")
        p2 = x2 / n2 if n2 > 0 else float("nan")
        z, p = two_prop_ztest_right(x1, n1, x2, n2)
        results.append({
            "node": node,
            "seed_rate": round(p1, 6),
            "llm_rate": round(p2, 6),
            "z": round(z, 4),
            "p_value": round(p, 8),
            "alpha_bonferroni": alpha_bonf,
            "reject_h0_bonferroni": int(p < alpha_bonf),
        })

    print(f"\n{'Node':>4}  {'Seed Rate':>10}  {'LLM Rate':>9}  {'z':>7}  {'p-value':>10}  {'α (Bonf)':>10}  {'Reject':>7}")
    print("-" * 75)
    for row in results:
        print(
            f"{row['node']:>4}  {row['seed_rate']:>10.4f}  {row['llm_rate']:>9.4f}  "
            f"{row['z']:>7.3f}  {row['p_value']:>10.8f}  {alpha_bonf:>10.6f}  "
            f"{'Yes' if row['reject_h0_bonferroni'] else 'No':>7}"
        )
    print(f"\nBonferroni: {ALPHA} / {n_tests} = {alpha_bonf}  (node 0 excluded: no LLM exposures)")
    print("H1 (right-tailed): seed_rate > llm_rate")

    out_path = f"{BASE}/output/master/seed_vs_llm_ztest.jsonl"
    with open(out_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {out_path}")

    headers = ["Node", "Seed Rate", "LLM Rate", "z", "p-value", f"α (Bonf) = {alpha_bonf}"]
    save_table(
        results, headers,
        lambda r: [
            r["node"], f"{r['seed_rate']:.4f}", f"{r['llm_rate']:.4f}",
            f"{r['z']:.3f}", f"{r['p_value']:.8f}",
            "Reject H0" if r["reject_h0_bonferroni"] else "Fail to Reject",
        ],
        f"2-Prop Z-Test: Seed vs. LLM Citation Rate\n"
        f"H\u2081 (right-tailed): seed_rate > llm_rate  |  Bonferroni: {ALPHA} / {n_tests} = {alpha_bonf}",
        f"{BASE}/output/master/seed_vs_llm_ztest_table.png",
    )


# ---------------------------------------------------------------------------
# Test 3: Top-10% citation concentration  (LLM > random)
# ---------------------------------------------------------------------------

def run_concentration_test():
    llm = load_jsonl(LLM_CONC_PATH)
    rnd = load_jsonl(RND_CONC_PATH)
    assert len(llm) == len(rnd)

    n_tests = len(llm)
    alpha_bonf = round(ALPHA / n_tests, 6)

    results = []
    for lc, rc in zip(llm, rnd):
        node = lc["node"]
        # x = citations going to top-10% papers, n = total citations
        n1 = lc["total_citations"]
        x1 = round(lc["top_10pct_share"] / 100 * n1)
        n2 = rc["total_citations"]
        x2 = round(rc["top_10pct_share"] / 100 * n2)
        z, p = two_prop_ztest_right(x1, n1, x2, n2)
        results.append({
            "node": node,
            "top10_share_llm": round(lc["top_10pct_share"], 4),
            "top10_share_rnd": round(rc["top_10pct_share"], 4),
            "z": round(z, 4),
            "p_value": round(p, 8),
            "alpha_bonferroni": alpha_bonf,
            "reject_h0_bonferroni": int(p < alpha_bonf),
        })

    print(f"\n{'Node':>4}  {'Top10 LLM':>10}  {'Top10 Rnd':>10}  {'z':>7}  {'p-value':>10}  {'α (Bonf)':>10}  {'Reject':>7}")
    print("-" * 78)
    for row in results:
        print(
            f"{row['node']:>4}  {row['top10_share_llm']:>10.2f}%  {row['top10_share_rnd']:>9.2f}%  "
            f"{row['z']:>7.3f}  {row['p_value']:>10.8f}  {alpha_bonf:>10.6f}  "
            f"{'Yes' if row['reject_h0_bonferroni'] else 'No':>7}"
        )
    print(f"\nBonferroni: {ALPHA} / {n_tests} = {alpha_bonf}")
    print("H1 (right-tailed): top10%_LLM > top10%_random")

    out_path = f"{BASE}/output/master/concentration_ztest.jsonl"
    with open(out_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {out_path}")

    headers = ["Node", "Top-10% Share (LLM)", "Top-10% Share (Rnd)", "z", "p-value", f"α (Bonf) = {alpha_bonf}"]
    save_table(
        results, headers,
        lambda r: [
            r["node"], f"{r['top10_share_llm']:.2f}%", f"{r['top10_share_rnd']:.2f}%",
            f"{r['z']:.3f}", f"{r['p_value']:.8f}",
            "Reject H0" if r["reject_h0_bonferroni"] else "Fail to Reject",
        ],
        f"2-Prop Z-Test: Top-10% Citation Concentration (LLM vs. Random)\n"
        f"H\u2081 (right-tailed): top10%_LLM > top10%_random  |  Bonferroni: {ALPHA} / {n_tests} = {alpha_bonf}",
        f"{BASE}/output/master/concentration_ztest_table.png",
    )


def main():
    run_fn_rate_test()
    run_seed_vs_llm_test()
    run_concentration_test()


if __name__ == "__main__":
    main()
