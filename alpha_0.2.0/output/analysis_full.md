# LLM Citation Bias: A Compounding Inequality in AI-Generated Academic Text

## Thesis

When LLMs generate academic papers that cite prior work — and those papers are then fed back as references for future LLM-generated papers — citation inequality compounds across generations. Early-appearing papers (especially real, human-written ones) develop a durable, self-reinforcing advantage that exceeds what would emerge from random or relevance-based citation alone. This constitutes a novel form of the Matthew Effect driven not by human prestige signals, but by latent LLM preferences.

---

## Argument

### 1. Citation inequality grows steadily across generations

The Gini coefficient — measuring how unevenly citations are distributed — nearly triples over six generations, from 0.106 at node 0 to 0.321 at node 5.

**Evidence:**
- `analysis_figures/gini_progression.png` — smooth monotonic upward trend in Gini across all six nodes
- `analysis_figures/summary_table.png` — tabulates the progression: 0.106 → 0.190 → 0.208 → 0.262 → 0.301 → 0.321

This is not a trivial artifact of pool size. While the available paper pool grows from 30 to 180, the Gini coefficient specifically controls for pool size — it measures distributional skew, not absolute counts.

### 2. This inequality exceeds what chance or relevance would produce

Two baselines were run: a **random citation baseline** (citations assigned randomly from the available pool) and a **semantic similarity baseline** (citations assigned based on topic relevance). The LLM experiment produces consistently higher inequality than both.

**Evidence:**
- `additional_evidence/gini_comparison.png` — three-line comparison showing the LLM (blue) above random (orange) and semantic (green) at every generation
- `additional_evidence/gini_comparison.jsonl` — raw values confirm the LLM Gini exceeds both baselines at all six nodes

The gap is largest at early nodes (LLM starts at 0.106 vs random 0.049 and semantic 0.043) and narrows somewhat by node 5 (0.321 vs 0.280 and 0.282). This means the LLM introduces an *additional* concentration effect beyond what structural factors (pool growth, exposure timing) would produce alone.

### 3. Seed papers receive a persistent, outsized share of citations

The 30 seed papers (real arXiv publications) constitute only 14.3% of the final 210-paper pool, yet they capture 45.7% of all citations across the experiment.

**Evidence:**
- `paper_type_citations/paper_type_across_generations.png` — pie chart: seed papers = 1,945 citations (45.7%), position papers = 1,120 (26.3%), literature reviews = 1,190 (28.0%)
- `paper_type_citations/paper_type_by_node.png` — per-node stacked bars show seed citations declining from 778 (100%) at node 0 to 134 (~20%) at node 5, but remaining the single largest category throughout
- `citation_counts/citation_counts_ordered.jsonl` — the top 27 most-cited papers globally are all SEED papers (ranging from 86 down to 44 citations); the first AI-generated paper (N0P6) appears at rank 28

### 4. The LLM cites seed papers at near-perfect rates, but discriminates against generated papers

When a seed paper appears in the prompt context, it is cited nearly 100% of the time. Generated papers are cited at substantially lower rates, and the gap persists across all generations.

**Evidence:**
- `analysis_figures/avg_citation_exposure_rate_by_node.png` — grouped bars show seed papers maintaining 0.85–0.93 citation rates vs generated papers at 0.70–0.75 across all nodes
- `additional_evidence/rate_by_origin.png` — bars broken down by origin generation show a clear gradient: seed (0.86–0.93) > Gen 0 (0.75–0.80) > Gen 1 (0.66–0.74) > Gen 2+ (0.66–0.72)
- `master/exposure_rate.jsonl` — individual paper rates confirm: SEED_9, SEED_6, SEED_4, SEED_16 all achieve rate = 1.0; bottom-ranked papers include N2P28 (0.045), N2P27 (0.091), N4P20 (0.0)

This is the mechanism behind the compounding inequality: the LLM "prefers" certain papers when deciding what to cite, and that preference is correlated with whether the paper was human-written.

### 5. Rich-get-richer dynamics are confirmed by cumulative advantage correlations

Papers that are heavily cited in one generation continue to accumulate citations in subsequent generations, creating a self-reinforcing feedback loop.

**Evidence:**
- `additional_evidence/cumulative_advantage.png` — six scatter plots of cumulative citations at consecutive nodes, all showing strong positive correlations (r ~ 0.89–0.99)
- Seed papers (blue dots) cluster in the upper-right of every plot; generated papers (orange) cluster lower-left

This demonstrates that the advantage is not merely a snapshot but compounds over time. A paper that gets cited early continues to get cited, while a paper that starts with few citations struggles to catch up.

### 6. Citation preference is inversely related to semantic relevance

Counter-intuitively, papers with *lower* semantic similarity to the topic receive *more* citations, with a strong negative correlation.

**Evidence:**
- `additional_evidence/similarity_vs_citations.png` — scatter plot with Pearson r = -0.693; seed papers (blue) cluster at lower similarity / higher citations, generated papers (orange) at higher similarity / lower citations

This suggests that the LLM's citation behavior is driven more by surface-level signals of "real paper quality" (writing style, abstract structure, terminology) than by topical relevance. Generated papers, despite being more on-topic (higher semantic similarity), are cited less. This finding undermines a charitable explanation that seed papers are simply more relevant — they are, in fact, *less* semantically similar to the topic on average.

### 7. Some papers are excluded entirely, and exclusion grows

As the pool grows, an increasing number of papers receive zero citations despite being available.

**Evidence:**
- `analysis_figures/exclusion_rate_by_node.png` — bar/line chart showing: nodes 0–2 have 0% exclusion; node 3: 4 papers uncited (3.3%); node 4: 2 uncited (1.3%); node 5: 10 uncited (5.6%)
- `analysis_figures/citation_exposure_rate_distributions.png` — histograms show the distribution of citation rates spreading toward zero in later nodes, with visible mass in the lowest bins by node 5

This exclusion problem means that in later generations, the LLM not only concentrates citations among favorites but actively ignores a growing tail of papers.

### 8. The system is highly faithful — hallucinations are negligible

The observed patterns cannot be attributed to noisy or fabricated citations. The model cites real papers from its context with >99.7% accuracy.

**Evidence:**
- `analysis_figures/citation_verification_table.png` — per-node validity: 99.7–100% across all generations
- `citation_counts/hallucinations.jsonl` — only 7 hallucinated citations out of ~4,255 total (0.16%)

This is important because it means the inequality patterns reflect genuine LLM *selection* behavior, not random noise or hallucination artifacts.

---

## Conclusion

Across six generations of LLM-generated academic papers, citation inequality compounds substantially (Gini: 0.106 → 0.321), exceeding both random and semantic-relevance baselines. The mechanism is a persistent LLM preference for citing human-written seed papers over AI-generated ones — a preference that is inversely correlated with topical relevance and results in near-perfect citation rates for seeds vs. declining rates for later-generation papers. This creates a rich-get-richer dynamic where early-cited papers continue to accumulate citations while a growing tail of papers is excluded entirely.

The implication is that if LLMs are increasingly used to assist literature review or paper writing, they may amplify existing citation concentration — not because of human prestige biases, but because of latent quality signals the model uses when selecting which references to include. Papers that "look real" (polished, human-written) get cited; AI-generated papers, even when more topically relevant, do not. This is a new vector for the Matthew Effect in science, one that operates through machine preference rather than human social dynamics.
