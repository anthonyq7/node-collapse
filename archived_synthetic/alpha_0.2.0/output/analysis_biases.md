# Identified Biases in the LLM Citation Experiment

## Biases Revealed by the Experiment

### 1. Quality/Authenticity Bias

The LLM consistently prefers citing human-written seed papers over AI-generated ones. Seed exposure rates hover at 0.85–0.93, while generated papers sit at 0.65–0.80. The model appears to detect some latent quality signal — more polished abstracts, more varied vocabulary, more natural sentence structure — and preferentially cites those papers. This is the strongest bias in the data.

**Evidence:** `rate_by_origin.jsonl`, `avg_citation_exposure_rate_by_node.png`

### 2. Inverse Relevance Bias

The model cites papers that are *less* topically relevant. Seed papers average ~0.54 semantic similarity to the topic while generated papers average ~0.77, yet seeds get far more citations (r = -0.69). The LLM is not optimizing for topical fit — it's favoring something else (writing quality, specificity, or novelty of framing).

**Evidence:** `similarity_vs_citations.png`, `similarity_vs_citations.jsonl`

### 3. Incumbency / First-Mover Bias

Papers that exist earlier in the generational pipeline accumulate a compounding advantage. Cumulative advantage scatter plots show r ~ 0.89–0.99 correlations between citations at consecutive nodes. A paper cited heavily early on continues to be cited, not because it gets "more relevant" but because the model has already seen it referenced in prior generated papers' bodies/abstracts — a self-reinforcing loop.

**Evidence:** `cumulative_advantage.png`

### 4. Exclusion Bias

The model doesn't just under-cite some papers — it completely ignores a growing fraction. By node 5, 10 of 180 papers (5.6%) receive zero citations despite being available. The excluded papers are disproportionately from later generations. N4P20 has a 0.0 citation rate — exposed twice, cited zero times. This creates a "silent death" for certain papers.

**Evidence:** `exclusion_rate_by_node.png`, `exposure_rate.jsonl`

### 5. Generation-Origin Bias

There is a clear gradient in exposure rates by origin generation: Seed > Gen 0 > Gen 1 > Gen 2+. Even among AI-generated papers, earlier-generation papers are cited more reliably than later ones. At node 5, seed papers have a 0.934 rate while Gen 1 papers sit at 0.665. The model treats "older" papers as more authoritative.

**Evidence:** `rate_by_origin.jsonl`, `rate_by_origin.png`

---

## Potential Methodological Biases in the Experiment Design

### 6. Structural Exposure Bias

Seed papers appear in more generations (all 6 nodes) while later papers appear in fewer (Gen 4 papers only appear in node 5). This means seeds have more *opportunities* to be cited. The exposure rate metric partially controls for this, but the cumulative reputation effect cannot be disentangled from cumulative exposure. A paper cited 86 times across 6 generations (SEED_3) vs. a Gen 4 paper that can only be cited in 1 generation are not on equal footing.

**Evidence:** `citation_counts_ordered.jsonl`, `citation_by_origin.jsonl`

### 7. Information Asymmetry in Prompt Context

Each prompt provides 30 randomly sampled papers as context. Seed papers carry real arXiv titles and abstracts, while generated papers carry LLM-produced titles and abstracts. The model sees qualitatively different input for each — real abstracts tend to be more specific and information-dense, which may make them easier to cite meaningfully. This conflates "bias toward real papers" with "bias toward higher-information abstracts."

**Evidence:** `experiment.py` (lines 48–49, 256–260), `seed/seed.jsonl`

### 8. Anonymization Artifact

Seed papers get fake surnames and years in the range 2017–2022 (`experiment.py` line 319), while generated papers get years 2017–2025 (line 63). The year ranges overlap but differ. If the LLM has any preference for citing older-seeming work (a plausible "foundational work" heuristic), the narrower seed year range (skewing slightly older) could amplify seed citation rates. The fake names via Faker are drawn from the same pool, so name bias is unlikely.

**Evidence:** `experiment.py` (lines 319, 63)

### 9. Single-Topic, Single-Model Bias

The experiment uses one topic ("Knowledge distillation or model compression in deep learning or NLP") and one model (GPT-5-mini). The observed biases may be topic-specific (e.g., this field has distinctive writing conventions that the model recognizes) or model-specific (a different LLM might exhibit different citation preferences). The results don't yet generalize.

**Evidence:** `experiment.py` (lines 15, 20)

### 10. Fixed Pool Size per Generation

Every generation produces exactly 30 papers and samples exactly 30 references from the pool. As the pool grows from 30 to 180, each paper's probability of being sampled drops from 100% to 16.7%. This dilution is mechanical and unavoidable, but it means that later-generation papers are structurally disadvantaged in exposure — they have fewer chances to be seen before the experiment ends.

**Evidence:** `experiment.py` (lines 12, 19), `summary_table.png`

### 11. No Content Feedback Loop

The experiment tracks citations but doesn't model whether highly-cited papers *influence the content* of later papers. In real academia, heavily cited work shapes the vocabulary and framing of a subfield. Here, generated papers may echo seed paper language, further reinforcing seed paper citability in a way the experiment captures but doesn't isolate.

**Evidence:** `citation_by_origin.png`, node JSONL files (body text)
