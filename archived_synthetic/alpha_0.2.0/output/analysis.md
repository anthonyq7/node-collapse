# Citation Bias Experiment — Analysis

## Experiment Design

This is a multi-generational citation propagation experiment. 30 real arXiv papers on knowledge distillation/model compression are used as seeds, then GPT-4.5-mini generates 30 new papers per generation (nodes 0-5), each citing from the pool of all prior papers. The question: does citation inequality grow as AI-generated papers cite AI-generated papers?

## Key Findings

### 1. Rising Inequality (Gini Coefficient)

The Gini coefficient steadily increases across generations:

| Node | Gini  | Total Citations | Unique Cited | Available |
|------|-------|-----------------|--------------|-----------|
| 0    | 0.106 | 778             | 30/30        | 30        |
| 1    | 0.190 | 748             | 60/60        | 60        |
| 2    | 0.208 | 701             | 90/90        | 90        |
| 3    | 0.262 | 686             | 116/120      | 120       |
| 4    | 0.301 | 665             | 148/150      | 150       |
| 5    | 0.321 | 677             | 170/180      | 180       |

Gini triples from 0.106 to 0.321 — citation distribution becomes substantially more unequal over generations.

### 2. Seed Papers Dominate Overall

The top ~27 most-cited papers globally are **all SEED papers** (ranging from 86 down to 44 citations). The first AI-generated paper to appear is N0P6 at rank 28 with 47 citations. This is a strong first-mover / incumbency advantage — seed papers accumulate citations across all 6 generations while later papers have fewer generations to be cited in.

### 3. Top-5 Concentration Declines Per-Node (But That's Mechanical)

Top-5 share drops from 19.3% (node 0) to 6.4% (node 5), but this is largely because the pool grows from 30 to 180 papers. With a uniform distribution, top-5 share would be 16.7% at node 0 and 2.8% at node 5. The observed shares are consistently above uniform, confirming preferential concentration.

### 4. Near-Perfect Exposure Rates for Seeds

Most seed papers have exposure rates at or near 1.0 — meaning when a seed paper is included in the prompt context, it is almost always cited. For example, SEED_9, SEED_6, SEED_4, SEED_24, SEED_16 all have rate = 1.0. This suggests the model treats seed papers as highly authoritative, possibly because they have more polished abstracts (being real papers).

### 5. Later-Generation Papers Get Ignored

Later-generation papers show much lower exposure rates. The bottom of the list includes papers like N2P28 (rate 0.045), N2P27 (0.091), N0P3 (0.222), and N4P20 (0.0 — never cited despite being exposed). The model appears to discriminate based on some quality signal, citing AI-generated papers less reliably than real ones.

### 6. Growing Exclusion

By node 3, 4 papers go uncited (116/120 cited). By node 5, 10 papers are uncited (170/180). The exclusion rate grows, suggesting a "long tail" problem where some papers fall off entirely.

### 7. Very Few Hallucinations

Only 7 hallucinated citations across ~4,255 total citations (~0.16%) — the model is remarkably faithful to the provided references rather than inventing fake ones.

## Interpretation

The experiment demonstrates a **compounding citation bias** in AI-generated academic text. Seed papers (real arXiv papers) enjoy a persistent advantage not just from appearing earlier, but from being cited at near-100% rates when shown to the model. AI-generated papers from later generations are increasingly likely to be overlooked. This is a synthetic analog of the Matthew Effect ("rich get richer") in citation networks, but driven by LLM preferences rather than human prestige signals.

The rising Gini coefficient is the central result — even starting from a relatively equal baseline (0.106), inequality roughly triples in 6 generations. If this pattern holds in real-world scenarios where LLMs assist in literature review or paper writing, it could amplify existing citation concentration in academia.
