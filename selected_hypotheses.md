# Selected Hypotheses

Selected **4** hypotheses across **1** anomalies.

## Anomaly a001 — benchmark_inconsistency

**Central question:** When does RAG help on domain-QA, and when does it fail?

**Shared entities:** method=RAG, task=domain-QA

**Evidence claims:**
- `c001` (openalex:W4402670290 — "Benchmarking Retrieval-Augmented Generation for Medicine", 2024, positive): MEDRAG improves the accuracy of six different LLMs by up to 18% over chain-of-thought prompting
- `c002` (openalex:W4402670290 — "Benchmarking Retrieval-Augmented Generation for Medicine", 2024, positive): Combination of various medical corpora and retrievers achieves the best performance
- `c004` (openalex:W4387156782 — "Design and Evaluation of a Retrieval-Augmented Generation Architecture for OWASP Security Data", 2023, positive): RAG system enables security-focused question answering with reduced risk of hallucinated responses
- `c005` (openalex:W4387156782 — "Design and Evaluation of a Retrieval-Augmented Generation Architecture for OWASP Security Data", 2023, positive): RAG pipeline grounds large language model outputs in authoritative security documentation
- `c003` (openalex:W4402670290 — "Benchmarking Retrieval-Augmented Generation for Medicine", 2024, mixed): Medical RAG exhibits log-linear scaling property and lost-in-the-middle effects
- `c008` (openalex:W4401198848 — "MedExpQA: Multilingual benchmarking of Large Language Models for Medical Question Answering", 2024, negative): LLMs achieve best results around 75% accuracy for English medical QA, with accuracy dropping 10 points for languages other than English
- `c009` (openalex:W4401198848 — "MedExpQA: Multilingual benchmarking of Large Language Models for Medical Question Answering", 2024, negative): State-of-the-art RAG methods demonstrate difficulty in obtaining and integrating readily available medical knowledge to positively impact Medical Question Answering results

### h001 — High top-k retrieval injects distractor passages that overwhelm the generator on domain-QA, flipping the effect of RAG from positive to negative.

**Mechanism.** Distractor evidence competes for the generator's attention; the signal-to-noise ratio of the retrieved set, not retrieval recall, governs downstream accuracy.

**Predictions:**
- Holding the retriever fixed, increasing top_k from 3 to 20 monotonically degrades domain-QA accuracy.
- Entailment or reranker filtering improves accuracy without changing various and related retrieval settings.

**Minimal test.** Sweep top_k in {1, 3, 5, 10, 20} on MIRAGE and related benchmarks with a fixed generator and various and related retrieval settings; measure accuracy plus evidence support.

**Scope.** task_type=factual, retrieval_noise=high

**Evidence gap.** Most RAG papers hold top_k constant or sweep it only on factual QA, rarely on multi-hop.

**Graph bridge.** RAG → domain-QA

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | cost | utility |
|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.96 | 0.93 | 0.00 | 0.98 |

### h003 — Partial test-set contamination in pretraining inflates closed-book baselines, which shrinks or reverses the apparent benefit of retrieval on standard benchmarks.

**Mechanism.** Web-crawled pretraining data overlaps with benchmark test questions, letting larger models answer without retrieval and collapsing the measured RAG gain.

**Predictions:**
- On a held-out, date-filtered version of MIRAGE and related benchmarks, RAG gains move away from up to 18% and reported effects.
- chain-of-thought prompting performance correlates with membership-inference scores on the evaluation set.

**Minimal test.** Build a post-cutoff evaluation set matching MIRAGE and related benchmarks; evaluate RAG and chain-of-thought prompting; compare delta accuracy to the original benchmark.

**Scope.** task_type=factual

**Evidence gap.** Few papers quantify pretraining overlap with MIRAGE and related benchmarks.

**Graph bridge.** RAG → domain-QA

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | cost | utility |
|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.97 | 0.92 | 0.00 | 0.98 |

### h005 — An unreported moderator variable drives the conflicting results around RAG on domain-QA.

**Mechanism.** A confound in data preprocessing, prompt formatting, or decoding parameters correlates with outcome direction and is not held constant across the claims.

**Predictions:**
- Holding prompt template and decoding fixed shrinks the between-claim variance by >50%.
- A covariate analysis reveals prompt/decoding parameters account for the sign flip.

**Minimal test.** Replay all claims on MIRAGE and related benchmarks in a common harness with identical prompts and decoding settings; recompute accuracy deltas.

**Evidence gap.** Prompt and decoding configurations are inconsistently reported across the claims.

**Graph bridge.** RAG → domain-QA

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | cost | utility |
|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.89 | 0.94 | 0.00 | 0.98 |

### h004 — Gains attributed to RAG are inflated when compared against weak baselines; the sign of the effect depends primarily on baseline choice.

**Mechanism.** Weak baselines (under-prompted closed-book LLMs, small models) leave more headroom, so any additional evidence appears to help even when retrieval is noisy.

**Predictions:**
- Replacing chain-of-thought prompting with a stronger matched baseline reduces the reported RAG gain.
- The correlation between baseline strength and RAG gain is negative across papers in this anomaly.

**Minimal test.** Re-run the positive claims on MIRAGE and related benchmarks against chain-of-thought prompting and a stronger matched baseline; report the baseline-conditioned delta in accuracy.

**Scope.** baseline_strength=low_vs_high

**Evidence gap.** Baseline strength is rarely controlled across RAG reports.

**Graph bridge.** RAG → domain-QA

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | cost | utility |
|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.96 | 0.93 | 0.00 | 0.98 |

## Evidence claims

- **c001** (openalex:W4402670290 — "Benchmarking Retrieval-Augmented Generation for Medicine", 2024, positive): MEDRAG improves the accuracy of six different LLMs by up to 18% over chain-of-thought prompting
- **c002** (openalex:W4402670290 — "Benchmarking Retrieval-Augmented Generation for Medicine", 2024, positive): Combination of various medical corpora and retrievers achieves the best performance
- **c003** (openalex:W4402670290 — "Benchmarking Retrieval-Augmented Generation for Medicine", 2024, mixed): Medical RAG exhibits log-linear scaling property and lost-in-the-middle effects
- **c004** (openalex:W4387156782 — "Design and Evaluation of a Retrieval-Augmented Generation Architecture for OWASP Security Data", 2023, positive): RAG system enables security-focused question answering with reduced risk of hallucinated responses
- **c005** (openalex:W4387156782 — "Design and Evaluation of a Retrieval-Augmented Generation Architecture for OWASP Security Data", 2023, positive): RAG pipeline grounds large language model outputs in authoritative security documentation
- **c008** (openalex:W4401198848 — "MedExpQA: Multilingual benchmarking of Large Language Models for Medical Question Answering", 2024, negative): LLMs achieve best results around 75% accuracy for English medical QA, with accuracy dropping 10 points for languages other than English
- **c009** (openalex:W4401198848 — "MedExpQA: Multilingual benchmarking of Large Language Models for Medical Question Answering", 2024, negative): State-of-the-art RAG methods demonstrate difficulty in obtaining and integrating readily available medical knowledge to positively impact Medical Question Answering results
