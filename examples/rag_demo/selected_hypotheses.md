# Selected Hypotheses

Selected **3** hypotheses across **1** anomalies.

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

### h003 — Performance discrepancies reflect benchmark-specific difficulty and language coverage, where MIRAGE (c001) enables high gains due to English-only questions, while MedExpQA (c008, c009) introduces multilingual complexity that triggers lost-in-the-middle effects (c003) and 10-point accuracy drops in non-English settings.

**Mechanism.** Dataset difficulty and linguistic scope

**Predictions:**
- MEDRAG achieves <80% accuracy on MedExpQA despite 18% gains on MIRAGE
- Non-English queries show 10+ point larger accuracy gaps than English within the same RAG system

**Minimal test.** Evaluate MEDRAG on MedExpQA stratified by English vs non-English queries and compare against MIRAGE baseline

**Scope.** method=MEDRAG, task=medical QA

**Evidence gap.** Difficulty metrics and language distribution of the MIRAGE dataset

**Graph bridge.** benchmark_language_scope → rag_accuracy_ceiling

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | cost | utility |
|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.93 | 0.83 | 0.00 | 0.96 |

### h001 — RAG effectiveness in domain-QA is moderated by domain knowledge structure, where authoritative stable documentation enables consistent gains in security (c004, c005) while fragmented medical knowledge causes integration failures (c009) and accuracy ceilings (c008) despite high performance with optimized ensembles (c001, c002).

**Mechanism.** Domain authority and knowledge stability

**Predictions:**
- RAG accuracy on security QA exceeds medical QA by >20% using identical retrievers
- Medical RAG shows higher variance in knowledge integration success than security RAG

**Minimal test.** Apply identical RAG pipeline to OWASP docs and MedExpQA measuring accuracy and retrieval precision

**Scope.** retriever_configuration=fixed, task_type=domain-QA

**Evidence gap.** Direct measurement of corpus noise and authority differences between medical and security domains

**Graph bridge.** domain_knowledge_structure → rag_integration_success

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | cost | utility |
|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.90 | 0.84 | 0.00 | 0.96 |

### h002 — The inconsistency stems from retrieval configuration, where multi-corpus ensemble methods (c002) overcome lost-in-the-middle effects (c003) to achieve 18% gains (c001), while standard single-retriever setups on MedExpQA hit accuracy ceilings (c008) and fail to integrate knowledge (c009).

**Mechanism.** Ensemble retrieval mitigates context window limitations

**Predictions:**
- Single-retriever setups exhibit stronger lost-in-the-middle effects than multi-retriever configurations
- Ensemble methods reduce knowledge integration difficulty on MedExpQA by >15%

**Minimal test.** Ablate MEDRAG from multi-corpus to single-corpus retrieval and measure lost-in-the-middle effect magnitude and accuracy on MedExpQA

**Scope.** domain=medical, language=English

**Evidence gap.** Whether c009 used single or multi-retriever configurations

**Graph bridge.** retriever_ensemble_size → context_window_utilization

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | cost | utility |
|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.92 | 0.83 | 0.00 | 0.96 |

## Evidence claims

- **c001** (openalex:W4402670290 — "Benchmarking Retrieval-Augmented Generation for Medicine", 2024, positive): MEDRAG improves the accuracy of six different LLMs by up to 18% over chain-of-thought prompting
- **c002** (openalex:W4402670290 — "Benchmarking Retrieval-Augmented Generation for Medicine", 2024, positive): Combination of various medical corpora and retrievers achieves the best performance
- **c003** (openalex:W4402670290 — "Benchmarking Retrieval-Augmented Generation for Medicine", 2024, mixed): Medical RAG exhibits log-linear scaling property and lost-in-the-middle effects
- **c004** (openalex:W4387156782 — "Design and Evaluation of a Retrieval-Augmented Generation Architecture for OWASP Security Data", 2023, positive): RAG system enables security-focused question answering with reduced risk of hallucinated responses
- **c005** (openalex:W4387156782 — "Design and Evaluation of a Retrieval-Augmented Generation Architecture for OWASP Security Data", 2023, positive): RAG pipeline grounds large language model outputs in authoritative security documentation
- **c008** (openalex:W4401198848 — "MedExpQA: Multilingual benchmarking of Large Language Models for Medical Question Answering", 2024, negative): LLMs achieve best results around 75% accuracy for English medical QA, with accuracy dropping 10 points for languages other than English
- **c009** (openalex:W4401198848 — "MedExpQA: Multilingual benchmarking of Large Language Models for Medical Question Answering", 2024, negative): State-of-the-art RAG methods demonstrate difficulty in obtaining and integrating readily available medical knowledge to positively impact Medical Question Answering results
