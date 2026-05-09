# Selected Hypotheses

Selected **10** hypotheses across **7** anomalies.

Exploratory report: no synthesized insights were supported yet, so the sections below show claim-level evidence and candidate explanations rather than a settled takeaway.

## Conflict Hypotheses

### Anomaly a003 — impact_conflict

**Central question:** Why do high-impact papers disagree about multimodal on multi-step-reasoning?

**Shared entities:** method=multimodal, task=multi-step-reasoning

**Evidence claims:**
- `arxiv:2604.03888v1#c03` (arxiv:2604.03888v1 — "PolySwarm: A Multi-Agent Large Language Model Framework for Prediction Market Trading and Latency Arbitrage", 2026, positive): GPT-4 improved reasoning and instruction-following while extending to multimodal inputs and longer context windows up to 128K tokens.
- `arxiv:2502.01081#c03` (arxiv:2502.01081 — "The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles", 2025, positive): Providing ground-truth visual perception improved performance across models.
- `arxiv:2503.11495v1#c01` (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, positive): Gemini-2-Flash has more balanced performance than GPT-4o on the AM score.
- `arxiv:2503.11495v1#c02` (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, positive): Gemini-2-Flash outperforms other models on long videos, achieving the highest mLGM and mAM.
- `arxiv:2503.08540v1#c02` (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, positive): Mellow achieves the highest BLEU-4 and SPICE among listed models on CLD-1.
- `arxiv:2503.08540v1#c03` (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, positive): Mellow achieves the highest BLEU-4 and SPICE among listed models on ACD-3.
- `arxiv:2503.08540v1#c04` (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, positive): QwenAC outperforms Mellow on Tier-2 in BLEU-4 for both CLD and ACD.
- `arxiv:2504.05599v2#c01` (arxiv:2504.05599v2 — "Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought", 2025, positive): Skywork R1V extends R1-series large language models to visual modalities via an efficient multimodal transfer method.
- `arxiv:2505.23764v2#c01` (arxiv:2505.23764v2 — "MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence", 2025, positive): Larger model scale yields only small accuracy gains within the Qwen2.5-VL and InternVL3 families on MMSI-Bench.
- `arxiv:2601.19673v1#c01` (arxiv:2601.19673v1 — "A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models", 2026, negative): The previous version Ultravox-v0.4.1-Llama3.1-8B generated 10% less relevant responses and stayed below 50% absolute accuracy across all five runs.
- `arxiv:2601.19673v1#c02` (arxiv:2601.19673v1 — "A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models", 2026, mixed): Only the cascaded system using Qwen3 achieved accuracy above 50%, and only under self-evaluation.
- `arxiv:2603.26742v1#c01` (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, negative): Accuracy drops by 9.8 to 25 percentage points when switching from English to an Indian language, and Dravidian languages are affected more than Indo-Aryan languages.
- `arxiv:2603.26742v1#c03` (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, mixed): GPT-4o has the highest average accuracy on Indian languages, while Gemma 3-27B has the best language robustness with only a 9.8-point drop from English.
- `arxiv:2603.26742v1#c04` (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, negative): For Aya-Vision-8B, Marathi accuracy is 8.5 points lower than Hindi accuracy despite the two languages sharing the same script.
- `arxiv:2503.11495v1#c03` (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, mixed): Qwen2.5-VL has the best spatial grounding but struggles with temporal localization in the what-when-where chain.
- `arxiv:2503.11495v1#c04` (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, mixed): InternVL-2.5 excels in temporal grounding but fails in spatial accuracy in the what-when-where chain.
- `arxiv:2503.11495v1#c05` (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, mixed): GPT-4o performs well on short videos but struggles with long sequences, suggesting weaker long-range dependency modelling.
- `arxiv:2503.08540v1#c01` (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, mixed): Tier-2 contains about 15% linguistics-related and stop words rather than audio-contrasting information, while Tier-1 focuses more on audio details.

### h097 — An unreported moderator variable drives the conflicting results around multimodal on multi-step-reasoning.

**Mechanism.** A confound in data preprocessing, prompt formatting, or decoding parameters correlates with outcome direction and is not held constant across the claims.

**Predictions:**
- Holding prompt template and decoding fixed shrinks the between-claim variance by >50%.
- A covariate analysis reveals prompt/decoding parameters account for the sign flip.

**Minimal test.** Replay all claims on CLD-1 and related benchmarks in a common harness with identical prompts and decoding settings; recompute Percent and related metrics deltas.

**Evidence gap.** Prompt and decoding configurations are inconsistently reported across the claims.

**Graph bridge.** multimodal → multi-step-reasoning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.89 | 1.00 | 1.00 | 0.56 | 0.00 | 0.96 |

### Anomaly a040 — metric_mismatch

**Central question:** Do different metrics explain why evaluation-method appears inconsistent on evaluation?

**Shared entities:** method=evaluation-method, task=evaluation, metrics=accuracy, cga and f-score, confidence, confidence / accuracy, f-score, f1, normalized helpfulness, percent, performance
**Varying settings:** metric

**Evidence claims:**
- `arxiv:2601.19673v1#c04` (arxiv:2601.19673v1 — "A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models", 2026, negative): GAMA had the lowest Yes/No accuracy at 21.55% in the evaluation.
- `arxiv:2602.04234v4#c05` (arxiv:2602.04234v4 — "On the Uncertainty of Large Language Model-Based Multi-Agent Systems", 2026, positive): The Entropy Judger consistently boosts accuracy across all configurations and tasks.
- `arxiv:2602.12889v1#c02` (arxiv:2602.12889v1 — "BaziQA-Benchmark: Evaluating Symbolic and Temporally Compositional Reasoning in Large Language Models", 2026, mixed): Differences in macro-average accuracy should be interpreted as descriptive ordering rather than statistically decisive separation.
- `arxiv:2411.16508#c02` (arxiv:2411.16508 — "All Languages Matter: Evaluating LMMs on Culturally Diverse 100 Languages", 2025, mixed): The evaluation uses GPT-4o as a judge to score responses for cultural category images.
- `arxiv:2501.06186v1#c04` (arxiv:2501.06186v1 — "LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs", 2025, positive): A comparison function using a secondary system prompt was developed to evaluate how well final answer predictions align with ground truth.
- `arxiv:2501.17399#c02` (arxiv:2501.17399 — "MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs", 2025, negative): All frontier models score below 50% accuracy on the benchmark despite near-perfect scores on existing multi-turn benchmarks.
- `arxiv:2501.17399#c03` (arxiv:2501.17399 — "MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs", 2025, negative): The top-performing Claude 3.5 Sonnet (June 2024) reaches only 41.4% average accuracy on the benchmark.
- `arxiv:2501.17399#c04` (arxiv:2501.17399 — "MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs", 2025, negative): Frontier LLMs such as GPT-4o have under 50% accuracy as judges, limiting confidence in their ability to evaluate other models on the benchmark.
- `arxiv:2502.08826v3#c03` (arxiv:2502.08826v3 — "Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation", 2025, positive): Recall@K is preferred over standard recall for retrieval-based tasks.
- `arxiv:2504.16074v2#c03` (arxiv:2504.16074v2 — "PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models", 2025, mixed): PHYBench is original and challenging for measuring LLM reasoning using physics problems.
- `arxiv:2602.02537v1#c01` (arxiv:2602.02537v1 — "WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models", 2026, positive): Gemini-3-pro achieves the highest F-score of 47.5% on WorldVQA.
- `arxiv:2602.02537v1#c02` (arxiv:2602.02537v1 — "WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models", 2026, mixed): GPT-5.1 has high Correct Given Attempted but low F-score, suggesting a conservative answering strategy.
- `arxiv:2602.02537v1#c03` (arxiv:2602.02537v1 — "WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models", 2026, negative): Gemini-3-pro displays binary confidence behavior, assigning 95% confidence in over 85% of cases regardless of accuracy.
- `arxiv:2602.02537v1#c04` (arxiv:2602.02537v1 — "WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models", 2026, negative): Most models are severely overconfident, concentrating predictions in the 90-100% confidence range.
- `arxiv:2603.05553v1#c01` (arxiv:2603.05553v1 — "EigenData: A Self-Evolving Multi-Agent Platform for Function-Calling Data Synthesis, Auditing, and Repair", 2026, positive): GPT-5.2 shows a much higher human pass rate than its original BFCL score, indicating benchmark errors penalized it.
- `arxiv:2603.05553v1#c02` (arxiv:2603.05553v1 — "EigenData: A Self-Evolving Multi-Agent Platform for Function-Calling Data Synthesis, Auditing, and Repair", 2026, negative): Requiring stricter evaluation criteria reduces average pass rate substantially.
- `arxiv:2603.05553v1#c03` (arxiv:2603.05553v1 — "EigenData: A Self-Evolving Multi-Agent Platform for Function-Calling Data Synthesis, Auditing, and Repair", 2026, mixed): GLM-4.6 scores highest on original BFCL but only ranks fourth in human evaluation, suggesting benchmark-specific overfitting.
- `arxiv:2602.00933v1#c02` (arxiv:2602.00933v1 — "MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP Servers", 2026, mixed): There is substantial variation in model capabilities, with coverage IQR spanning 28.8% to 61.2%.
- `arxiv:2602.00933v1#c06` (arxiv:2602.00933v1 — "MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP Servers", 2026, positive): Claude Opus 4.5 has the highest reported pass rate at 62.3%, while Gemini 2.5 Flash has 3.4%.
- `arxiv:2603.26742v1#c06` (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, negative): High extraction failure rates for Aya-Vision-8B and InternVL2.5-8B may cause their reported scores to underestimate true reasoning capability.
- `arxiv:2604.12379v1#c02` (arxiv:2604.12379v1 — "Beyond Output Correctness: Benchmarking and Evaluating Large Language Model Reasoning in Coding Tasks", 2026, negative): Existing reasoning evaluators can produce scores that misalign with ground-truth reasoning correctness on CodeRQ-Bench.
- `arxiv:2604.12379v1#c03` (arxiv:2604.12379v1 — "Beyond Output Correctness: Benchmarking and Evaluating Large Language Model Reasoning in Coding Tasks", 2026, negative): Agreement with self-generated references can reflect shared mistakes rather than true correctness, causing mostly missed errors.
- `arxiv:2501.17399#c05` (arxiv:2501.17399 — "MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs", 2025, positive): The authors' auto-evaluation with IR substantially improves alignment with human raters over the baseline across all challenge categories.
- `arxiv:2504.02807v1#c05` (arxiv:2504.02807v1 — "MegaMath: Pushing the Limits of Open Math Corpora", 2025, positive): -Web-Pro outperforms FineMath-3+ and FineMath-4+ by 4%.
- `arxiv:2504.16074v2#c01` (arxiv:2504.16074v2 — "PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models", 2025, positive): DeepSeek-V3, Claude 3.7 Sonnet, and GPT-4.1 achieve relatively strong results, with accuracies of 13.6%, 13.2%, and 12.9% respectively.
- `arxiv:2504.16074v2#c04` (arxiv:2504.16074v2 — "PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models", 2025, positive): Gemini 2.5 Pro has the highest reported S_EED and ACC values in the efficiency table.
- `arxiv:2505.23802#c03` (arxiv:2505.23802 — "MedHELM: Holistic Evaluation of Large Language Models for Medical Tasks", 2025, negative): Most LLM evaluations in medicine rely on closed-form exam-style question answering, with only 5% incorporating real EHR data.
- `arxiv:2505.23764v2#c04` (arxiv:2505.23764v2 — "MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence", 2025, negative): GPT-4o labels error types correctly for only 53% of selected samples, suggesting answer correctness alone is insufficient to assess reasoning quality.
- `arxiv:2503.10291v1#c04` (arxiv:2503.10291v1 — "VisualPRM: An Effective Process Reward Model for Multimodal Reasoning", 2025, negative): InternVL2.5-8B rarely identifies incorrect steps, with much lower F1 on negative than positive steps.
- `arxiv:2504.02807v1#c04` (arxiv:2504.02807v1 — "MegaMath: Pushing the Limits of Open Math Corpora", 2025, mixed): Evaluation on 20K in-distribution samples can mask true performance by yielding over 90% F1.
- `arxiv:2503.21460v1#c03` (arxiv:2503.21460v1 — "Large Language Model Agent: A Survey on Methodology, Applications and Challenges", 2025, positive): Dify supports continuous improvement of prompts, datasets, and models by monitoring and analyzing application logs and performance over time using production data and annotations.
- `arxiv:2504.15253v2#c03` (arxiv:2504.15253v2 — "Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators", 2025, mixed): The pairwise protocol's normalized helpfulness is compared against Llama-3.1 8B with judge prompt, the best reward model, and random reranking, averaged across generator models and datasets.

### h005 — Another mismatch is that some metrics measure confidence calibration while others measure answer correctness or rank quality, so a model can appear strong on accuracy/F-score yet poor on confidence-based evaluation.

**Mechanism.** Correctness metrics ignore the probability attached to predictions, but calibration metrics penalize overconfidence, underconfidence, and poorly separated confidence bins; a model with many right answers can still score badly if it assigns extreme confidence regardless of correctness.

**Predictions:**
- High-accuracy outputs with flat 95-100% confidence produce poor calibration scores.
- Re-scoring the same predictions without confidence leaves accuracy unchanged but collapses the confidence metric difference.

**Minimal test.** On a shared eval set, retain each model's raw answer plus its self-reported confidence or proxy confidence score for every item. Cross-score the exact same outputs twice: once with correctness metrics such as accuracy/F-score and once with calibration metrics such as confidence-accuracy gap, ECE, or Brier-style binning; then test whether models praised under correctness are penalized under calibration because confidence is weakly coupled to correctness.

**Scope.** method=evaluation-method, task=evaluation

**Evidence gap.** The current record contains separate claims about confidence behavior and correctness, but not a joint table linking the same instance-level predictions to both score families.

**Graph bridge.** confidence, confidence / accuracy, accuracy → evaluation-method

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.89 | 0.83 | 1.00 | 0.81 | 0.00 | 0.95 |

### Anomaly a006 — impact_conflict

**Central question:** Why do high-impact papers disagree about chain-of-thought on multi-step-reasoning?

**Shared entities:** method=chain-of-thought, task=multi-step-reasoning

**Evidence claims:**
- `arxiv:2501.06186v1#c01` (arxiv:2501.06186v1 — "LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs", 2025, positive): Curriculum Learning combined with Multi-Step CoT yields a 9.14% absolute gain over the base Llama-3.2-11B-Vision-Inst model.
- `arxiv:2501.06186v1#c02` (arxiv:2501.06186v1 — "LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs", 2025, positive): LlamaV-o1 achieves better final answer accuracy than GPT-4o-mini and LLava-CoT, with competitive step scores of 68.93%.
- `arxiv:2503.16419v4#c01` (arxiv:2503.16419v4 — "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models", 2025, positive): Pretraining recipes for reasoning-capable models often encourage extended reasoning steps to improve accuracy, making the overthinking challenge difficult to address.
- `arxiv:2503.16419v4#c03` (arxiv:2503.16419v4 — "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models", 2025, positive): Models tend to perform better with extended reasoning steps.
- `arxiv:2601.07780v1#c05` (arxiv:2601.07780v1 — "Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection", 2026, mixed): The increased computational cost of the method may be justified in critical applications requiring accuracy, robustness, and trustworthiness.
- `arxiv:2601.07780v1#c06` (arxiv:2601.07780v1 — "Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection", 2026, negative): Chain-of-Thought prompting still has limitations in consistency, accuracy, and self-correction, especially on complex or ethically sensitive tasks.
- `arxiv:2502.17419v6#c02` (arxiv:2502.17419v6 — "From System 1 to System 2: A Survey of Reasoning Large Language Models", 2025, negative): Foundational LLMs excel at fast decision-making but lack depth for complex reasoning.
- `arxiv:2505.23764v2#c05` (arxiv:2505.23764v2 — "MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence", 2025, mixed): Chain-of-thought prompting modestly improves performance only for GPT-4o and degrades performance on other models.

### h019 — The sign of chain-of-thought effects flips with dataset composition, especially whether benchmarks reward decomposable symbolic/visual subproblems versus brittle spatial or ethically ambiguous cases.

**Mechanism.** On datasets with explicit intermediate structure and low annotation ambiguity, CoT helps by externalizing subgoals; on spatially entangled or normatively ambiguous items, extra verbal steps can introduce confabulated relations and error propagation, making CoT look harmful or inconsistent.

**Predictions:**
- CoT gains are positive on decomposable items but negative on spatially entangled items
- Controlling item type reduces the cross-paper contradiction

**Minimal test.** Build a shared evaluation slice stratified into decomposable visual reasoning, multi-image spatial reasoning, and ethically sensitive reasoning items; run the same models with and without chain-of-thought on each stratum and compare the CoT-minus-direct effect within each stratum to see whether the reported disagreement disappears after matching dataset composition.

**Scope.** method=chain-of-thought, task=multi-step-reasoning

**Evidence gap.** The papers do not report a common item taxonomy or per-subset CoT deltas, so it is unclear whether one side evaluated mostly CoT-friendly items.

**Graph bridge.** chain-of-thought → multi-step-reasoning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.96 | 1.00 | 0.65 | 0.00 | 0.96 |

### h021 — Model scale and decoding strategy jointly moderate CoT effectiveness, so CoT helps larger or reasoning-tuned models under careful decoding but hurts smaller or differently decoded models.

**Mechanism.** Larger or reasoning-specialized models can use intermediate steps to search useful latent trajectories, whereas smaller models often generate low-quality chains that anchor wrong answers; aggressive sampling or long-generation decoding amplifies this divergence, making one paper observe gains and another degradations.

**Predictions:**
- CoT benefit increases with model capability under fixed decoding
- Greedy short decoding reduces CoT harm on weaker models

**Minimal test.** Run a model ladder spanning smaller and larger models, including GPT-4o where possible, on the same multi-step-reasoning set with direct prompting and CoT under identical greedy and sampled decoding settings; then re-estimate the CoT effect by model size and decoding regime to test whether standardizing these moderators removes the contradiction.

**Scope.** method=chain-of-thought, task=multi-step-reasoning

**Evidence gap.** The claims mix different models and likely different decoding defaults, but they do not provide a controlled model-by-decoding interaction analysis.

**Graph bridge.** arxiv:2505.23764v2#c05 → chain-of-thought

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.94 | 1.00 | 0.65 | 0.00 | 0.96 |

### h020 — The contradiction is driven by prompt format and reasoning-length control: papers using scaffolded or curriculum-style multi-step prompts measure benefits, while papers using generic free-form CoT trigger overthinking and degraded answers.

**Mechanism.** Structured prompts can constrain step order, encourage relevant evidence use, and reduce wandering; unconstrained prompts can inflate token count, increase hallucinated intermediate states, and bias models toward verbose but less accurate reasoning, flipping the effect sign.

**Predictions:**
- Scaffolded CoT outperforms free-form CoT under the same model and data
- Matching reasoning-length budget across prompt styles shrinks the discrepancy

**Minimal test.** Using one shared benchmark and one fixed model, compare direct answering, free-form 'think step by step', and a scaffolded/curriculum multi-step prompt while holding max tokens, stop criteria, and demonstrations fixed; if the free-form versus scaffolded difference explains the contradiction, the cross-paper sign mismatch should largely vanish when prompt format is standardized.

**Scope.** method=chain-of-thought, task=multi-step-reasoning

**Evidence gap.** Current claims do not isolate prompt template, step scaffolding, or token-budget settings, so improvements attributed to CoT may partly come from prompt engineering rather than reasoning per se.

**Graph bridge.** chain-of-thought → arxiv:2501.06186v1#c01

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.95 | 1.00 | 0.65 | 0.00 | 0.96 |

### Anomaly a021 — impact_conflict

**Central question:** Why do high-impact papers disagree about RAG on domain-QA?

**Shared entities:** method=RAG, task=domain-QA

**Evidence claims:**
- `arxiv:2502.04413v2#c01` (arxiv:2502.04413v2 — "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot", 2025, positive): MedRAG outperforms six baseline RAG methods on CPDD at L2 using the Llama-3.1-Instruct 8B backbone.
- `arxiv:2502.04413v2#c02` (arxiv:2502.04413v2 — "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot", 2025, positive): Adding KG-elicited reasoning improves MedRAG diagnostic accuracy across L1, L2, and L3 for all backbone LLMs compared with no KG-elicited reasoning.
- `arxiv:2502.04413v2#c03` (arxiv:2502.04413v2 — "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot", 2025, positive): KG-augmented knowledge mitigates noise and increases average accuracy versus random or no KG-elicited reasoning.
- `arxiv:2502.04413v2#c04` (arxiv:2502.04413v2 — "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot", 2025, positive): For Mixtral-8x7B, KG-elicited reasoning substantially improves L3 accuracy.
- `arxiv:2502.04413v2#c05` (arxiv:2502.04413v2 — "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot", 2025, positive): RAG with GPT-4o backbone outperforms other open-source and closed-source backbone choices.
- `arxiv:2504.12330v1#c01` (arxiv:2504.12330v1 — "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation", 2025, positive): HM-RAG achieves state-of-the-art average accuracy of 58.55% on CrisisMMD, improving over GPT-4o and a text-only Qwen2.5-72B variant despite using 7B parameters.
- `arxiv:2504.12330v1#c02` (arxiv:2504.12330v1 — "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation", 2025, positive): A 7B multimodal enhanced variant improves average accuracy by 2.3% over text-only Qwen2.5-72B, indicating better parameter efficiency.
- `arxiv:2504.12330v1#c03` (arxiv:2504.12330v1 — "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation", 2025, positive): HM-RAG reaches 93.73% average accuracy on ScienceQA, surpassing LLaMA-SciTune and GPT-4o.
- `arxiv:2504.12330v1#c04` (arxiv:2504.12330v1 — "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation", 2025, positive): On Task 1 of CrisisMMD, the method attains 72.06% accuracy and outperforms GPT-4o by 3.86%.
- `arxiv:2504.12330v1#c05` (arxiv:2504.12330v1 — "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation", 2025, negative): Disabling WA reduces average performance by 5.63%, with a larger 6.35% accuracy drop on grade 7-12 tasks.

### h117 — The reported advantage of RAG depends on retrieval recall quality, with high-recall retrieval making RAG strongly positive and lower-recall or noisier retrieval flipping the effect toward neutral or harmful.

**Mechanism.** Retrieval recall is a sign-flipping moderator: when retrieved evidence covers the answer-bearing documents, generation is grounded and improved; when recall is poor or retrieved items are noisy, RAG injects distractors and compounds errors, especially in domain settings with sparse terminology or multimodal evidence.

**Predictions:**
- Higher retrieval recall monotonically increases RAG benefit
- Matched-recall systems reduce cross-paper discrepancy

**Minimal test.** Implement a shared retriever and corpus construction protocol across the compared setups, tune top-k to equalize oracle answer coverage on CPDD and ScienceQA/CrisisMMD, and then rerun generation with identical readers; if holding recall fixed removes most of the disagreement in RAG gains, retrieval quality is the relevant moderator.

**Scope.** method=RAG, task=domain-QA

**Evidence gap.** Neither paper reports directly comparable retrieval recall, oracle coverage, or noise rates under a shared corpus/retriever setup.

**Graph bridge.** RAG → domain-QA

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.97 | 1.00 | 0.54 | 0.00 | 0.95 |

### h118 — The sign of the RAG effect is moderated by inference format and decoding choices, with structured reasoning prompts and conservative decoding amplifying gains while shorter or more stochastic prompting can erase them.

**Mechanism.** Prompt format and decoding strategy change how much the model uses retrieved evidence versus prior knowledge: explicit stepwise reasoning or agent coordination can force evidence use and improve answers, whereas terse prompts or high-temperature decoding can cause hallucinated reasoning, context neglect, or unstable aggregation that weakens RAG.

**Predictions:**
- Structured evidence-grounded prompts yield larger gains
- Low-temperature decoding reduces variance and disagreement

**Minimal test.** Take one common backbone and retriever, then run both papers' prompt templates or closest reproductions under a fixed decoding grid (e.g., greedy vs temperature sampling) on a shared benchmark slice; if the contradiction disappears when prompt and decoding are standardized, the sign flip is attributable to inference-format differences.

**Scope.** method=RAG, task=domain-QA

**Evidence gap.** The claims do not expose enough detail about prompt templates, evidence formatting, agent messages, or decoding hyperparameters for direct comparison.

**Graph bridge.** KG-elicited reasoning in MedRAG → HM-RAG

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.90 | 0.93 | 1.00 | 0.54 | 0.00 | 0.95 |

### Anomaly a014 — impact_conflict

**Central question:** Why do high-impact papers disagree about tool-use on agentic-reasoning?

**Shared entities:** method=tool-use, task=agentic-reasoning

**Evidence claims:**
- `arxiv:2603.24943v1#c04` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): Top models lead by maintaining a better precision-recall balance across diverse scenarios rather than excelling in only a few.
- `arxiv:2603.24943v1#c05` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): Stronger models improve from Easy to Hard, suggesting they leverage richer constraints and multi-tool opportunities in harder queries.
- `arxiv:2603.24943v1#c06` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): Qwen3-235B-A22B-Thinking achieves the best overall TF1 and EMR among listed models on FinMCP-Bench.
- `arxiv:2501.10132v1#c03` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, positive): GPT-4o significantly outperforms other models in the Hotel and Attraction domains, with success rates of 70% and 82%.
- `arxiv:2503.05659v2#c03` (arxiv:2503.05659v2 — "A Survey of Large Language Model Empowered Agents for Recommendation and Search: Towards Next-Generation Information Retrieval", 2025, positive): Reducing LLM cognitive load in understanding tool functionalities enhances tool-use efficiency and accuracy.
- `arxiv:2503.21460v1#c02` (arxiv:2503.21460v1 — "Large Language Model Agent: A Survey on Methodology, Applications and Challenges", 2025, positive): CRAFRT is a framework for tool creation and retrieval that builds specialized tool sets by collecting GPT-4 code solutions and abstracting them into code snippets.
- `arxiv:2603.24943v1#c01` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, negative): Tool Precision is lower on single-tool samples because models often over-predict multiple tools when only one is needed.
- `arxiv:2603.24943v1#c02` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Model size does not consistently correlate with performance, as Qwen3-4B-Thinking beats Qwen3-30B-A3B-Thinking on EMR while losing on TF1.
- `arxiv:2603.24943v1#c03` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Easy cases penalize over-calling, while harder cases reward recall and planning, leading to higher TF1 for models with balanced tool selection.
- `arxiv:2604.06185v1#c01` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, mixed): Gemini-2.0-Thinking exhibits a cautious failure profile, with higher refusal than wrong-tool selection.
- `arxiv:2604.06185v1#c02` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, mixed): Grok-4 exhibits an eager failure profile, minimizing refusals at the cost of more wrong-tool selections.
- `arxiv:2604.06185v1#c03` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, mixed): Reasoning models like o1 and gemini-2.0-thinking excel at inferring omitted information and intent in partial information tasks, while Claude-4-Sonnet leads on coreferential reference tasks.
- `arxiv:2604.06185v1#c04` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, mixed): No single model outperforms others across all aspects.
- `arxiv:2604.06185v1#c05` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, negative): Wrong Name / Missing Info and Redundant Call are the most prevalent errors across models.
- `arxiv:2604.06185v1#c06` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, mixed): The primary challenge in LLM tool-use has shifted from syntactic correctness to semantic and logical reasoning.
- `arxiv:2501.02506v4#c01` (arxiv:2501.02506v4 — "ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use", 2025, negative): LLaMA3.1-Instruct-8B has invocation errors on over 40% of queries, indicating challenges in documentation understanding.
- `arxiv:2501.10132v1#c01` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): Models with fewer than 10B parameters perform poorly on complex function calling; the best such model, GLM-4-9B, reaches only 8.4% success rate.
- `arxiv:2501.10132v1#c02` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, mixed): Claude-3.5-Sonnet, GPT-4o, and GLM-4-Long have comparable overall success rates among closed-source models.
- `arxiv:2501.10132v1#c04` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): Stop_early errors are especially severe for Claude-3.5-Sonnet and GPT-4o, with rates of 19.7% and 21.0%.
- `arxiv:2501.10132v1#c05` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): GLM-4-Long fails on this parameter more than 40% of the time.
- `arxiv:2501.10132v1#c06` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): Value_error is a major error source across models, especially for Qwen2.5-72B with a 78.8% value_error rate.
- `arxiv:2505.16700v2#c01` (arxiv:2505.16700v2 — "MCP-RADAR: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models", 2025, mixed): In the Filemanagement domain, GPT-4o had high tool-selection success but much lower end-task accuracy.

### h032 — The contradiction is moderated by prompt and tool-spec format: tool-use helps when tool documentation and call schemas reduce cognitive load, but hurts when prompts are verbose, underspecified, or require latent semantic inference about tool arguments.

**Mechanism.** Prompt packaging changes how much reasoning budget is spent on understanding tools versus solving the task. Concise schemas, curated tool descriptions, or executor abstractions can improve accuracy, whereas long or ambiguous documentation increases wrong-name, missing-info, value_error, and stop_early failures, flipping the measured effect.

**Predictions:**
- Schema-simplified prompts reduce value_error and wrong-name errors on the same tasks.
- With identical tools but rewritten concise descriptions, the positive and negative studies move toward similar performance.

**Minimal test.** Take one benchmark from each side of the conflict and run a cross-over ablation: original prompt/tool docs versus a normalized concise schema with identical tool inventory and outputs; if contradiction is due to prompt format, performance differences across papers should diminish under the normalized format.

**Scope.** method=tool-use, task=agentic-reasoning

**Evidence gap.** There is no shared release showing the exact prompt templates, tool descriptions, and argument schemas used across the conflicting papers.

**Graph bridge.** tool-use → agentic-reasoning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.93 | 1.00 | 0.64 | 0.00 | 0.95 |

### Anomaly a038 — metric_mismatch

**Central question:** Do different metrics explain why multimodal appears inconsistent on multi-step-reasoning?

**Shared entities:** method=multimodal, task=multi-step-reasoning, metrics=accuracy, am score, bleu_4, bleu_4 and spice, mlgm and mam, mlgm; mam, percent
**Varying settings:** metric

**Evidence claims:**
- `arxiv:2601.19673v1#c01` (arxiv:2601.19673v1 — "A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models", 2026, negative): The previous version Ultravox-v0.4.1-Llama3.1-8B generated 10% less relevant responses and stayed below 50% absolute accuracy across all five runs.
- `arxiv:2601.19673v1#c02` (arxiv:2601.19673v1 — "A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models", 2026, mixed): Only the cascaded system using Qwen3 achieved accuracy above 50%, and only under self-evaluation.
- `arxiv:2603.26742v1#c01` (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, negative): Accuracy drops by 9.8 to 25 percentage points when switching from English to an Indian language, and Dravidian languages are affected more than Indo-Aryan languages.
- `arxiv:2603.26742v1#c03` (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, mixed): GPT-4o has the highest average accuracy on Indian languages, while Gemma 3-27B has the best language robustness with only a 9.8-point drop from English.
- `arxiv:2603.26742v1#c04` (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, negative): For Aya-Vision-8B, Marathi accuracy is 8.5 points lower than Hindi accuracy despite the two languages sharing the same script.
- `arxiv:2503.11495v1#c04` (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, mixed): InternVL-2.5 excels in temporal grounding but fails in spatial accuracy in the what-when-where chain.
- `arxiv:2505.23764v2#c01` (arxiv:2505.23764v2 — "MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence", 2025, positive): Larger model scale yields only small accuracy gains within the Qwen2.5-VL and InternVL3 families on MMSI-Bench.
- `arxiv:2502.01081#c03` (arxiv:2502.01081 — "The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles", 2025, positive): Providing ground-truth visual perception improved performance across models.
- `arxiv:2503.08540v1#c01` (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, mixed): Tier-2 contains about 15% linguistics-related and stop words rather than audio-contrasting information, while Tier-1 focuses more on audio details.
- `arxiv:2503.11495v1#c01` (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, positive): Gemini-2-Flash has more balanced performance than GPT-4o on the AM score.
- `arxiv:2503.11495v1#c02` (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, positive): Gemini-2-Flash outperforms other models on long videos, achieving the highest mLGM and mAM.
- `arxiv:2503.11495v1#c05` (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, mixed): GPT-4o performs well on short videos but struggles with long sequences, suggesting weaker long-range dependency modelling.
- `arxiv:2503.08540v1#c02` (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, positive): Mellow achieves the highest BLEU-4 and SPICE among listed models on CLD-1.
- `arxiv:2503.08540v1#c03` (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, positive): Mellow achieves the highest BLEU-4 and SPICE among listed models on ACD-3.
- `arxiv:2503.08540v1#c04` (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, positive): QwenAC outperforms Mellow on Tier-2 in BLEU-4 for both CLD and ACD.

### h077 — BLEU-4 is measuring surface n-gram overlap, while SPICE and percent-style reasoning success are closer to semantic content selection, so outputs with generic or template-like wording can look good on one metric and not the other.

**Mechanism.** A model can match reference phrasing or frequent Tier-2 lexical patterns and gain BLEU-4 without expressing the right comparative evidence, while semantically grounded metrics such as SPICE or direct success percentages depend more on whether the key entities, relations, and contrasts are actually conveyed.

**Predictions:**
- Tier-2 outputs with more stopwords and boilerplate will retain BLEU-4 better than SPICE.
- Semantically correct paraphrases will gain SPICE or task percent while losing BLEU-4.

**Minimal test.** On a shared comparative-reasoning evaluation set such as CLD/ACD, save the same generated explanations from each model. Cross-score every output with BLEU-4 and SPICE, and if available also mark binary task success percent from the same outputs. To generate a paired set if references are sparse, create two controlled variants of each original output: a paraphrase preserving meaning and a lexicalized template preserving common n-grams; then rescore both metrics on those paired outputs.

**Scope.** method=multimodal, task=multi-step-reasoning

**Evidence gap.** Existing claims report leaderboard-style metric outcomes but do not isolate whether lexical overlap versus semantic proposition matching causes the reversals on the same answers.

**Graph bridge.** bleu_4, bleu_4 and spice, percent → comparative reasoning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.92 | 0.90 | 1.00 | 0.58 | 0.00 | 0.95 |

### Anomaly a046 — metric_mismatch

**Central question:** Do different metrics explain why tool-use appears inconsistent on agentic-reasoning?

**Shared entities:** method=tool-use, task=agentic-reasoning, metrics=accuracy, dtsr; acc, emr, tf1, failure rate, overall success rates, percent, refusal rate; wrong name error, stop_early rates, success rate, success rates, tf1, tf1; emr, tool precision (tp), value_error rate
**Varying settings:** metric

**Evidence claims:**
- `arxiv:2603.24943v1#c01` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, negative): Tool Precision is lower on single-tool samples because models often over-predict multiple tools when only one is needed.
- `arxiv:2603.24943v1#c02` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Model size does not consistently correlate with performance, as Qwen3-4B-Thinking beats Qwen3-30B-A3B-Thinking on EMR while losing on TF1.
- `arxiv:2603.24943v1#c03` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Easy cases penalize over-calling, while harder cases reward recall and planning, leading to higher TF1 for models with balanced tool selection.
- `arxiv:2603.24943v1#c06` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): Qwen3-235B-A22B-Thinking achieves the best overall TF1 and EMR among listed models on FinMCP-Bench.
- `arxiv:2604.06185v1#c01` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, mixed): Gemini-2.0-Thinking exhibits a cautious failure profile, with higher refusal than wrong-tool selection.
- `arxiv:2604.06185v1#c02` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, mixed): Grok-4 exhibits an eager failure profile, minimizing refusals at the cost of more wrong-tool selections.
- `arxiv:2604.06185v1#c05` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, negative): Wrong Name / Missing Info and Redundant Call are the most prevalent errors across models.
- `arxiv:2604.06185v1#c06` (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, mixed): The primary challenge in LLM tool-use has shifted from syntactic correctness to semantic and logical reasoning.
- `arxiv:2501.02506v4#c01` (arxiv:2501.02506v4 — "ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use", 2025, negative): LLaMA3.1-Instruct-8B has invocation errors on over 40% of queries, indicating challenges in documentation understanding.
- `arxiv:2501.10132v1#c01` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): Models with fewer than 10B parameters perform poorly on complex function calling; the best such model, GLM-4-9B, reaches only 8.4% success rate.
- `arxiv:2501.10132v1#c02` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, mixed): Claude-3.5-Sonnet, GPT-4o, and GLM-4-Long have comparable overall success rates among closed-source models.
- `arxiv:2501.10132v1#c03` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, positive): GPT-4o significantly outperforms other models in the Hotel and Attraction domains, with success rates of 70% and 82%.
- `arxiv:2501.10132v1#c04` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): Stop_early errors are especially severe for Claude-3.5-Sonnet and GPT-4o, with rates of 19.7% and 21.0%.
- `arxiv:2501.10132v1#c05` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): GLM-4-Long fails on this parameter more than 40% of the time.
- `arxiv:2501.10132v1#c06` (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): Value_error is a major error source across models, especially for Qwen2.5-72B with a 78.8% value_error rate.
- `arxiv:2503.05659v2#c03` (arxiv:2503.05659v2 — "A Survey of Large Language Model Empowered Agents for Recommendation and Search: Towards Next-Generation Information Retrieval", 2025, positive): Reducing LLM cognitive load in understanding tool functionalities enhances tool-use efficiency and accuracy.
- `arxiv:2505.16700v2#c01` (arxiv:2505.16700v2 — "MCP-RADAR: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models", 2025, mixed): In the Filemanagement domain, GPT-4o had high tool-selection success but much lower end-task accuracy.

### h083 — Exact-match completion metrics and token-overlap tool metrics measure different granularity: one requires the whole multi-step plan to be exactly right, while the other gives partial credit for overlap in tool sets or calls.

**Mechanism.** EMR/overall success act like all-or-nothing sequence metrics, whereas TF1 and related overlap scores reward partial recovery of needed tools even with extra or missing calls. This definitional gap can reverse rankings, especially on multi-tool and multi-turn problems where partial plans are common.

**Predictions:**
- Models with extra redundant calls will lose EMR more sharply than TF1 on the same items.
- Rank correlation between TF1 and EMR will drop as tasks require more tools or turns.

**Minimal test.** On a shared eval split containing single-tool, multi-tool, and multi-turn items, save the exact same model outputs including ordered tool traces. Re-score each output twice: once with exact-match completion (EMR or success) and once with overlap-based scoring (TF1 over tool calls). Compare per-example metric gaps and model ranking changes across difficulty buckets defined by number of required calls/turns.

**Scope.** method=tool-use, task=agentic-reasoning

**Evidence gap.** Missing joint per-instance annotations showing when outputs are partially correct enough for TF1 but not exact enough for EMR/success.

**Graph bridge.** EMR, TF1 → agentic-reasoning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.87 | 0.92 | 1.00 | 0.58 | 0.00 | 0.95 |

## Evidence claims

- **arxiv:2501.02506v4#c01** (arxiv:2501.02506v4 — "ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use", 2025, negative): LLaMA3.1-Instruct-8B has invocation errors on over 40% of queries, indicating challenges in documentation understanding.
- **arxiv:2501.06186v1#c01** (arxiv:2501.06186v1 — "LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs", 2025, positive): Curriculum Learning combined with Multi-Step CoT yields a 9.14% absolute gain over the base Llama-3.2-11B-Vision-Inst model.
- **arxiv:2501.06186v1#c02** (arxiv:2501.06186v1 — "LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs", 2025, positive): LlamaV-o1 achieves better final answer accuracy than GPT-4o-mini and LLava-CoT, with competitive step scores of 68.93%.
- **arxiv:2501.10132v1#c02** (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, mixed): Claude-3.5-Sonnet, GPT-4o, and GLM-4-Long have comparable overall success rates among closed-source models.
- **arxiv:2501.10132v1#c04** (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): Stop_early errors are especially severe for Claude-3.5-Sonnet and GPT-4o, with rates of 19.7% and 21.0%.
- **arxiv:2501.10132v1#c06** (arxiv:2501.10132v1 — "ComplexFuncBench: Exploring Multi-Step and Constrained Function Calling under Long-Context Scenario", 2025, negative): Value_error is a major error source across models, especially for Qwen2.5-72B with a 78.8% value_error rate.
- **arxiv:2502.01081#c03** (arxiv:2502.01081 — "The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles", 2025, positive): Providing ground-truth visual perception improved performance across models.
- **arxiv:2502.04413v2#c01** (arxiv:2502.04413v2 — "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot", 2025, positive): MedRAG outperforms six baseline RAG methods on CPDD at L2 using the Llama-3.1-Instruct 8B backbone.
- **arxiv:2502.04413v2#c02** (arxiv:2502.04413v2 — "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot", 2025, positive): Adding KG-elicited reasoning improves MedRAG diagnostic accuracy across L1, L2, and L3 for all backbone LLMs compared with no KG-elicited reasoning.
- **arxiv:2502.04413v2#c03** (arxiv:2502.04413v2 — "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot", 2025, positive): KG-augmented knowledge mitigates noise and increases average accuracy versus random or no KG-elicited reasoning.
- **arxiv:2502.04413v2#c04** (arxiv:2502.04413v2 — "MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot", 2025, positive): For Mixtral-8x7B, KG-elicited reasoning substantially improves L3 accuracy.
- **arxiv:2502.17419v6#c02** (arxiv:2502.17419v6 — "From System 1 to System 2: A Survey of Reasoning Large Language Models", 2025, negative): Foundational LLMs excel at fast decision-making but lack depth for complex reasoning.
- **arxiv:2503.05659v2#c03** (arxiv:2503.05659v2 — "A Survey of Large Language Model Empowered Agents for Recommendation and Search: Towards Next-Generation Information Retrieval", 2025, positive): Reducing LLM cognitive load in understanding tool functionalities enhances tool-use efficiency and accuracy.
- **arxiv:2503.08540v1#c01** (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, mixed): Tier-2 contains about 15% linguistics-related and stop words rather than audio-contrasting information, while Tier-1 focuses more on audio details.
- **arxiv:2503.08540v1#c02** (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, positive): Mellow achieves the highest BLEU-4 and SPICE among listed models on CLD-1.
- **arxiv:2503.08540v1#c03** (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, positive): Mellow achieves the highest BLEU-4 and SPICE among listed models on ACD-3.
- **arxiv:2503.08540v1#c04** (arxiv:2503.08540v1 — "Mellow: a small audio language model for reasoning", 2025, positive): QwenAC outperforms Mellow on Tier-2 in BLEU-4 for both CLD and ACD.
- **arxiv:2503.11495v1#c01** (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, positive): Gemini-2-Flash has more balanced performance than GPT-4o on the AM score.
- **arxiv:2503.11495v1#c02** (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, positive): Gemini-2-Flash outperforms other models on long videos, achieving the highest mLGM and mAM.
- **arxiv:2503.11495v1#c03** (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, mixed): Qwen2.5-VL has the best spatial grounding but struggles with temporal localization in the what-when-where chain.
- **arxiv:2503.11495v1#c04** (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, mixed): InternVL-2.5 excels in temporal grounding but fails in spatial accuracy in the what-when-where chain.
- **arxiv:2503.11495v1#c05** (arxiv:2503.11495v1 — "V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning", 2025, mixed): GPT-4o performs well on short videos but struggles with long sequences, suggesting weaker long-range dependency modelling.
- **arxiv:2503.16419v4#c01** (arxiv:2503.16419v4 — "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models", 2025, positive): Pretraining recipes for reasoning-capable models often encourage extended reasoning steps to improve accuracy, making the overthinking challenge difficult to address.
- **arxiv:2503.16419v4#c03** (arxiv:2503.16419v4 — "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models", 2025, positive): Models tend to perform better with extended reasoning steps.
- **arxiv:2503.21460v1#c02** (arxiv:2503.21460v1 — "Large Language Model Agent: A Survey on Methodology, Applications and Challenges", 2025, positive): CRAFRT is a framework for tool creation and retrieval that builds specialized tool sets by collecting GPT-4 code solutions and abstracting them into code snippets.
- **arxiv:2504.05599v2#c01** (arxiv:2504.05599v2 — "Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought", 2025, positive): Skywork R1V extends R1-series large language models to visual modalities via an efficient multimodal transfer method.
- **arxiv:2504.12330v1#c01** (arxiv:2504.12330v1 — "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation", 2025, positive): HM-RAG achieves state-of-the-art average accuracy of 58.55% on CrisisMMD, improving over GPT-4o and a text-only Qwen2.5-72B variant despite using 7B parameters.
- **arxiv:2504.12330v1#c03** (arxiv:2504.12330v1 — "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation", 2025, positive): HM-RAG reaches 93.73% average accuracy on ScienceQA, surpassing LLaMA-SciTune and GPT-4o.
- **arxiv:2504.12330v1#c04** (arxiv:2504.12330v1 — "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation", 2025, positive): On Task 1 of CrisisMMD, the method attains 72.06% accuracy and outperforms GPT-4o by 3.86%.
- **arxiv:2504.12330v1#c05** (arxiv:2504.12330v1 — "HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation", 2025, negative): Disabling WA reduces average performance by 5.63%, with a larger 6.35% accuracy drop on grade 7-12 tasks.
- **arxiv:2505.23764v2#c01** (arxiv:2505.23764v2 — "MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence", 2025, positive): Larger model scale yields only small accuracy gains within the Qwen2.5-VL and InternVL3 families on MMSI-Bench.
- **arxiv:2505.23764v2#c05** (arxiv:2505.23764v2 — "MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence", 2025, mixed): Chain-of-thought prompting modestly improves performance only for GPT-4o and degrades performance on other models.
- **arxiv:2601.07780v1#c05** (arxiv:2601.07780v1 — "Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection", 2026, mixed): The increased computational cost of the method may be justified in critical applications requiring accuracy, robustness, and trustworthiness.
- **arxiv:2601.07780v1#c06** (arxiv:2601.07780v1 — "Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection", 2026, negative): Chain-of-Thought prompting still has limitations in consistency, accuracy, and self-correction, especially on complex or ethically sensitive tasks.
- **arxiv:2601.19673v1#c01** (arxiv:2601.19673v1 — "A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models", 2026, negative): The previous version Ultravox-v0.4.1-Llama3.1-8B generated 10% less relevant responses and stayed below 50% absolute accuracy across all five runs.
- **arxiv:2601.19673v1#c02** (arxiv:2601.19673v1 — "A Benchmark for Audio Reasoning Capabilities of Multimodal Large Language Models", 2026, mixed): Only the cascaded system using Qwen3 achieved accuracy above 50%, and only under self-evaluation.
- **arxiv:2602.02537v1#c01** (arxiv:2602.02537v1 — "WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models", 2026, positive): Gemini-3-pro achieves the highest F-score of 47.5% on WorldVQA.
- **arxiv:2602.02537v1#c03** (arxiv:2602.02537v1 — "WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models", 2026, negative): Gemini-3-pro displays binary confidence behavior, assigning 95% confidence in over 85% of cases regardless of accuracy.
- **arxiv:2602.02537v1#c04** (arxiv:2602.02537v1 — "WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models", 2026, negative): Most models are severely overconfident, concentrating predictions in the 90-100% confidence range.
- **arxiv:2602.04234v4#c05** (arxiv:2602.04234v4 — "On the Uncertainty of Large Language Model-Based Multi-Agent Systems", 2026, positive): The Entropy Judger consistently boosts accuracy across all configurations and tasks.
- **arxiv:2603.24943v1#c02** (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Model size does not consistently correlate with performance, as Qwen3-4B-Thinking beats Qwen3-30B-A3B-Thinking on EMR while losing on TF1.
- **arxiv:2603.24943v1#c03** (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Easy cases penalize over-calling, while harder cases reward recall and planning, leading to higher TF1 for models with balanced tool selection.
- **arxiv:2603.24943v1#c06** (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): Qwen3-235B-A22B-Thinking achieves the best overall TF1 and EMR among listed models on FinMCP-Bench.
- **arxiv:2603.26742v1#c01** (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, negative): Accuracy drops by 9.8 to 25 percentage points when switching from English to an Indian language, and Dravidian languages are affected more than Indo-Aryan languages.
- **arxiv:2603.26742v1#c03** (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, mixed): GPT-4o has the highest average accuracy on Indian languages, while Gemma 3-27B has the best language robustness with only a 9.8-point drop from English.
- **arxiv:2603.26742v1#c04** (arxiv:2603.26742v1 — "Do Multilingual VLMs Reason Equally? A Cross-Lingual Visual Reasoning Audit for Indian Languages", 2026, negative): For Aya-Vision-8B, Marathi accuracy is 8.5 points lower than Hindi accuracy despite the two languages sharing the same script.
- **arxiv:2604.03888v1#c03** (arxiv:2604.03888v1 — "PolySwarm: A Multi-Agent Large Language Model Framework for Prediction Market Trading and Latency Arbitrage", 2026, positive): GPT-4 improved reasoning and instruction-following while extending to multimodal inputs and longer context windows up to 128K tokens.
- **arxiv:2604.06185v1#c06** (arxiv:2604.06185v1 — "Benchmarking LLM Tool-Use in the Wild", 2026, mixed): The primary challenge in LLM tool-use has shifted from syntactic correctness to semantic and logical reasoning.
