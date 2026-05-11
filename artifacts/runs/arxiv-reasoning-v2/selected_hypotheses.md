# Selected Hypotheses

Selected **10** hypotheses across **9** anomalies.

Exploratory report: no synthesized insights were supported yet, so the sections below show claim-level evidence and candidate explanations rather than a settled takeaway.

## Conflict Hypotheses

### Anomaly a011 — benchmark_inconsistency

**Central question:** When does evaluation-method help on multi-step-reasoning, and when does it fail?

**Shared entities:** method=evaluation-method, task=multi-step-reasoning

**Evidence claims:**
- `arxiv:2311.17667v2#c04` (arxiv:2311.17667v2 — "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models", 2023, positive): In zero-shot settings, GPT-4 and GPT-3.5 outperform all open-source models.
- `arxiv:2403.02615v2#c02` (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, positive): GPT-4 achieves nearly three times the accuracy of Llama2 7B.
- `arxiv:2403.02615v2#c05` (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, positive): Mistral 7B, ChatGPT, and GPT-4 show improved accuracy.
- `arxiv:2403.13315v3#c01` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, positive): Human participants achieved an average baseline score of 91.6% on sampled PuzzleVQA puzzles.
- `arxiv:2403.13315v3#c02` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, positive): GPT-4V scored 47.5% on the sampled puzzle set and was the highest-performing model there.
- `arxiv:2407.11963v3#c04` (arxiv:2407.11963v3 — "NeedleBench: Evaluating LLM Retrieval and Reasoning Across Varying Information Densities", 2024, positive): Qwen-2.5-72B achieves the best result on the Multi-Needle Reasoning task, but remains below 50%.
- `arxiv:2411.01307v1#c03` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): MLLMs show zero-shot multimodal analogical reasoning capability on MBARD, with ChatGPT-4 achieving the best accuracy of 68.0%.
- `arxiv:2407.01046v2#c02` (arxiv:2407.01046v2 — "FRoG: Evaluating Fuzzy Reasoning of Generalized Quantifiers in Large Language Models", 2024, negative): GPT-4-turbo achieves below 50% accuracy on FRoG across masking settings.
- `arxiv:2403.03864v3#c05` (arxiv:2403.03864v3 — "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning", 2024, mixed): The overall random baseline is 26.4%.
- `arxiv:2404.05221v2#c06` (arxiv:2404.05221v2 — "LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models", 2024, negative): Manual evaluation found up to 39% false-positive reasoning chains for Llama-2-70B on StrategyQA.
- `arxiv:2311.17667v2#c01` (arxiv:2311.17667v2 — "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models", 2023, negative): GPT-4 achieves only 66.4% accuracy on implicit temporal reasoning, indicating difficulty with implicit temporal relationships.
- `arxiv:2311.17667v2#c02` (arxiv:2311.17667v2 — "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models", 2023, negative): Duration-conversion performance decreases compared to other atomic tasks.
- `arxiv:2311.17667v2#c05` (arxiv:2311.17667v2 — "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models", 2023, negative): All models except GPT-4 perform unsatisfactorily on symbolic temporal reasoning.
- `arxiv:2410.17558v2#c01` (arxiv:2410.17558v2 — "CLR-Bench: Evaluating Large Language Models in College-level Reasoning", 2024, negative): Qwen2.5-72b-Instruct scores much lower on Q→AR than on Q→A.
- `arxiv:2410.17558v2#c02` (arxiv:2410.17558v2 — "CLR-Bench: Evaluating Large Language Models in College-level Reasoning", 2024, mixed): qwen2.5-32b-instruct trails qwen2.5-72b-instruct on Q→A but outperforms it on Q→AR.
- `arxiv:2410.17558v2#c03` (arxiv:2410.17558v2 — "CLR-Bench: Evaluating Large Language Models in College-level Reasoning", 2024, negative): Across all three models, performance drops by over 30% on average from Q→A to Q→AR.
- `arxiv:2410.17558v2#c04` (arxiv:2410.17558v2 — "CLR-Bench: Evaluating Large Language Models in College-level Reasoning", 2024, negative): GPT-4-turbo performs strongly on Q→A but drops substantially on Q→AR.
- `arxiv:2306.08952v2#c01` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): Most models perform significantly worse on L3 than L2 questions in the ReasonQA setting, except ChatGPT.
- `arxiv:2306.08952v2#c02` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): F1 scores before 1900 and after 2020 are significantly worse than other time periods.
- `arxiv:2306.08952v2#c03` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): Performance on intra-year questions is significantly worse than on inter-year questions.
- `arxiv:2306.08952v2#c04` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): Performance in 2020-2040 is significantly worse than in other time periods for the first two LMs, indicating failure to generalize to the future.
- `arxiv:2306.08952v2#c05` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): FLAN-T5-L has significantly lower EM scores than T5-L-NQ.
- `arxiv:2403.02615v2#c04` (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, negative): LLaMA-7B and GPT-4 disagree on over 60% of questions.
- `arxiv:2403.13315v3#c03` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, mixed): On single-concept abstract patterns in PuzzleVQA, model accuracy ranged from random-baseline level to 67.5%, with GPT-4V highest on average.
- `arxiv:2403.13315v3#c04` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, mixed): On dual-concept abstract patterns in PuzzleVQA, model accuracy ranged from 26.4 to 45.5 on average, with GPT-4V highest on average.
- `arxiv:2411.12580v2#c04` (arxiv:2411.12580v2 — "Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models", 2024, negative): When reasoning, the model relies less on each individual document and shows less volatile total influence than when answering factual questions.
- `arxiv:2411.12580v2#c06` (arxiv:2411.12580v2 — "Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models", 2024, negative): Models rely less on individual documents for generating reasoning traces than for answering factual questions.
- `arxiv:2407.11963v3#c01` (arxiv:2407.11963v3 — "NeedleBench: Evaluating LLM Retrieval and Reasoning Across Varying Information Densities", 2024, mixed): Reasoning scores increase with model scale within some model series, but not universally.
- `arxiv:2411.01307v1#c06` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, mixed): An ablation variant removes the textual description of the relation from Explainer.
- `arxiv:2411.14465v1#c04` (arxiv:2411.14465v1 — "Testing Uncertainty of Large Language Models for Physics Knowledge and Reasoning", 2024, negative): For single-step and multi-step reasoning questions, GPT-3.5-turbo produces few correct replies despite high diversity.

### h290 — The sign flips because evaluation protocol differs: papers using final-answer accuracy report evaluation-method as beneficial, while papers using stricter protocols such as chain-faithfulness checks, subgroup breakdowns, or harder success criteria report failure.

**Mechanism.** A lenient metric can credit correct end answers produced via flawed reasoning, whereas manual chain inspection or subgroup-specific scoring reveals hidden errors; changing the protocol therefore changes whether the same system is counted as successful or unsuccessful.

**Predictions:**
- Manual faithfulness auditing lowers scores relative to final-answer accuracy
- Using a common scoring rule narrows inter-paper disagreement

**Minimal test.** Re-score a common set of model outputs from the cited tasks with one unified protocol: exact final-answer accuracy plus blinded manual verification of reasoning validity and subgroup reporting; test whether positive and negative claims converge when the evaluation rule is held fixed.

**Scope.** method=evaluation-method, task=multi-step-reasoning

**Evidence gap.** There is no shared evaluation sheet linking final-answer correctness to reasoning-faithfulness across the cited benchmarks.

**Graph bridge.** arxiv:2404.05221v2#c06 → arxiv:2403.02615v2#c05

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.91 | 0.94 | 0.00 | 0.00 | 0.00 | 0.82 |

### Anomaly a004 — benchmark_inconsistency

**Central question:** When does GPT-4 help on evaluation, and when does it fail?

**Shared entities:** method=GPT-4, task=evaluation

**Evidence claims:**
- `arxiv:2311.08562v3#c02` (arxiv:2311.08562v3 — "MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration", 2023, positive): GPT-4 scored 90% in Judgment and 75.0% in Deception, indicating superiority in those scenarios.
- `arxiv:2407.08029v1#c01` (arxiv:2407.08029v1 — "A Critical Review of Causal Reasoning Benchmarks for Large Language Models", 2024, negative): GPT-3.5 and GPT-4 memorized the Tübingen cause-effect dataset, or a large portion of it.
- `arxiv:2407.14507v3#c01` (arxiv:2407.14507v3 — "Internal Consistency and Self-Feedback in Large Language Models: A Survey", 2024, negative): GPT-4 exhibited self-contradictions at a rate of 15.7% in one experiment.
- `arxiv:2402.08955v1#c05` (arxiv:2402.08955v1 — "Using Counterfactual Tasks to Evaluate the Generality of Analogical Reasoning in Large Language Models", 2024, mixed): GPT-4 achieved mean accuracy of 0.452 across all alphabets and problem types.

### h267 — The contradiction is driven by dataset composition: GPT-4 looks strong on socially grounded multi-agent judgment/deception items but weak or misleading on benchmarks dominated by causal, counterfactual, or consistency-sensitive items.

**Mechanism.** Different evaluation subsets place weight on different capabilities; socially familiar patterns can boost GPT-4 scores, while causal-direction discovery, counterfactual abstraction, and contradiction probes expose failure modes or contamination effects, flipping aggregate performance claims.

**Predictions:**
- Performance stays high on judgment/deception-only splits
- Performance drops on causal/counterfactual/consistency-only splits

**Minimal test.** Build a unified evaluation suite with matched prompt template and scoring, partitioned into social judgment/deception items versus causal-reasoning, counterfactual, and contradiction-check items; evaluate the same GPT-4 version on each partition and test whether the positive vs negative claims collapse into a capability-by-subset interaction.

**Scope.** method=GPT-4, task=evaluation across multi-agent judgment, causal reasoning, consistency, and counterfactual comprehension

**Evidence gap.** A cross-benchmark reanalysis with shared item taxonomy is missing, so current claims do not isolate whether benchmark content alone explains the sign flip.

**Graph bridge.** GPT-4 → evaluation

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.89 | 0.89 | 0.00 | 0.00 | 0.00 | 0.81 |

### Anomaly a002 — benchmark_inconsistency

**Central question:** When does chain-of-thought help on multi-step-reasoning, and when does it fail?

**Shared entities:** method=chain-of-thought, task=multi-step-reasoning

**Evidence claims:**
- `arxiv:2412.17970v1#c04` (arxiv:2412.17970v1 — "CARL-GT: Evaluating Causal Reasoning Capabilities of Large Language Models", 2024, positive): Adjacency matrix estimation from tables is a multi-step task that may benefit from in-context learning or chain-of-thought.
- `arxiv:2403.03864v3#c01` (arxiv:2403.03864v3 — "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning", 2024, positive): GPT-4V with eCoT achieved the best overall score, but only slightly outperformed random.
- `arxiv:2403.03864v3#c02` (arxiv:2403.03864v3 — "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning", 2024, positive): The highest absolute score reported was GPT-4V CoT on the Calendar puzzle with 57% accuracy.
- `arxiv:2403.03864v3#c04` (arxiv:2403.03864v3 — "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning", 2024, positive): GPT-4V eCoT performs better on combinatorics, graphs, and sets than on optimization and search.
- `arxiv:2309.04461v2#c01` (arxiv:2309.04461v2 — "Measuring and Improving Chain-of-Thought Reasoning in Vision-Language Models", 2023, positive): CoTBLIP achieves the best reported reasoning performance and consistency among the listed VLMs in Table 2.
- `arxiv:2309.04461v2#c02` (arxiv:2309.04461v2 — "Measuring and Improving Chain-of-Thought Reasoning in Vision-Language Models", 2023, positive): The paper reports about a 4% relative improvement in reasoning performance and consistency over the state of the art.
- `arxiv:2302.00923v5#c01` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, positive): Multimodal-CoT Large improves over the prior best published model from 86.54% to 90.45%.
- `arxiv:2302.00923v5#c02` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, positive): Multimodal-CoT Base attains 50.57 accuracy in the main results.
- `arxiv:2302.00923v5#c05` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, positive): Two-stage methods achieve relatively higher accuracy early on than one-stage baselines without CoT.
- `arxiv:2309.06275v4#c02` (arxiv:2309.06275v4 — "Re-Reading Improves Reasoning in Large Language Models", 2023, positive): davinci-003 with CoT+Re2 further improves average performance on arithmetic, commonsense, and symbolic tasks over CoT.
- `arxiv:2403.02615v2#c01` (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, positive): GPT-4's accuracy increases by 13.9% with a step-by-step instruction.
- `arxiv:2403.02615v2#c03` (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, positive): Zero-shot chain-of-thought yields higher average accuracy than standard zero-shot across all models.
- `arxiv:2401.12117v3#c03` (arxiv:2401.12117v3 — "The Curious Case of Nonverbal Abstract Reasoning with Multi-Modal Large Language Models", 2024, positive): Chain-of-thought demonstrations boosted GPT-4V performance by as much as 100%.
- `arxiv:2409.09788v1#c04` (arxiv:2409.09788v1 — "Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models", 2024, positive): Zero-shot CoT significantly improves GPT-4V over a standard prompt.
- `arxiv:2401.04925v4#c01` (arxiv:2401.04925v4 — "The Impact of Reasoning Step Length on Large Language Models", 2024, positive): GPT-4 showed the highest tolerance to the strategy, with no performance decreases.
- `arxiv:2401.04925v4#c02` (arxiv:2401.04925v4 — "The Impact of Reasoning Step Length on Large Language Models", 2024, positive): The strategy had the strongest boosting effect on text-davinci-002, the model with the worst initial performance.
- `arxiv:2401.04925v4#c03` (arxiv:2401.04925v4 — "The Impact of Reasoning Step Length on Large Language Models", 2024, positive): The chain's length is more crucial than its factual accuracy for effective problem-solving.
- `arxiv:2407.08516v5#c02` (arxiv:2407.08516v5 — "Converging Paradigms: The Synergy of Symbolic and Connectionist AI in LLM-Empowered Autonomous Agents", 2024, positive): A step-by-step thought elaboration method improves problem-solving accuracy and reliability.
- `arxiv:2402.04236v3#c04` (arxiv:2402.04236v3 — "CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning", 2024, positive): The authors prompt GPT-4V to solve image questions step-by-step to obtain reasoning chains.
- `arxiv:2401.11337v1#c02` (arxiv:2401.11337v1 — "Prompting Large Vision-Language Models for Compositional Reasoning", 2024, positive): Chain-of-thought prompting gives small gains over a simpler selection prompt.
- `arxiv:2412.13292v2#c01` (arxiv:2412.13292v2 — "Refining Answer Distributions for Improved Large Language Model Reasoning", 2024, positive): CoT+RAD improves accuracy over baselines on the Big-Bench Hard datasets discussed.
- `arxiv:2309.13339v4#c03` (arxiv:2309.13339v4 — "Enhancing Zero-Shot Chain-of-Thought Reasoning in Large Language Models through Logic", 2023, positive): More capable language models such as GPT-4 and GPT-3.5-turbo revise more actively than smaller models.
- `arxiv:2407.00938v2#c02` (arxiv:2407.00938v2 — "MalAlgoQA: Pedagogical Evaluation of Counterfactual Reasoning in Large Language Models and Implications for AI in Education", 2024, mixed): For GPT-4o, CoT prompting has inconsistent effects on MIA across subjects: negligible change in Math but worse performance in Reading than simple prompting.
- `arxiv:2407.00938v2#c03` (arxiv:2407.00938v2 — "MalAlgoQA: Pedagogical Evaluation of Counterfactual Reasoning in Large Language Models and Implications for AI in Education", 2024, negative): LLaMA-70B has lower MIA with CoT than with simple prompting in both Reading and Math.
- `arxiv:2412.08317v1#c04` (arxiv:2412.08317v1 — "Large Language Models Still Face Challenges in Multi-Hop Reasoning with External Knowledge", 2024, negative): Performance in non-sequential reasoning cases is lower than in sequential reasoning cases, with 56% accuracy versus 78%.
- `arxiv:2406.10621v3#c01` (arxiv:2406.10621v3 — "StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in Structure-Rich Text", 2024, negative): Qwen2-7B-Instruct has lower accuracy with Self-CoT and PS-CoT prompts than with other prompting methods, especially on XML and Tree structures.
- `arxiv:2409.00106v1#c02` (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, mixed): Chain-of-thought prompting performs better than standard prompting only for GPT-3.5-Turbo (175B), and performs worse for smaller models.
- `arxiv:2411.13112v3#c02` (arxiv:2411.13112v3 — "SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios with Vision Language Models", 2024, mixed): Generating chain-of-thoughts in batch using QVQ is computationally expensive and often produces verbose, unstructured, or format-inconsistent outputs.
- `arxiv:2411.13112v3#c03` (arxiv:2411.13112v3 — "SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios with Vision Language Models", 2024, mixed): Relying only on Qwen2.5-VL-72B for chain-of-thought generation degrades output quality and increases hallucinations.
- `arxiv:2407.20564v1#c01` (arxiv:2407.20564v1 — "CLR-Fact: Evaluating the Complex Logical Reasoning Capability of Large Language Models over Factual Knowledge", 2024, mixed): Chain-of-thought prompting improves Precision@10 more as reasoning operation variety increases on GPT-3.5-turbo.
- `arxiv:2302.00923v5#c03` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, negative): The two-stage baseline model has only 78.57% answer inference accuracy despite strong rationale generation.
- `arxiv:2302.00923v5#c04` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, negative): The generated rationale in the two-stage framework does not improve answer accuracy compared with the QCM→A variant.
- `arxiv:2403.13315v3#c05` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, negative): Direct prompting was less effective than chain-of-thought prompting for Gemini Pro and GPT-4V on single-concept puzzles.
- `arxiv:2401.12117v3#c06` (arxiv:2401.12117v3 — "The Curious Case of Nonverbal Abstract Reasoning with Multi-Modal Large Language Models", 2024, negative): In a similar setting for GPT-4V, safeguard triggers increased from 4% to 18%, precluding responses, and demonstration-boundary confusion rose from 0% to 10%, reducing performance.
- `arxiv:2411.00855v1#c03` (arxiv:2411.00855v1 — "Vision-Language Models Can Self-Improve Reasoning via Reflection", 2024, negative): Zero-shot CoT with the "Let’s think step by step" prompt is ineffective and performs worse than direct prompting.
- `arxiv:2411.10442v2#c06` (arxiv:2411.10442v2 — "Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization", 2024, negative): For InternVL2-8B on MathVista, Chain-of-Thought reasoning reduces performance relative to direct answers.
- `arxiv:2405.00402v1#c05` (arxiv:2405.00402v1 — "Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models", 2024, mixed): Teacher model accuracy differs between CoT prompting and standard prompting on training and testing data across several benchmarks.
- `arxiv:2412.13292v2#c02` (arxiv:2412.13292v2 — "Refining Answer Distributions for Improved Large Language Model Reasoning", 2024, negative): With Llama-3-8b-instruct, PHP's advantage over CoT is smaller than with GPT models.

### h261 — The sign of chain-of-thought effects flips with dataset composition, especially whether tasks require linear decomposable substeps versus non-sequential, structure-heavy, or low-step problems.

**Mechanism.** CoT helps when the benchmark instances can be decomposed into explicit intermediate states that mirror a serial verbal trace, but hurts when items are non-sequential, require holistic pattern recognition, or contain dense structured inputs where verbalization adds distraction or loses structural fidelity.

**Predictions:**
- CoT gains are positive on sequential/high-operation-variety subsets.
- CoT gains vanish or reverse on non-sequential/XML-tree subsets.

**Minimal test.** Take one or more shared multi-step-reasoning benchmarks and annotate each item for reasoning topology: sequential linear, high operation variety, non-sequential, and structure-rich. Evaluate the same model, same prompt length, same decoding, and same scoring with and without CoT within each subset. If the contradiction is due to dataset composition, the average disagreement across papers should shrink after matching subset proportions or conditioning on topology.

**Scope.** method=chain-of-thought, task=multi-step-reasoning

**Evidence gap.** Most claims report aggregate benchmark scores without a common annotation of reasoning topology or structure density, so cross-paper subset comparability is missing.

**Graph bridge.** multi-step-reasoning → chain-of-thought

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.87 | 0.95 | 0.00 | 0.00 | 0.00 | 0.82 |

### Anomaly a005 — benchmark_inconsistency

**Central question:** When does evaluation-method help on scientific-reasoning, and when does it fail?

**Shared entities:** method=evaluation-method, task=scientific-reasoning

**Evidence claims:**
- `arxiv:2407.08029v1#c02` (arxiv:2407.08029v1 — "A Critical Review of Causal Reasoning Benchmarks for Large Language Models", 2024, positive): A true causal reasoning task should require interventional and/or counterfactual reasoning rather than simple retrieval of domain knowledge.
- `arxiv:2412.17970v1#c01` (arxiv:2412.17970v1 — "CARL-GT: Evaluating Causal Reasoning Capabilities of Large Language Models", 2024, positive): Llama3-8B and Mistral-7B outperform Qwen2-7B and Gemma2-9B on d-separation estimation.
- `arxiv:2412.17970v1#c02` (arxiv:2412.17970v1 — "CARL-GT: Evaluating Causal Reasoning Capabilities of Large Language Models", 2024, positive): All methods achieve at least AUC 0.5 on d-separation estimation.
- `arxiv:2412.17970v1#c05` (arxiv:2412.17970v1 — "CARL-GT: Evaluating Causal Reasoning Capabilities of Large Language Models", 2024, positive): Mixtral-8×7B performs better at causal graph reasoning than the other compared models, while Mistral-7B performs better on the other tasks than Qwen2-7B and Gemma2-9B.
- `arxiv:2310.07018v1#c03` (arxiv:2310.07018v1 — "NEWTON: Are Large Language Models Capable of Physical Reasoning?", 2023, positive): GPT-4 consistently outperforms other models in most NEWTON scenarios.
- `arxiv:2310.07018v1#c04` (arxiv:2310.07018v1 — "NEWTON: Are Large Language Models Capable of Physical Reasoning?", 2023, positive): GPT-4 performs well on both boolean and multiple-choice questions in NEWTON.
- `arxiv:2411.00387v3#c01` (arxiv:2411.00387v3 — "STEM-POM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing", 2024, positive): GPT-4o outperforms Llama3.1-70B across multiple context lengths.
- `arxiv:2310.07018v1#c01` (arxiv:2310.07018v1 — "NEWTON: Are Large Language Models Capable of Physical Reasoning?", 2023, negative): GPT-4 shows strong scenario-based reasoning but is less consistent than humans on object-attribute reasoning.
- `arxiv:2310.07018v1#c06` (arxiv:2310.07018v1 — "NEWTON: Are Large Language Models Capable of Physical Reasoning?", 2023, mixed): GPT-4 leads overall performance, but strengths vary by attribute across models.
- `arxiv:2411.00387v3#c02` (arxiv:2411.00387v3 — "STEM-POM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing", 2024, mixed): GPT-4o outperformed smaller models, but errors remained high on challenging symbols.

### h271 — The contradiction is moderated by prompt/output format: evaluation-method helps when the benchmark uses constrained boolean/multiple-choice or explicit graph-query formulations, but under free-form answer formatting it loses advantage because formatting and extraction errors mask reasoning quality.

**Mechanism.** Constrained prompts reduce variance in generation and scoring, making latent reasoning improvements visible. Free-form outputs for symbol parsing or open-ended explanations introduce extra failure channels—verbosity, malformed answers, ambiguous normalization—that can erase or reverse the measured benefit even if underlying reasoning is better.

**Predictions:**
- Multiple-choice/boolean reformulations increase measured gains for the same items.
- Answer normalization reduces the negative effect on symbol-heavy tasks.

**Minimal test.** Take one shared set of scientific-reasoning items and evaluate the same models under two controlled prompt regimes: constrained answers (boolean/MCQ/schema-constrained graph outputs) versus free-form text; use identical item content and a common parser/normalizer, then test whether the discrepancy between positive and mixed papers disappears within the same format.

**Scope.** method=evaluation-method, task=scientific-reasoning

**Evidence gap.** The current claims compare tasks that likely differ in output constraints, but they do not provide matched ablations isolating prompt format from item difficulty.

**Graph bridge.** boolean and multiple-choice style questions → math-symbol reasoning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.90 | 0.90 | 0.00 | 0.00 | 0.00 | 0.82 |

## Bridge Opportunities

These transfer questions are listed separately from conflicts because they do not imply opposing evidence.

### Bridge opportunity a333 — bridge_opportunity

**Transfer question:** Could the effect on multi-step-reasoning with prompting transfer to multi-step-reasoning with multimodal?

**Shared entities:** method_from=prompting, task_from=multi-step-reasoning, method_to=multimodal, task_to=multi-step-reasoning, shared_tokens=multi, step

**Evidence claims:**
- `arxiv:2408.05093v4#c05` (arxiv:2408.05093v4 — "Order Matters in Hallucination: Reasoning Order as Benchmark and Reflexive Prompting for Large-Language-Models", 2024, positive): The method is particularly suited to reasoning tasks because it compares outputs from an order prompt pair.
- `arxiv:2408.15778v4#c01` (arxiv:2408.15778v4 — "LogicGame: Benchmarking Rule-Based Reasoning Abilities of Large Language Models", 2024, positive): Stronger models such as gpt-4o and qwen2-72b-instruct gain more AP-Acc from few-shot prompting than weaker models.
- `arxiv:2410.01952v2#c01` (arxiv:2410.01952v2 — "TypedThinker: Diversify Large Language Model Reasoning with Typed Thinking", 2024, positive): TypedThinker improves performance across multiple benchmarks for Mistral 7B, LLaMA3 8B, and Qwen 2 7B on logical and mathematical reasoning tasks.
- `arxiv:2412.15296v1#c01` (arxiv:2412.15296v1 — "Confidence in the Reasoning of Large Language Models", 2024, negative): Second-answer accuracy is often worse than first-answer accuracy when models are asked to reconsider.
- `arxiv:2412.15296v1#c02` (arxiv:2412.15296v1 — "Confidence in the Reasoning of Large Language Models", 2024, negative): Changing their mind is associated with significantly worse accuracy for Mistral, reaching 36% and 32%, below the target value.
- `arxiv:2405.05508v2#c04` (arxiv:2405.05508v2 — "Redefining Information Retrieval of Structured Database via Large Language Models", 2024, negative): Few-shot learning with black-box LLMs like GPT-4 may yield unsatisfactory accuracy.
- `arxiv:2406.10621v3#c02` (arxiv:2406.10621v3 — "StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in Structure-Rich Text", 2024, positive): Meta-Llama-3.1-70B-Instruct-Turbo improves accuracy when using the w/ Hint prompt instead of the Naive prompt.
- `arxiv:2406.10621v3#c05` (arxiv:2406.10621v3 — "StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in Structure-Rich Text", 2024, positive): In a 3-shot scenario, GPT-4 substantially outperforms Gemini-Pro-Flash and Mistral.
- `arxiv:2407.01525v3#c04` (arxiv:2407.01525v3 — "ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities", 2024, positive): The LLM-based 3D reasoning grounding baseline segments scene objects and converts their categories and 3D bounding boxes into text for InternLM2-7B.
- `arxiv:2409.17906v1#c01` (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo+1-shot increases accuracy over 0-shot on Edge count and Node degree tasks for small graphs.
- `arxiv:2409.17906v1#c02` (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo+1-shot improves accuracy over 0-shot on Connected components, MST, and Shortest path tasks for small graphs.
- `arxiv:2409.17906v1#c03` (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo+1-shot achieves the highest accuracy on Node count, Edge count, and Node degree tasks.
- `arxiv:2409.17906v1#c04` (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo+1-shot yields a large accuracy increase on the Edge count task for medium-sized graphs.
- `arxiv:2409.17906v1#c05` (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo significantly outperforms 0-shot on the Cycle check task when using GPT-3.5.
- `arxiv:2409.17906v1#c06` (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo-code prompts improve performance across various graph tasks with GPT-3.5 and Mixtral.
- `arxiv:2305.12001v2#c01` (arxiv:2305.12001v2 — "OPT-R: Exploring the Role of Explanations in Finetuning and Prompting for Reasoning Skills of Large Language Models", 2023, positive): Incorporating explanations during finetuning and prompting benefits Numerical reasoning the most, with a +20.4% gain.
- `arxiv:2305.12001v2#c02` (arxiv:2305.12001v2 — "OPT-R: Exploring the Role of Explanations in Finetuning and Prompting for Reasoning Skills of Large Language Models", 2023, positive): Incorporating explanations during finetuning and prompting improves Analogical reasoning by +13.9%.
- `arxiv:2305.12001v2#c03` (arxiv:2305.12001v2 — "OPT-R: Exploring the Role of Explanations in Finetuning and Prompting for Reasoning Skills of Large Language Models", 2023, mixed): Some skills exhibit negligible or negative effects from incorporating explanations during finetuning and prompting.
- `arxiv:2405.01533v2#c01` (arxiv:2405.01533v2 — "OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning", 2024, positive): Using GPT-4 for counterfactual reasoning based on high-level decision making achieves good accuracy and interpretability.
- `arxiv:2407.11511v3#c03` (arxiv:2407.11511v3 — "Multi-Step Reasoning with Large Language Models, a Survey", 2024, positive): With older LLMs such as GPT-3, reasoning approaches improve over standard prompting by 20–50 percentage points.
- `arxiv:2409.00106v1#c06` (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): Flan-T5-XXL (11B) with standard prompting achieves the best performance and outperforms GPT-3.5-Turbo (175B).
- `arxiv:2412.13540v3#c02` (arxiv:2412.13540v3 — "Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning", 2024, positive): Adding MCDGraph improves performance on QA samples across various tasks.
- `arxiv:2412.13540v3#c03` (arxiv:2412.13540v3 — "Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning", 2024, positive): Adding MCDGraph improves performance on FC samples across various tasks.
- `arxiv:2404.05221v2#c04` (arxiv:2404.05221v2 — "LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models", 2024, negative): Inappropriate prompt format design can lead to false-positive reasoning chains.
- `arxiv:2408.04648v1#c01` (arxiv:2408.04648v1 — "PLUGH: A Benchmark for Spatial Understanding and Reasoning in Large Language Models", 2024, positive): In Graph Reconstruction, performance generally improves with more in-context examples.
- `arxiv:2408.04648v1#c02` (arxiv:2408.04648v1 — "PLUGH: A Benchmark for Spatial Understanding and Reasoning in Large Language Models", 2024, positive): In Novel Shortest Path, more shots reduce normalized Levenshtein distance.
- `arxiv:2408.04648v1#c03` (arxiv:2408.04648v1 — "PLUGH: A Benchmark for Spatial Understanding and Reasoning in Large Language Models", 2024, positive): In Temporal Hinted Shortest Path, more shots reduce normalized Levenshtein distance.
- `arxiv:2407.20564v1#c02` (arxiv:2407.20564v1 — "CLR-Fact: Evaluating the Complex Logical Reasoning Capability of Large Language Models over Factual Knowledge", 2024, positive): Demonstration selection yields an average performance improvement of 12-25% across two datasets.
- `arxiv:2409.12437v2#c04` (arxiv:2409.12437v2 — "Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data", 2024, negative): A task-specific prompt causes performance drops in the few-shot setting.
- `arxiv:2312.14890v4#c03` (arxiv:2312.14890v4 — "NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes", 2023, mixed): Open-source models such as Yi-34b and Mistral-7b generalize better from harder few-shot examples than from simpler ones.
- `arxiv:2312.14890v4#c04` (arxiv:2312.14890v4 — "NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes", 2023, positive): Phi-1.5 on EDP can generalize better from easier few-shot examples than from harder ones at some difficulty levels.
- `arxiv:2403.00816v3#c01` (arxiv:2403.00816v3 — "Read and Think: An Efficient Step-wise Multimodal Language Model for Document Understanding and Reasoning", 2024, positive): Step-wise generation improves performance over directly generated answers on InfoVQA and ChartQA.
- `arxiv:2406.03843v3#c02` (arxiv:2406.03843v3 — "POEM: Interactive Prompt Optimization for Enhancing Multimodal Reasoning of Large Language Models", 2024, mixed): The paper primarily focuses on visual and language modalities in multimodal LLM research.
- `arxiv:2403.11381v2#c05` (arxiv:2403.11381v2 — "Can LLM-Augmented autonomous agents cooperate?, An evaluation of their cooperative capabilities through Melting Pot", 2024, negative): GPT-3.5 and GPT-4 found it challenging to interpret and reason about spatial information in a matrix-based state representation.
- `arxiv:2309.06275v4#c01` (arxiv:2309.06275v4 — "Re-Reading Improves Reasoning in Large Language Models", 2023, positive): davinci-003 with Vanilla+Re2 improves average performance on arithmetic, commonsense, and symbolic tasks over Vanilla.
- `arxiv:2309.06275v4#c03` (arxiv:2309.06275v4 — "Re-Reading Improves Reasoning in Large Language Models", 2023, positive): LLMs with Re2 achieve consistent improvements across davinci-003 and ChatGPT under both Vanilla and CoT prompting.
- `arxiv:2309.06275v4#c04` (arxiv:2309.06275v4 — "Re-Reading Improves Reasoning in Large Language Models", 2023, positive): On Llama-2 models, the re-reading mechanism enhances Vanilla and CoT performance across most tasks.
- `arxiv:2403.13315v3#c06` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, positive): LLaVA-13B and Gemini Pro improved most when guided jointly by visual perception, inductive reasoning, and deductive reasoning.
- `arxiv:2405.13872v2#c01` (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, positive): Compared to GPT-4o, the model shows higher performance in Physical Relation, Object Location, and Spatial Relation categories.
- `arxiv:2405.13872v2#c02` (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, negative): Removing visual rationales and keeping only text rationales reduces performance on MMVet, especially in Knowledge and Spatial Awareness.
- `arxiv:2405.13872v2#c03` (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, positive): IoT improves performance on Cognition tasks, with gains reported for GPT-4o and Gemini-pro-1.5.
- `arxiv:2405.13872v2#c04` (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, positive): Compared with Gemini, IoT improves performance across more categories, with especially large gains in Action Recognition, Image Emotion, and Image Topic.
- `arxiv:2405.13872v2#c05` (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, mixed): Gemini-Pro's limited input image size makes it difficult to input a vision rationale series during MMVet experiments.
- `arxiv:2405.13872v2#c06` (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, positive): For GPT-4o, IoT improves performance across many categories, especially when comparing different objects through reasoning.
- `arxiv:2409.09788v1#c01` (arxiv:2409.09788v1 — "Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models", 2024, positive): Using a reference object is associated with a higher success rate than not using one on Q-Spatial Bench.
- `arxiv:2409.09788v1#c02` (arxiv:2409.09788v1 — "Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models", 2024, negative): Prompting GPT-4V and GPT-4o with detailed procedures hurts performance on Q-Spatial Bench.
- `arxiv:2409.09788v1#c03` (arxiv:2409.09788v1 — "Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models", 2024, positive): SpatialPrompt improves success rate across almost all VLMs on Q-Spatial Bench.
- `arxiv:2402.04236v3#c03` (arxiv:2402.04236v3 — "CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning", 2024, mixed): 5-shot prompting to GPT-4 yields stable solving steps, but makes manipulation and general-thinking descriptions similar.
- `arxiv:2408.00754v2#c04` (arxiv:2408.00754v2 — "Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Model", 2024, positive): The training-free Coarse Correspondences approach yields gains for GPT4-V/O across four spatial-temporal reasoning benchmarks.
- `arxiv:2410.02203v4#c02` (arxiv:2410.02203v4 — "GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning", 2024, positive): GraphIC surpasses all training-free baselines and outperforms all training-based methods under GPT-4o-mini and Llama-3.
- `arxiv:2410.02203v4#c05` (arxiv:2410.02203v4 — "GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning", 2024, positive): Even when the reasoning process is incorrect, GraphIC still maintains high accuracy.
- `arxiv:2410.02203v4#c06` (arxiv:2410.02203v4 — "GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning", 2024, positive): GraphIC outperforms 10 baseline methods across mathematical reasoning, code generation, and logical reasoning tasks.
- `arxiv:2310.01446v2#c04` (arxiv:2310.01446v2 — "Adaptive-Solver Framework for Dynamic Strategy Selection in Large Language Model Reasoning", 2023, mixed): Switching to a stronger LLM can improve performance but raises costs much more than prompting adaptation.
- `arxiv:2402.13577v1#c01` (arxiv:2402.13577v1 — "BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models", 2024, positive): Bba surpasses all compared baseline methods on three tasks with relative performance gains.
- `arxiv:2402.13577v1#c02` (arxiv:2402.13577v1 — "BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models", 2024, positive): Bba improves GPT-4V(ision) performance on geometry problem solving from 28.34% to 34.22%.
- `arxiv:2402.13577v1#c03` (arxiv:2402.13577v1 — "BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models", 2024, positive): The Bba prompting method enhances GPT-4V(ision)'s multimodal reasoning by integrating DSL.
- `arxiv:2405.00402v1#c04` (arxiv:2405.00402v1 — "Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models", 2024, positive): GPT-3.5 has a more robust baseline performance among the teacher models.
- `arxiv:2412.15238v2#c01` (arxiv:2412.15238v2 — "Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks", 2024, positive): Dipper with n=9 improves accuracy by about 10 percentage points over a single LLM baseline.
- `arxiv:2412.15238v2#c02` (arxiv:2412.15238v2 — "Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks", 2024, positive): An ensemble using seven different prompts outperforms both a self-ensemble without prompt variation and the average of seven single-prompt self-ensembles.
- `arxiv:2412.15238v2#c03` (arxiv:2412.15238v2 — "Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks", 2024, positive): The full Dipper implementation with FASV achieves the highest accuracy among the self-ensemble baseline and other Dipper variants across ensemble sizes.
- `arxiv:2402.11574v1#c01` (arxiv:2402.11574v1 — "Visual In-Context Learning for Large Vision-Language Models", 2024, positive): VICL improves performance over zero-shot and standard ICL baselines, especially for LLaVA-13B.
- `arxiv:2402.11574v1#c03` (arxiv:2402.11574v1 — "Visual In-Context Learning for Large Vision-Language Models", 2024, positive): Token reduction enables including more demonstrations within the LVLM token budget.
- `arxiv:2402.11574v1#c04` (arxiv:2402.11574v1 — "Visual In-Context Learning for Large Vision-Language Models", 2024, mixed): In LVLM ICL, the model must infer task intent and an image parsing strategy from reference triplets before analyzing the target image.
- `arxiv:2404.13985v2#c06` (arxiv:2404.13985v2 — "Information Re-Organization Improves Reasoning in Large Language Models", 2024, negative): Using GPT-3.5 for information re-organization with GPT-4 reasoning decreases reasoning ability.
- `arxiv:2312.01714v2#c04` (arxiv:2312.01714v2 — "Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models", 2023, positive): Adding demonstration examples in the context improves overall accuracy, especially on ScienceQA and MathVista.
- `arxiv:2406.11698v1#c01` (arxiv:2406.11698v1 — "Meta Reasoning for Large Language Models", 2024, positive): MRP attains the highest overall performance across 7 tasks with an average of 0.772.
- `arxiv:2406.11698v1#c02` (arxiv:2406.11698v1 — "Meta Reasoning for Large Language Models", 2024, positive): MRP outperforms comparison methods on BigToM and Code tasks.
- `arxiv:2406.11698v1#c04` (arxiv:2406.11698v1 — "Meta Reasoning for Large Language Models", 2024, mixed): MRP effectiveness depends on base model capability, performing satisfactorily with GPT-4 but suboptimally with GPT-3.5.
- `arxiv:2406.11698v1#c06` (arxiv:2406.11698v1 — "Meta Reasoning for Large Language Models", 2024, positive): MRP improves LLM ability on tasks requiring mixed reasoning strategies, especially for larger models like GPT-4.
- `arxiv:2306.17820v4#c01` (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): Meta-Reasoning improves LLM performance by 27.0% on the Letter task with 1/2 demonstrations compared to Chain-of-Thought.
- `arxiv:2306.17820v4#c02` (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): Using only one demonstration on the Track(7) task leads to a 39.6% performance boost.
- `arxiv:2306.17820v4#c03` (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): Meta-Reasoning achieves higher accuracy on tasks where pure CoT struggles, with gains of 27.0% on Letter and 37.7% on Track.
- `arxiv:2306.17820v4#c04` (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): GPT-3 text-davinci-002 outperforms the other three LLMs on five datasets after adopting Meta-Reasoning.
- `arxiv:2306.17820v4#c05` (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): Meta-Reasoning has lower error rates than Chain-of-Thought across all datasets shown in Table 4.
- `arxiv:2309.13339v4#c01` (arxiv:2309.13339v4 — "Enhancing Zero-Shot Chain-of-Thought Reasoning in Large Language Models through Logic", 2023, positive): GPT-4 showed improved accuracy on Date Understanding, LastLetter, and OddOneOut tasks with LoT revisions over default outputs.
- `arxiv:2409.13980v1#c01` (arxiv:2409.13980v1 — "Enhancing Advanced Visual Reasoning Ability of Large Language Models", 2024, positive): CVR-LLMLlama3 surpasses BLIP2 on WinoGAViL in the SWOW setting with 88.7% accuracy and a +17.1 improvement.
- `arxiv:2409.13980v1#c02` (arxiv:2409.13980v1 — "Enhancing Advanced Visual Reasoning Ability of Large Language Models", 2024, positive): The model outperforms MiniGPT4 with 62.0% accuracy and a +13.8 improvement on the GPT4-rated evaluation by Bitton-Guetta et al.
- `arxiv:2409.13980v1#c03` (arxiv:2409.13980v1 — "Enhancing Advanced Visual Reasoning Ability of Large Language Models", 2024, mixed): CVR-LLMLlama3 scores higher than LLaVA 1.5 on Whoops, VCR (Q->A), and NYCCC (Match).
- `arxiv:2408.08105v4#c01` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, mixed): LLaVA-Next achieves the best reported results, but performance remains near random baseline on MuCR tasks.
- `arxiv:2408.08105v4#c02` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, negative): Form-1 and Form-2 limit MLLMs' ability to recognize and use critical visual cues compared to Form-3.
- `arxiv:2408.08105v4#c03` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, negative): Using Form-1, GPT-4o failed to incorporate a specific visual cue and chose an effect image based on abstract textual interpretation rather than direct visual correlation.
- `arxiv:2408.08105v4#c04` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, positive): Form-3 gives GPT-4o direct visual information that helps it identify essential details across cause-and-effect images.
- `arxiv:2408.08105v4#c05` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, positive): Form-3 improves GPT-4o's ability to establish causal links by allowing freer analysis of raw visual inputs.
- `arxiv:2412.16599v1#c01` (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, negative): LLaVA-7B and LLaVA-13B achieve very low accuracy on relative compass direction reasoning, lower than on other tasks.
- `arxiv:2412.16599v1#c02` (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, negative): Claude-3 series and GPT-4o-mini perform poorly on relative compass direction reasoning, with accuracy mostly between 10% and 20%.
- `arxiv:2412.16599v1#c05` (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, positive): On absolute spatial direction reasoning, LLaVA-13B performs slightly better than LLaVA-7B, improving from 14.08% to 25.72% accuracy.
- `arxiv:2401.06805v2#c01` (arxiv:2401.06805v2 — "Exploring the Reasoning Abilities of Multimodal Large Language Models (MLLMs): A Comprehensive Survey on Emerging Trends in Multimodal Reasoning", 2024, positive): InfiMM-LLaMA-13B shows improved reasoning abilities over LLaVA-1.5 across various benchmarks despite using the same instruction dataset.
- `arxiv:2401.10529v2#c01` (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, positive): GPT-4V with sequential input demonstrates the best reasoning capability among evaluated MLLMs for understanding image sequences, except being on par with Gemini and LLaVA-1.5 in behavior precision.
- `arxiv:2401.10529v2#c02` (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, negative): Video-LLaMA-2 and Chat-UniVi do not show an advantage over LLaVA-1.5 on this benchmark, and Video-LLaMA-2 performs notably worse.
- `arxiv:2401.10529v2#c03` (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, mixed): GPT-4V performs much better on reasoning about objects in image sequences than on reasoning about behaviors.
- `arxiv:2401.10529v2#c04` (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, negative): GPT-4V scores only around 30% on behavior reasoning, with best recall barely exceeding 40%.
- `arxiv:2405.07229v2#c01` (arxiv:2405.07229v2 — "MM-InstructEval: Zero-Shot Evaluation of (Multimodal) Large Language Models on Multimodal Reasoning Tasks", 2024, mixed): Flan-T5 series and Qwen2.5-VL open-source models show strong multimodal reasoning ability but still trail closed-source models.
- `arxiv:2403.03864v3#c03` (arxiv:2403.03864v3 — "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning", 2024, positive): Guided vision substantially improves GPT-4V performance on some puzzles, indicating a visual perception bottleneck.
- `arxiv:2406.01584v3#c01` (arxiv:2406.01584v3 — "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models", 2024, positive): SpatialRGPT outperforms the state of the art by over 20% accuracy compared to GPT-4V-Turbo.
- `arxiv:2406.01584v3#c02` (arxiv:2406.01584v3 — "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models", 2024, positive): SpatialRGPT-7B performs better than the SpatialRGPT-7B(rgb) variant, especially when relative depth helps resolve ambiguities.
- `arxiv:2406.01584v3#c03` (arxiv:2406.01584v3 — "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models", 2024, positive): On BLINK Relative Depth results, SpatialRGPT-7B scores 82.3% and SpatialRGPT-VILA-1.5-8B scores 87.9%, above GPT-4V-Turbo's 66.9%.
- `arxiv:2406.02537v1#c02` (arxiv:2406.02537v1 — "TopViewRS: Vision-Language Models as Top-View Spatial Reasoners", 2024, negative): GPT-4V performs worse than Idefics-9B on both Static and Dynamic Spatial Reasoning tasks.
- `arxiv:2406.02537v1#c03` (arxiv:2406.02537v1 — "TopViewRS: Vision-Language Models as Top-View Spatial Reasoners", 2024, negative): Current state-of-the-art VLMs perform unsatisfactorily on TopViewRS, with model-wise average EM and PM below 50% over all tasks.
- `arxiv:2406.02537v1#c05` (arxiv:2406.02537v1 — "TopViewRS: Vision-Language Models as Top-View Spatial Reasoners", 2024, negative): Humans outperform GPT-4V by 47.78% on average.
- `arxiv:2409.00106v1#c01` (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): Providing scene metadata in addition to the image yields a 2% improvement over the base LLM for GPT-4, but not for BLIP2.
- `arxiv:2409.00106v1#c03` (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): BLIP-2 Flan-T5 XXL generally achieves higher accuracy than BLIP-2 Flan-T5 XL on CLEVR and PTR regardless of prompting technique.
- `arxiv:2409.00106v1#c05` (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): GPT-4 is 17% more accurate than GPT-4V on CLEVR.
- `arxiv:2302.00923v5#c06` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, positive): With vision features, rationale generation RougeL rises to 93.46% and this corresponds to better answer accuracy of 85.31%.
- `arxiv:2404.06479v5#c01` (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, negative): GPT-4o has substantial failure rates on simple vector-graphics reasoning tasks.
- `arxiv:2404.06479v5#c02` (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, positive): GPT-4o performs better than VDLM-mm on some tasks such as Shapeworld Spatial Reasoning.
- `arxiv:2404.06479v5#c03` (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, negative): Perception errors contribute to VDLM-mm performing worse than GPT-4o on Shapeworld tasks.
- `arxiv:2404.06479v5#c04` (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, positive): VDLM-mm yields consistent overall improvements over GPT-4V and GPT-4o on high-level visual reasoning tasks.
- `arxiv:2407.04212v1#c01` (arxiv:2407.04212v1 — "Smart Vision-Language Reasoners", 2024, positive): SmartestVLM lr0.0003 achieves 34.7, a +48% gain over BERT+ResNet50 on the counting skill in SMART.
- `arxiv:2407.04212v1#c02` (arxiv:2407.04212v1 — "Smart Vision-Language Reasoners", 2024, positive): Including the QF layer representation improves accuracy on all skill sets.
- `arxiv:2407.04212v1#c04` (arxiv:2407.04212v1 — "Smart Vision-Language Reasoners", 2024, negative): Baseline models from Cherian et al. (2022) struggle on the SMART task, especially when employing transformers.
- `arxiv:2411.01307v1#c01` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Explainer+FLAVA improves over FLAVA by 6.9%-9.5% across all metrics.
- `arxiv:2411.01307v1#c02` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Explainer+MKGformer performs better than MKGformer by 1.6%-2.6% in all five metrics.
- `arxiv:2411.01307v1#c04` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Predictor(LLaVA) has accuracy close to ChatGPT-4 and significantly outperforms other methods on MBARD.
- `arxiv:2411.01307v1#c05` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Predictor(VisualGLM) achieves 47.2% accuracy on MARS, significantly higher than other methods.
- `arxiv:2406.10923v1#c01` (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, positive): Replacing BLIP-2 with Gemini improves F1 on TiM by 4.5.
- `arxiv:2406.10923v1#c02` (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, positive): Higher frame rate consistently improves performance over sparse sampling on TiM.
- `arxiv:2406.10923v1#c03` (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, positive): Current methods only marginally outperform a random baseline on TiM for abstract perception and long-range compositional reasoning.
- `arxiv:2406.10923v1#c04` (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, mixed): SOTA methods achieve at most 25 F1 on TiM and remain far below human performance.
- `arxiv:2406.09403v3#c03` (arxiv:2406.09403v3 — "Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models", 2024, positive): Sketchpad substantially improves performance over strong base models with no sketching on both math and vision tasks.

### h164 — An unreported moderator variable drives the conflicting results around prompting on multi-step-reasoning.

**Mechanism.** A confound in data preprocessing, prompt formatting, or decoding parameters correlates with outcome direction and is not held constant across the claims.

**Predictions:**
- Holding prompt template and decoding fixed shrinks the between-claim variance by >50%.
- A covariate analysis reveals prompt/decoding parameters account for the sign flip.

**Minimal test.** Replay all claims on StrucText-Eval and related benchmarks in a common harness with identical prompts and decoding settings; recompute Accuracy and related metrics deltas.

**Evidence gap.** Prompt and decoding configurations are inconsistently reported across the claims.

**Graph bridge.** prompting → multimodal

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.86 | 0.93 | 0.00 | 0.05 | 0.00 | 0.82 |

### h163 — The reported reversal is driven by Accuracy and related metrics: surface-oriented scores reward overlap, so evidence-enriched answers that paraphrase can score lower even when factually correct.

**Mechanism.** Exact Match / F1 treats lexical overlap as ground truth and ignores whether the cited passages actually support the answer, rewarding short confident guesses.

**Predictions:**
- Re-scoring with evidence-chain F1 or LLM-as-judge reduces the sign flip between claims.
- Rankings of prompting variants on StrucText-Eval and related benchmarks shift when switching from Accuracy and related metrics to evidence-chain scoring.

**Minimal test.** Re-score the same StrucText-Eval and related benchmarks predictions with Accuracy and related metrics and evidence-chain F1; compute rank correlation.

**Scope.** metric=Accuracy and related metrics

**Evidence gap.** Few benchmarks report both surface metrics and evidence-chain metrics on the same runs.

**Graph bridge.** prompting → multimodal

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.86 | 0.93 | 0.00 | 0.05 | 0.00 | 0.82 |

### Bridge opportunity a353 — bridge_opportunity

**Transfer question:** Could the effect on multi-step-reasoning with multimodal transfer to multi-step-reasoning with search-based?

**Shared entities:** method_from=multimodal, task_from=multi-step-reasoning, method_to=search-based, task_to=multi-step-reasoning, shared_tokens=multi, step

**Evidence claims:**
- `arxiv:2409.13980v1#c01` (arxiv:2409.13980v1 — "Enhancing Advanced Visual Reasoning Ability of Large Language Models", 2024, positive): CVR-LLMLlama3 surpasses BLIP2 on WinoGAViL in the SWOW setting with 88.7% accuracy and a +17.1 improvement.
- `arxiv:2409.13980v1#c02` (arxiv:2409.13980v1 — "Enhancing Advanced Visual Reasoning Ability of Large Language Models", 2024, positive): The model outperforms MiniGPT4 with 62.0% accuracy and a +13.8 improvement on the GPT4-rated evaluation by Bitton-Guetta et al.
- `arxiv:2409.13980v1#c03` (arxiv:2409.13980v1 — "Enhancing Advanced Visual Reasoning Ability of Large Language Models", 2024, mixed): CVR-LLMLlama3 scores higher than LLaVA 1.5 on Whoops, VCR (Q->A), and NYCCC (Match).
- `arxiv:2408.08105v4#c01` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, mixed): LLaVA-Next achieves the best reported results, but performance remains near random baseline on MuCR tasks.
- `arxiv:2408.08105v4#c02` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, negative): Form-1 and Form-2 limit MLLMs' ability to recognize and use critical visual cues compared to Form-3.
- `arxiv:2408.08105v4#c03` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, negative): Using Form-1, GPT-4o failed to incorporate a specific visual cue and chose an effect image based on abstract textual interpretation rather than direct visual correlation.
- `arxiv:2408.08105v4#c04` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, positive): Form-3 gives GPT-4o direct visual information that helps it identify essential details across cause-and-effect images.
- `arxiv:2408.08105v4#c05` (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, positive): Form-3 improves GPT-4o's ability to establish causal links by allowing freer analysis of raw visual inputs.
- `arxiv:2412.16599v1#c01` (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, negative): LLaVA-7B and LLaVA-13B achieve very low accuracy on relative compass direction reasoning, lower than on other tasks.
- `arxiv:2412.16599v1#c02` (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, negative): Claude-3 series and GPT-4o-mini perform poorly on relative compass direction reasoning, with accuracy mostly between 10% and 20%.
- `arxiv:2412.16599v1#c05` (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, positive): On absolute spatial direction reasoning, LLaVA-13B performs slightly better than LLaVA-7B, improving from 14.08% to 25.72% accuracy.
- `arxiv:2401.06805v2#c01` (arxiv:2401.06805v2 — "Exploring the Reasoning Abilities of Multimodal Large Language Models (MLLMs): A Comprehensive Survey on Emerging Trends in Multimodal Reasoning", 2024, positive): InfiMM-LLaMA-13B shows improved reasoning abilities over LLaVA-1.5 across various benchmarks despite using the same instruction dataset.
- `arxiv:2401.10529v2#c01` (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, positive): GPT-4V with sequential input demonstrates the best reasoning capability among evaluated MLLMs for understanding image sequences, except being on par with Gemini and LLaVA-1.5 in behavior precision.
- `arxiv:2401.10529v2#c02` (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, negative): Video-LLaMA-2 and Chat-UniVi do not show an advantage over LLaVA-1.5 on this benchmark, and Video-LLaMA-2 performs notably worse.
- `arxiv:2401.10529v2#c03` (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, mixed): GPT-4V performs much better on reasoning about objects in image sequences than on reasoning about behaviors.
- `arxiv:2401.10529v2#c04` (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, negative): GPT-4V scores only around 30% on behavior reasoning, with best recall barely exceeding 40%.
- `arxiv:2405.07229v2#c01` (arxiv:2405.07229v2 — "MM-InstructEval: Zero-Shot Evaluation of (Multimodal) Large Language Models on Multimodal Reasoning Tasks", 2024, mixed): Flan-T5 series and Qwen2.5-VL open-source models show strong multimodal reasoning ability but still trail closed-source models.
- `arxiv:2403.03864v3#c03` (arxiv:2403.03864v3 — "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning", 2024, positive): Guided vision substantially improves GPT-4V performance on some puzzles, indicating a visual perception bottleneck.
- `arxiv:2406.01584v3#c01` (arxiv:2406.01584v3 — "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models", 2024, positive): SpatialRGPT outperforms the state of the art by over 20% accuracy compared to GPT-4V-Turbo.
- `arxiv:2406.01584v3#c02` (arxiv:2406.01584v3 — "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models", 2024, positive): SpatialRGPT-7B performs better than the SpatialRGPT-7B(rgb) variant, especially when relative depth helps resolve ambiguities.
- `arxiv:2406.01584v3#c03` (arxiv:2406.01584v3 — "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models", 2024, positive): On BLINK Relative Depth results, SpatialRGPT-7B scores 82.3% and SpatialRGPT-VILA-1.5-8B scores 87.9%, above GPT-4V-Turbo's 66.9%.
- `arxiv:2406.02537v1#c02` (arxiv:2406.02537v1 — "TopViewRS: Vision-Language Models as Top-View Spatial Reasoners", 2024, negative): GPT-4V performs worse than Idefics-9B on both Static and Dynamic Spatial Reasoning tasks.
- `arxiv:2406.02537v1#c03` (arxiv:2406.02537v1 — "TopViewRS: Vision-Language Models as Top-View Spatial Reasoners", 2024, negative): Current state-of-the-art VLMs perform unsatisfactorily on TopViewRS, with model-wise average EM and PM below 50% over all tasks.
- `arxiv:2406.02537v1#c05` (arxiv:2406.02537v1 — "TopViewRS: Vision-Language Models as Top-View Spatial Reasoners", 2024, negative): Humans outperform GPT-4V by 47.78% on average.
- `arxiv:2409.00106v1#c01` (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): Providing scene metadata in addition to the image yields a 2% improvement over the base LLM for GPT-4, but not for BLIP2.
- `arxiv:2409.00106v1#c03` (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): BLIP-2 Flan-T5 XXL generally achieves higher accuracy than BLIP-2 Flan-T5 XL on CLEVR and PTR regardless of prompting technique.
- `arxiv:2409.00106v1#c05` (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): GPT-4 is 17% more accurate than GPT-4V on CLEVR.
- `arxiv:2302.00923v5#c06` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, positive): With vision features, rationale generation RougeL rises to 93.46% and this corresponds to better answer accuracy of 85.31%.
- `arxiv:2404.06479v5#c01` (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, negative): GPT-4o has substantial failure rates on simple vector-graphics reasoning tasks.
- `arxiv:2404.06479v5#c02` (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, positive): GPT-4o performs better than VDLM-mm on some tasks such as Shapeworld Spatial Reasoning.
- `arxiv:2404.06479v5#c03` (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, negative): Perception errors contribute to VDLM-mm performing worse than GPT-4o on Shapeworld tasks.
- `arxiv:2404.06479v5#c04` (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, positive): VDLM-mm yields consistent overall improvements over GPT-4V and GPT-4o on high-level visual reasoning tasks.
- `arxiv:2407.04212v1#c01` (arxiv:2407.04212v1 — "Smart Vision-Language Reasoners", 2024, positive): SmartestVLM lr0.0003 achieves 34.7, a +48% gain over BERT+ResNet50 on the counting skill in SMART.
- `arxiv:2407.04212v1#c02` (arxiv:2407.04212v1 — "Smart Vision-Language Reasoners", 2024, positive): Including the QF layer representation improves accuracy on all skill sets.
- `arxiv:2407.04212v1#c04` (arxiv:2407.04212v1 — "Smart Vision-Language Reasoners", 2024, negative): Baseline models from Cherian et al. (2022) struggle on the SMART task, especially when employing transformers.
- `arxiv:2411.01307v1#c01` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Explainer+FLAVA improves over FLAVA by 6.9%-9.5% across all metrics.
- `arxiv:2411.01307v1#c02` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Explainer+MKGformer performs better than MKGformer by 1.6%-2.6% in all five metrics.
- `arxiv:2411.01307v1#c04` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Predictor(LLaVA) has accuracy close to ChatGPT-4 and significantly outperforms other methods on MBARD.
- `arxiv:2411.01307v1#c05` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Predictor(VisualGLM) achieves 47.2% accuracy on MARS, significantly higher than other methods.
- `arxiv:2406.10923v1#c01` (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, positive): Replacing BLIP-2 with Gemini improves F1 on TiM by 4.5.
- `arxiv:2406.10923v1#c02` (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, positive): Higher frame rate consistently improves performance over sparse sampling on TiM.
- `arxiv:2406.10923v1#c03` (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, positive): Current methods only marginally outperform a random baseline on TiM for abstract perception and long-range compositional reasoning.
- `arxiv:2406.10923v1#c04` (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, mixed): SOTA methods achieve at most 25 F1 on TiM and remain far below human performance.
- `arxiv:2406.09403v3#c03` (arxiv:2406.09403v3 — "Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models", 2024, positive): Sketchpad substantially improves performance over strong base models with no sketching on both math and vision tasks.
- `arxiv:2404.05221v2#c01` (arxiv:2404.05221v2 — "LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models", 2024, positive): Reward-guided search improves final accuracy and reduces false-positive reasoning chains.
- `arxiv:2404.05221v2#c02` (arxiv:2404.05221v2 — "LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models", 2024, positive): Breadth of search is generally more important than depth for efficient reasoning-space search on most tasks.
- `arxiv:2412.14835v1#c03` (arxiv:2412.14835v1 — "Progressive Multimodal Reasoning via Active Retrieval", 2024, positive): GPT-4o and Qwen2-VL-7B improve over backbone and self-consistency approaches when combined with AR-MCTS.
- `arxiv:2412.14835v1#c04` (arxiv:2412.14835v1 — "Progressive Multimodal Reasoning via Active Retrieval", 2024, positive): AR-MCTS with GPT-4o yields stable gains in mathematics and physics, with some gains in humanities.
- `arxiv:2412.14835v1#c06` (arxiv:2412.14835v1 — "Progressive Multimodal Reasoning via Active Retrieval", 2024, positive): AR-MCTS gains are more pronounced in smaller MLLMs.
- `arxiv:2402.04236v3#c02` (arxiv:2402.04236v3 — "CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning", 2024, negative): Integrating backtracking leads to high time complexity.
- `arxiv:2304.14732v7#c01` (arxiv:2304.14732v7 — "Search-in-the-Chain: Interactively Enhancing Large Language Models with Search for Knowledge-intensive Tasks", 2023, positive): SearChain without IR outperforms Self-Ask without IR and Least-to-Most.
- `arxiv:2404.12253v2#c02` (arxiv:2404.12253v2 — "Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing", 2024, positive): Models trained on trajectories generated by the method outperform models trained on reranked trajectories.
- `arxiv:2404.12253v2#c03` (arxiv:2404.12253v2 — "Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing", 2024, mixed): When decoded with the specified decoder, models trained on generated trajectories perform on par with GPT-4.
- `arxiv:2410.24155v4#c01` (arxiv:2410.24155v4 — "Blind Spot Navigation in Large Language Model Reasoning with Thought Space Explorer", 2024, positive): TSE improves accuracy over non-thinking outputs by 59.2% on Qwen3-4B and 51.6% on Qwen3-8B on average.
- `arxiv:2410.24155v4#c03` (arxiv:2410.24155v4 — "Blind Spot Navigation in Large Language Model Reasoning with Thought Space Explorer", 2024, positive): TSE improves the accuracy of both final answers and intermediate reasoning steps compared with existing baseline methods, while offering a better effectiveness-efficiency trade-off.
- `arxiv:2410.24155v4#c04` (arxiv:2410.24155v4 — "Blind Spot Navigation in Large Language Model Reasoning with Thought Space Explorer", 2024, mixed): TSE has a limitation in that it requires more tokens than direct decoding, despite a favorable token-accuracy trade-off versus baselines.

### h223 — Breadth-first thought expansion from search-based reasoning will help multimodal sequence reasoning more than deeper single-chain rollouts because image/video tasks contain multiple plausible event hypotheses that benefit from parallel exploration.

**Mechanism.** When evidence is temporally ambiguous, broader exploration can preserve competing interpretations of object state and behavior before committing, analogous to claims that breadth matters more than depth in reasoning-space search and that multimodal sequence models struggle especially on behavior reasoning.

**Predictions:**
- Breadth-8 shallow search improves behavior F1 on Mementos more than depth-8 rollouts.
- Benefit is larger for behavior questions than object questions on Mementos.

**Minimal test.** Take GPT-4V or Gemini as the base model on Mementos and TiM, implement two search policies over intermediate event hypotheses: breadth-first expansion with width 8 and depth 2 versus depth-oriented rollout with width 2 and depth 8; compare to direct sequential prompting baseline using F1 on Mementos and TiM, with separate reporting for object versus behavior subsets on Mementos.

**Scope.** method=breadth-prioritized search policy transferred to multimodal temporal reasoning, task=image-sequence and video multi-step reasoning with latent event-state ambiguity

**Evidence gap.** Existing multimodal results compare input formats and frame rates, but do not test whether search-width versus search-depth is the main driver under fixed compute.

**Graph bridge.** breadth of search → multi-step-reasoning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.85 | 0.92 | 0.00 | 0.05 | 0.00 | 0.82 |

### Bridge opportunity a347 — bridge_opportunity

**Transfer question:** Could the effect on multi-step-reasoning with agent transfer to multi-step-reasoning with fine-tuning?

**Shared entities:** method_from=agent, task_from=multi-step-reasoning, method_to=fine-tuning, task_to=multi-step-reasoning, shared_tokens=multi, step

**Evidence claims:**
- `arxiv:2405.18358v1#c01` (arxiv:2405.18358v1 — "MMCTAgent: Multi-modal Critical Thinking Agent Framework for Complex Visual Reasoning", 2024, positive): MMCTAgent achieves 74.2% accuracy on MMVET and outperforms Claude 3, GPT-4V, and Gemini by 22.3, 14.1, and 10.4 percentage points, respectively.
- `arxiv:2405.18358v1#c02` (arxiv:2405.18358v1 — "MMCTAgent: Multi-modal Critical Thinking Agent Framework for Complex Visual Reasoning", 2024, positive): MMCTAgent with a vision-based critic consistently outperforms Claude 3, GPT-4V, and Gemini by at least 10% across all datasets.
- `arxiv:2405.18358v1#c03` (arxiv:2405.18358v1 — "MMCTAgent: Multi-modal Critical Thinking Agent Framework for Complex Visual Reasoning", 2024, positive): MMCTAgent surpasses GPT-4V by 10%, Claude 3 by 15%, and Gemini models by 10% on average across all datasets.
- `arxiv:2406.07155v3#c03` (arxiv:2406.07155v3 — "Scaling Large Language Model-based Multi-Agent Collaboration", 2024, negative): Ablating agents’ profiles causes an average performance drop of 3.67% across all topologies.
- `arxiv:2406.07155v3#c04` (arxiv:2406.07155v3 — "Scaling Large Language Model-based Multi-Agent Collaboration", 2024, positive): When a critic suggests a particular aspect, an actor implements the recommended refinement with 93.10% likelihood rather than disregarding it.
- `arxiv:2308.05960v1#c04` (arxiv:2308.05960v1 — "BOLAA: Benchmarking and Orchestrating LLM-augmented Autonomous Agents", 2023, positive): text-davinci-003 consistently outperforms Llama-2-70b across all levels of complexity.
- `arxiv:2411.16189v1#c01` (arxiv:2411.16189v1 — "Enhancing Multi-Agent Consensus through Third-Party LLM Integration: Analyzing Uncertainty and Mitigating Hallucinations in Large Language Models", 2024, positive): The proposed method achieved 0.940 accuracy under the Attention-All setting and outperformed baseline methods.
- `arxiv:2411.16189v1#c05` (arxiv:2411.16189v1 — "Enhancing Multi-Agent Consensus through Third-Party LLM Integration: Analyzing Uncertainty and Mitigating Hallucinations in Large Language Models", 2024, mixed): The study has limitations in computational efficiency, attention mechanism application, and cross-domain experiments.
- `arxiv:2402.18240v2#c06` (arxiv:2402.18240v2 — "Prospect Personalized Recommendation on Large Language Model-based Agent Platform", 2024, positive): Collaborative or mutually critical multi-agent approaches aim to improve group problem-solving over individuals.
- `arxiv:2406.11776v1#c03` (arxiv:2406.11776v1 — "Improving Multi-Agent Debate with Sparse Communication Topology", 2024, positive): Sparse MAD reduces token usage by 40.6% in multimodal reasoning, excluding input image tokens.
- `arxiv:2406.11776v1#c04` (arxiv:2406.11776v1 — "Improving Multi-Agent Debate with Sparse Communication Topology", 2024, positive): Placing the stronger LLM at a higher-centrality node leads to better performance than placing it at a lower-centrality node.
- `arxiv:2408.06849v2#c01` (arxiv:2408.06849v2 — "Causal Agent based on Large Language Model", 2024, positive): Using GLM-4-plus, the model is 6% higher than the SOTA on real-world tabular data.
- `arxiv:2408.06849v2#c02` (arxiv:2408.06849v2 — "Causal Agent based on Large Language Model", 2024, positive): The base model still outperforms SOTA results despite being less powerful than GPT-4.
- `arxiv:2408.06849v2#c03` (arxiv:2408.06849v2 — "Causal Agent based on Large Language Model", 2024, positive): Compared with code-LLM, the model shows significant variable-level accuracy improvement.
- `arxiv:2311.08152v2#c01` (arxiv:2311.08152v2 — "Towards Reasoning in Large Language Models via Multi-Agent Peer Review Collaboration", 2023, positive): The peer review strategy outperforms single-agent and multi-agent baselines on all datasets, with notable gains on challenging benchmarks.
- `arxiv:2311.08152v2#c02` (arxiv:2311.08152v2 — "Towards Reasoning in Large Language Models via Multi-Agent Peer Review Collaboration", 2023, negative): Increasing the number of review rounds does not significantly improve accuracy.
- `arxiv:2311.08152v2#c03` (arxiv:2311.08152v2 — "Towards Reasoning in Large Language Models via Multi-Agent Peer Review Collaboration", 2023, negative): A larger capability gap reduces the performance gains for the stronger model despite similar diversity.
- `arxiv:2311.08152v2#c04` (arxiv:2311.08152v2 — "Towards Reasoning in Large Language Models via Multi-Agent Peer Review Collaboration", 2023, negative): The method exhibits significant overconfidence because accuracy is much lower than confidence within bins.
- `arxiv:2409.14051v2#c01` (arxiv:2409.14051v2 — "GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion", 2024, positive): GD consistently reduces token cost across models, agent settings, and round settings.
- `arxiv:2409.14051v2#c02` (arxiv:2409.14051v2 — "GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion", 2024, positive): The 3+3 group strategy achieves the best accuracy in the experiments.
- `arxiv:2409.14051v2#c03` (arxiv:2409.14051v2 — "GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion", 2024, mixed): Accuracy increases as debate rounds increase, but decreases when rounds exceed 4.
- `arxiv:2409.14051v2#c04` (arxiv:2409.14051v2 — "GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion", 2024, positive): Different group strategies improve accuracy and reduce token cost compared to not grouping.
- `arxiv:2409.14051v2#c05` (arxiv:2409.14051v2 — "GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion", 2024, positive): MAD+Group reduces token usage and improves accuracy compared to MAD+Forget.
- `arxiv:2412.20145v2#c01` (arxiv:2412.20145v2 — "Efficient Multi-Agent Collaboration with Tool Use for Online Planning in Complex Table Question Answering", 2024, positive): MACT outperforms previous state-of-the-art systems on three of four benchmarks and is comparable to GPT-4 on two benchmarks using only open-weight models without fine-tuning.
- `arxiv:2412.20145v2#c02` (arxiv:2412.20145v2 — "Efficient Multi-Agent Collaboration with Tool Use for Online Planning in Complex Table Question Answering", 2024, positive): On CRT, MACT outperforms GPT-4 by 5.7%.
- `arxiv:2412.20145v2#c03` (arxiv:2412.20145v2 — "Efficient Multi-Agent Collaboration with Tool Use for Online Planning in Complex Table Question Answering", 2024, positive): MACT (Qw+CL) outperforms SC(Qw+CL) by about 6 EM points on average across all datasets.
- `arxiv:2412.20145v2#c04` (arxiv:2412.20145v2 — "Efficient Multi-Agent Collaboration with Tool Use for Online Planning in Complex Table Question Answering", 2024, positive): Using multiple agents improves EM over individually prompting Qwen and CodeLLaMA.
- `arxiv:2409.12411v1#c01` (arxiv:2409.12411v1 — "Textualized Agent-Style Reasoning for Complex Tasks by Multiple Round LLM Generation", 2024, positive): AgentCOT improves average accuracy over traditional COT on text-davinci-002 and gpt-3.5-turbo.
- `arxiv:2409.12411v1#c03` (arxiv:2409.12411v1 — "Textualized Agent-Style Reasoning for Complex Tasks by Multiple Round LLM Generation", 2024, negative): Removing action description hurts AgentCOT more than removing action.
- `arxiv:2409.12411v1#c04` (arxiv:2409.12411v1 — "Textualized Agent-Style Reasoning for Complex Tasks by Multiple Round LLM Generation", 2024, negative): Without action or action description, AgentCOT performance drops on average.
- `arxiv:2410.20007v1#c02` (arxiv:2410.20007v1 — "Cooperative Strategic Planning Enhances Reasoning Capabilities in Large Language Models", 2024, positive): The LLaMA-3-8B-based CoPlanner outperforms all other baselines on LogiQA and BBH.
- `arxiv:2410.20007v1#c05` (arxiv:2410.20007v1 — "Cooperative Strategic Planning Enhances Reasoning Capabilities in Large Language Models", 2024, mixed): The two agents in CoPlanner interact over several rounds rather than solving the problem in a single step.
- `arxiv:2411.14432v2#c05` (arxiv:2411.14432v2 — "Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models", 2024, negative): With limited data, the reasoning agent fails to generalize and performs worse than baseline models.
- `arxiv:2412.05237v2#c01` (arxiv:2412.05237v2 — "MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale", 2024, positive): Training MLLMs on this dataset significantly improves reasoning capabilities and achieves state-of-the-art performance on MathVerse, MMMU-Pro, and MuirBench.
- `arxiv:2412.05237v2#c02` (arxiv:2412.05237v2 — "MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale", 2024, positive): MAmmoTH-VL-8B improves performance on MathVerse, MMMU-Pro, and MuirBench.
- `arxiv:2412.16599v1#c03` (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, positive): Fine-tuning LLaVA-7B on CDR with CoTAll+40K raises relative compass reasoning accuracy to 53.43%, above the 11.90% base model.
- `arxiv:2305.12001v2#c04` (arxiv:2305.12001v2 — "OPT-R: Exploring the Role of Explanations in Finetuning and Prompting for Reasoning Skills of Large Language Models", 2023, positive): Finetuned RE and R models outperform the vanilla OPT model on some reasoning skills.
- `arxiv:2305.12001v2#c05` (arxiv:2305.12001v2 — "OPT-R: Exploring the Role of Explanations in Finetuning and Prompting for Reasoning Skills of Large Language Models", 2023, positive): RE has better accuracy than other models on some skills, suggesting the importance of finetuning on explanations for those skills.
- `arxiv:2405.07938v2#c06` (arxiv:2405.07938v2 — "EconLogicQA: A Question-Answering Benchmark for Evaluating Large Language Models in Economic Sequential Reasoning", 2024, negative): Mistral-7B-Instruct-v0.3 lags behind GPT-4 overall despite achieving 39.23% accuracy in 1-shot and 35.38% in 5-shot.
- `arxiv:2409.12437v2#c01` (arxiv:2409.12437v2 — "Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data", 2024, positive): On StepGame, training with synthetic graph-based data significantly improves performance over SFT on textual stories and over GPT-4o across all hops.
- `arxiv:2409.12437v2#c02` (arxiv:2409.12437v2 — "Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data", 2024, negative): Extra tuning with synthetic data was necessary because SFT-S underperformed GPT-4o in most cases.
- `arxiv:2409.12437v2#c03` (arxiv:2409.12437v2 — "Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data", 2024, mixed): Scaling up with synthetic data yields more pronounced performance gaps between SFT-S and GPT-4o than those seen on CLUTRR.
- `arxiv:2403.04483v4#c01` (arxiv:2403.04483v4 — "GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability", 2024, positive): GraphSolver+ achieves 53% accuracy on the medium-size DFS task, about six times LLaMA-3.1-8B-Instruct's performance.
- `arxiv:2403.04483v4#c02` (arxiv:2403.04483v4 — "GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability", 2024, positive): GraphSolver+ significantly outperforms all baseline models across all five tasks when intermediate reasoning steps are included in supervision.
- `arxiv:2403.04483v4#c03` (arxiv:2403.04483v4 — "GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability", 2024, positive): GraphSolver outperforms baseline models such as Phi-4-mini-Instruct and GPT-3.5 Turbo even when node IDs are altered.
- `arxiv:2403.04483v4#c04` (arxiv:2403.04483v4 — "GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability", 2024, positive): Following fine-tuning, GraphSolver substantially improves over LLaMA-3.1-8B-Instruct on most tasks.
- `arxiv:2403.04483v4#c05` (arxiv:2403.04483v4 — "GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability", 2024, negative): GraphSolver fails to effectively solve unseen graph reasoning tasks.
- `arxiv:2405.20535v2#c01` (arxiv:2405.20535v2 — "Unveiling the Impact of Coding Data Instruction Fine-Tuning on Large Language Models Reasoning", 2024, positive): Fine-tuning with Code yields substantial relative gains for Mistral-7B-v0.1 and Gemma-7B compared to General.
- `arxiv:2405.20535v2#c02` (arxiv:2405.20535v2 — "Unveiling the Impact of Coding Data Instruction Fine-Tuning on Large Language Models Reasoning", 2024, positive): Llama-3-8B shows only a modest relative improvement.
- `arxiv:2405.20535v2#c03` (arxiv:2405.20535v2 — "Unveiling the Impact of Coding Data Instruction Fine-Tuning on Large Language Models Reasoning", 2024, positive): In Llama-1, the Code setting yields a larger gain on Babi-Inductive than the Half-half setting does on Babi-Deductive.
- `arxiv:2405.20535v2#c04` (arxiv:2405.20535v2 — "Unveiling the Impact of Coding Data Instruction Fine-Tuning on Large Language Models Reasoning", 2024, mixed): Performance does not increase monotonically with coding data proportion between 25% and 50%.
- `arxiv:2410.02338v2#c02` (arxiv:2410.02338v2 — "How Much Can RAG Help the Reasoning of LLM?", 2024, negative): The LoRA-fine-tuned model can perform worse than vanilla reasoning with RAG.
- `arxiv:2406.10625v2#c02` (arxiv:2406.10625v2 — "On the Hardness of Faithful Chain-of-Thought Reasoning in Large Language Models", 2024, mixed): Some sampling strategies improve faithfulness for fine-tuned GPT-3.5-Turbo, but remain below the GTA baseline for fine-tuned Llama-3-8b-Instruct.
- `arxiv:2404.03577v1#c01` (arxiv:2404.03577v1 — "Untangle the KNOT: Interweaving Conflicting Knowledge and Reasoning Skills in Large Language Models", 2024, positive): Assistant models improve performance compared with pre-trained only LLMs.
- `arxiv:2404.03577v1#c04` (arxiv:2404.03577v1 — "Untangle the KNOT: Interweaving Conflicting Knowledge and Reasoning Skills in Large Language Models", 2024, positive): Fine-tuning yields accuracy gains compared with the Table 2 baseline.
- `arxiv:2408.00754v2#c05` (arxiv:2408.00754v2 — "Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Model", 2024, positive): Applying Coarse Correspondences in both training and inference improves open-source MLLMs on ScanQA and generalizes to unseen SQA3D.
- `arxiv:2412.03467v1#c01` (arxiv:2412.03467v1 — "Training-Free Mitigation of Language Reasoning Degradation After Multimodal Instruction Tuning", 2024, mixed): LLaVA-1.6-Mistral underperforms its base Mistral model on nearly all evaluated tasks, while LLaVA-1.5 and LLaVA-1.6 match or exceed Vicuna on most tasks.
- `arxiv:2412.08125v2#c02` (arxiv:2412.08125v2 — "Progressive Multi-granular Alignments for Grounded Reasoning in Large Vision-Language Models", 2024, positive): PromViL outperforms Kosmos-8K and Kosmos-16K in accuracy.
- `arxiv:2412.08125v2#c03` (arxiv:2412.08125v2 — "Progressive Multi-granular Alignments for Grounded Reasoning in Large Vision-Language Models", 2024, positive): PromViL's accuracy increases as scene complexity increases from 1 to 3 objects.
- `arxiv:2412.08125v2#c05` (arxiv:2412.08125v2 — "Progressive Multi-granular Alignments for Grounded Reasoning in Large Vision-Language Models", 2024, positive): PromViL shows significant improvements over same-size models with limited tunable parameters and fine-tuning data.
- `arxiv:2410.04055v3#c05` (arxiv:2410.04055v3 — "Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks", 2024, positive): Increasing the number of training samples consistently improves overall accuracy for SCL.
- `arxiv:2410.18890v1#c01` (arxiv:2410.18890v1 — "Improving Small-Scale Large Language Models Function Calling for Reasoning Tasks", 2024, positive): The fine-tuned model outperforms the original model in overall accuracy on the whole set when nmax=10.
- `arxiv:2410.18890v1#c02` (arxiv:2410.18890v1 — "Improving Small-Scale Large Language Models Function Calling for Reasoning Tasks", 2024, positive): The fine-tuned model outperforms the original model in overall accuracy on the whole set when nmax=20.
- `arxiv:2410.18890v1#c03` (arxiv:2410.18890v1 — "Improving Small-Scale Large Language Models Function Calling for Reasoning Tasks", 2024, positive): The fine-tuned model outperforms the original model in overall accuracy on the whole set when combining nmax=10 and 20.
- `arxiv:2412.16653v1#c02` (arxiv:2412.16653v1 — "Internalized Self-Correction for Large Language Models", 2024, mixed): The experiments fine-tune Meta Llama3.1 8B on synthetic chain-of-thought data with and without negative samples.
- `arxiv:2410.09489v1#c01` (arxiv:2410.09489v1 — "Towards Efficient Visual-Language Alignment of the Q-Former for Visual Reasoning Tasks", 2024, positive): Applying LoRA to both the Q-Former and LLM achieves superior performance on both benchmarks with fewer than 12% of trainable parameters.
- `arxiv:2410.09489v1#c02` (arxiv:2410.09489v1 — "Towards Efficient Visual-Language Alignment of the Q-Former for Visual Reasoning Tasks", 2024, mixed): Applying PEFT to the Q-Former achieves comparable performance to full fine-tuning while using less than 2% of trainable parameters.
- `arxiv:2410.09489v1#c03` (arxiv:2410.09489v1 — "Towards Efficient Visual-Language Alignment of the Q-Former for Visual Reasoning Tasks", 2024, positive): Fine-tuning the base LLMs with LoRA consistently outperforms the baseline InstructBLIP model on both benchmarks.
- `arxiv:2410.09489v1#c04` (arxiv:2410.09489v1 — "Towards Efficient Visual-Language Alignment of the Q-Former for Visual Reasoning Tasks", 2024, mixed): Applying LoRA to the Q-Former yields competitive performance, matching or surpassing full fine-tuning while using less than 2% of the original trainable parameters.
- `arxiv:2410.20007v1#c04` (arxiv:2410.20007v1 — "Cooperative Strategic Planning Enhances Reasoning Capabilities in Large Language Models", 2024, negative): Removing behavioral cloning from CoPlanner reduces performance on LogiQA and BBH.
- `arxiv:2411.14432v2#c01` (arxiv:2411.14432v2 — "Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models", 2024, positive): Insight-V yields average performance gains when applied to LLaVA-NeXT and the base model across all benchmarks.

### h206 — Communication-topology ideas from multi-agent reasoning can be transferred into parameter-efficient fine-tuning by training structured modular adapters with asymmetric centrality, producing similar reasoning gains at lower trainable-parameter cost than uniform LoRA.

**Mechanism.** Agent studies report that sparse/grouped communication and central placement of stronger components can improve cost-performance tradeoffs; an analogous fine-tuning design would assign larger or cross-layer adapters to central reasoning layers and smaller adapters elsewhere, encouraging staged information aggregation inside one model.

**Predictions:**
- Centrality-weighted adapters outperform uniform LoRA at equal trainable-parameter budget.
- Sparse grouped adapters reduce training FLOPs with no more than 1 point accuracy loss.

**Minimal test.** Implement a centrality-weighted LoRA scheme on LLaVA-7B or InstructBLIP for MathVerse or ScanQA: allocate higher ranks to middle/high-impact transformer blocks and low ranks to peripheral blocks, and compare against uniform-rank LoRA and full fine-tuning baselines under the same total trainable-parameter budget; report accuracy/EM, trainable-parameter count, and training FLOPs, with an ablation for sparse grouped adapter sharing across block groups.

**Scope.** method=parameter-efficient fine-tuning with topology-inspired nonuniform adapter allocation, task=multimodal or text multi-step reasoning where PEFT is already feasible

**Evidence gap.** The analogy between external agent communication topology and internal adapter topology has not been directly tested, so it is unknown whether agent centrality effects map onto layerwise fine-tuning structure.

**Graph bridge.** agent → fine-tuning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.91 | 0.00 | 0.05 | 0.00 | 0.82 |

### Bridge opportunity a362 — bridge_opportunity

**Transfer question:** Could the effect on factual-QA with fine-tuning transfer to domain-QA with fine-tuning?

**Shared entities:** method_from=fine-tuning, task_from=factual-QA, method_to=fine-tuning, task_to=domain-QA, shared_tokens=fine, tuning

**Evidence claims:**
- `arxiv:2412.01370v2#c03` (arxiv:2412.01370v2 — "Understanding Museum Exhibits using Vision-Language Reasoning", 2024, positive): Models fine-tuned with the 20mn dataset perform best, with LLaVA20mn-1ep achieving 80% partial precision and 64% complete precision.
- `arxiv:2410.22353v3#c01` (arxiv:2410.22353v3 — "RuleRAG: Rule-Guided Retrieval-Augmented Generation with Language Models for Question Answering", 2024, positive): RGFT in RuleRAG-FT substantially improves over the best RuleRAG-ICL performance.
- `arxiv:2410.22353v3#c05` (arxiv:2410.22353v3 — "RuleRAG: Rule-Guided Retrieval-Augmented Generation with Language Models for Question Answering", 2024, positive): RuleRAG-FT still shows substantial gains when retrieval is limited to top-1 with RGFT applied to retrievers and generators.
- `arxiv:2308.07107v5#c01` (arxiv:2308.07107v5 — "Large Language Models for Information Retrieval: A Survey", 2023, positive): Training on an unlabeled corpus at moderate scale substantially improves retrieval performance over the basic Llama-2 model.
- `arxiv:2308.07107v5#c02` (arxiv:2308.07107v5 — "Large Language Models for Information Retrieval: A Survey", 2023, positive): RepLLaMA brings major improvements on MSMARCO passage/doc retrieval and BEIR benchmarks.
- `arxiv:2410.11321v1#c05` (arxiv:2410.11321v1 — "Self-adaptive Multimodal Retrieval-Augmented Generation", 2024, positive): Fine-tuned embeddings improve relevant context retrieval, with R* outperforming R in Recall@N across all top-k values.
- `arxiv:2410.05983v1#c04` (arxiv:2410.05983v1 — "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG", 2024, positive): Increasing training data consistently improves accuracy for fine-tuned LLMs in RAG applications.
- `arxiv:2410.05983v1#c05` (arxiv:2410.05983v1 — "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG", 2024, positive): Data-augmented RAG fine-tuning leads to further improvements over implicit RAG fine-tuning, and implicit RAG fine-tuning outperforms no RAG-specific tuning and direct fine-tuning on Mistral-Nemo-12B-Chat.
- `arxiv:2410.05983v1#c06` (arxiv:2410.05983v1 — "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG", 2024, positive): Data-augmented RAG fine-tuning leads to further improvements over implicit RAG fine-tuning, and implicit RAG fine-tuning outperforms no RAG-specific tuning and direct fine-tuning on Gemini-1.0-Pro.
- `arxiv:2407.08223v2#c02` (arxiv:2407.08223v2 — "Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting", 2024, positive): Mixtral8x7B performance improves when paired with the instruction-tuned RAG drafter Drafter-7B.
- `arxiv:2407.08223v2#c05` (arxiv:2407.08223v2 — "Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting", 2024, negative): Finetuning the RAG drafter without the rationale component causes a significant performance drop across three benchmarks.
- `arxiv:2404.08695v2#c03` (arxiv:2404.08695v2 — "Enhancing Question Answering for Enterprise Knowledge Bases using Large Language Models", 2024, positive): The training strategy improves retrieval by selecting irrelevant documents from BM25 top-1000 results as negative pairs for each generated document-query pair.
- `arxiv:2410.22874v1#c03` (arxiv:2410.22874v1 — "Eliciting Critical Reasoning in Retrieval-Augmented Language Models via Contrastive Explanations", 2024, positive): Llama-2 models fine-tuned via C-RAG outperform GPT4-o by 6.1%.
- `arxiv:2407.07053v5#c04` (arxiv:2407.07053v5 — "Multimodal Self-Instruct: Synthetic Abstract Image and Visual Reasoning Instruction Using Language Model", 2024, positive): Training only on charts or tables can improve road map task performance by about 5%.
- `arxiv:2408.02545v1#c04` (arxiv:2408.02545v1 — "RAG Foundry: A Framework for Enhancing LLMs for Retrieval Augmented Generation", 2024, positive): After fine-tuning on PubmedQA, the RAG method appeared to perform best in both models.
- `arxiv:2405.07938v2#c01` (arxiv:2405.07938v2 — "EconLogicQA: A Question-Answering Benchmark for Evaluating Large Language Models in Economic Sequential Reasoning", 2024, positive): Instruction-tuned Llama-3-8B-Instruct outperforms Llama-3-8B, reaching 34.62% accuracy in 1-shot and 37.69% in 5-shot.
- `arxiv:2405.07938v2#c02` (arxiv:2405.07938v2 — "EconLogicQA: A Question-Answering Benchmark for Evaluating Large Language Models in Economic Sequential Reasoning", 2024, positive): Instruction-tuned Llama-3.1-8B-Instruct outperforms Llama-3.1-8B, reaching 36.15% accuracy in 1-shot and 36.92% in 5-shot.
- `arxiv:2409.12437v2#c06` (arxiv:2409.12437v2 — "Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data", 2024, positive): This adaptation is important for open LLMs in domain-specific settings to reach performance comparable to GPT-4o.
- `arxiv:2403.00827v1#c01` (arxiv:2403.00827v1 — "Self-Refinement of Language Models from External Proxy Metrics Feedback", 2024, positive): Fine-tuning llama-2-13b-chat on synthetic dialogue data generated by ProMiSe improves performance over zero-shot and supervised human-annotated baselines.
- `arxiv:2406.02030v2#c03` (arxiv:2406.02030v2 — "Multimodal Reasoning with Multimodal Knowledge Graph", 2024, positive): Pre-training on the MMKG-grounded dataset improves average accuracy by 0.42%.
- `arxiv:2407.01212v1#c04` (arxiv:2407.01212v1 — "EconNLI: Evaluating Large Language Models on Economics Reasoning", 2024, positive): Supervised fine-tuning on LLAMA2-chat models significantly improved results over encoder-only models.
- `arxiv:2408.11557v5#c02` (arxiv:2408.11557v5 — "Enhancing Spectral Knowledge Interrogation: A Reliable Retrieval-Augmented Generative Framework on Large Language Models", 2024, positive): The Llama3-8b model fine-tuned with LoRA performed best for response generation.
- `arxiv:2403.18405v3#c01` (arxiv:2403.18405v3 — "Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval", 2024, positive): Using the proposed workflow with a small amount of GPT-3.5-generated synthetic data significantly improves two 7B LLMs for legal relevance judgment, reaching or surpassing GPT-3.5.
- `arxiv:2403.18405v3#c02` (arxiv:2403.18405v3 — "Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval", 2024, positive): Training two 7B-scale LLMs on synthetic data with labels and interpretable reasoning can match or outperform GPT-3.5 on legal case relevance judgment.
- `arxiv:2412.08125v2#c01` (arxiv:2412.08125v2 — "Progressive Multi-granular Alignments for Grounded Reasoning in Large Vision-Language Models", 2024, positive): PromViL improves over Kosmos-2 in zero-shot settings on RefCOCOg and RefCOCO.
- `arxiv:2405.16506v3#c03` (arxiv:2405.16506v3 — "GRAG: Graph Retrieval-Augmented Generation", 2024, positive): Fine-tuning yields only marginal gains when GRAG is used on WebQSP, with Hit@1 increasing from 0.7236 to 0.7275.
- `arxiv:2405.11640v1#c02` (arxiv:2405.11640v1 — "Inquire, Interact, and Integrate: A Proactive Agent Collaborative Framework for Zero-Shot Multimodal Medical Reasoning", 2024, positive): With 5% of the training data, augmented training data outperforms MultiMedRes zero-shot prediction.
- `arxiv:2405.11640v1#c03` (arxiv:2405.11640v1 — "Inquire, Interact, and Integrate: A Proactive Agent Collaborative Framework for Zero-Shot Multimodal Medical Reasoning", 2024, mixed): With only 1% training data, the EKAID model with augmented training data achieves performance comparable to training on all data without chatlog.
- `arxiv:2405.11640v1#c04` (arxiv:2405.11640v1 — "Inquire, Interact, and Integrate: A Proactive Agent Collaborative Framework for Zero-Shot Multimodal Medical Reasoning", 2024, positive): Training EKAID only with difference questions moderately improves performance over the original baseline.
- `arxiv:2310.08975v3#c01` (arxiv:2310.08975v3 — "ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models", 2023, positive): ChatKBQA achieves strong KBQA results on WebQSP and CWQ, exceeding listed baselines in the comparison table.
- `arxiv:2310.08975v3#c03` (arxiv:2310.08975v3 — "ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models", 2023, positive): Fine-tuning Llama-2-7B without retrieval performs better in exact match than fine-tuning with retrieved results on WebQSP.
- `arxiv:2310.08975v3#c04` (arxiv:2310.08975v3 — "ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models", 2023, negative): T5 and Flan-T5 have much lower exact match than Llama-2-7B after fine-tuning.
- `arxiv:2310.08975v3#c05` (arxiv:2310.08975v3 — "ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models", 2023, positive): Fine-tuned open-source LLMs such as Llama-2-7B and ChatGLM2-6B show stronger semantic parsing ability than T5 and ChatGPT and generate higher-quality logical forms in EM and SM.

### h250 — Data-augmented fine-tuning recipes from factual-QA will transfer to low-resource domain-QA by improving sample efficiency, especially when augmentation creates evidence-grounded question-context pairs rather than generic instructions.

**Mechanism.** Synthetic or augmented training examples can expand supervision coverage and stabilize adaptation, letting domain-QA models approach full-data performance with a small fraction of labeled examples when the augmented data preserves domain evidence patterns.

**Predictions:**
- With 5% labels, augmented RAG fine-tuning beats standard fine-tuning.
- Evidence-grounded augmentation beats instruction-only augmentation.

**Minimal test.** On PubMedQA, train Llama-3-8B-Instruct under three 5%-label conditions: standard supervised fine-tuning, instruction-only synthetic augmentation, and data-augmented RAG fine-tuning that adds retrieved evidence to synthetic QA pairs; use the same base retriever for all conditions and compare against the zero-shot and full-label fine-tuning baselines using accuracy as the main metric and calibration error as a secondary metric.

**Scope.** method=synthetic or augmented fine-tuning with retrieved evidence, task=low-resource domain-QA with scarce labels and accessible corpora

**Evidence gap.** The payload shows augmentation benefits separately in factual-QA and domain-QA, but not whether factual-QA-style evidence-grounded augmentation is the best transfer recipe for low-resource domain-QA.

**Graph bridge.** fine-tuning → domain-QA

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.86 | 0.00 | 0.05 | 0.00 | 0.81 |

### Bridge opportunity a306 — bridge_opportunity

**Transfer question:** Could the effect on multi-step-reasoning with evaluation-method transfer to multi-step-reasoning with RAG?

**Shared entities:** method_from=evaluation-method, task_from=multi-step-reasoning, method_to=RAG, task_to=multi-step-reasoning, shared_tokens=multi, step

**Evidence claims:**
- `arxiv:2407.01046v2#c02` (arxiv:2407.01046v2 — "FRoG: Evaluating Fuzzy Reasoning of Generalized Quantifiers in Large Language Models", 2024, negative): GPT-4-turbo achieves below 50% accuracy on FRoG across masking settings.
- `arxiv:2403.03864v3#c05` (arxiv:2403.03864v3 — "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning", 2024, mixed): The overall random baseline is 26.4%.
- `arxiv:2404.05221v2#c06` (arxiv:2404.05221v2 — "LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models", 2024, negative): Manual evaluation found up to 39% false-positive reasoning chains for Llama-2-70B on StrategyQA.
- `arxiv:2311.17667v2#c01` (arxiv:2311.17667v2 — "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models", 2023, negative): GPT-4 achieves only 66.4% accuracy on implicit temporal reasoning, indicating difficulty with implicit temporal relationships.
- `arxiv:2311.17667v2#c02` (arxiv:2311.17667v2 — "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models", 2023, negative): Duration-conversion performance decreases compared to other atomic tasks.
- `arxiv:2311.17667v2#c04` (arxiv:2311.17667v2 — "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models", 2023, positive): In zero-shot settings, GPT-4 and GPT-3.5 outperform all open-source models.
- `arxiv:2311.17667v2#c05` (arxiv:2311.17667v2 — "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models", 2023, negative): All models except GPT-4 perform unsatisfactorily on symbolic temporal reasoning.
- `arxiv:2410.17558v2#c01` (arxiv:2410.17558v2 — "CLR-Bench: Evaluating Large Language Models in College-level Reasoning", 2024, negative): Qwen2.5-72b-Instruct scores much lower on Q→AR than on Q→A.
- `arxiv:2410.17558v2#c02` (arxiv:2410.17558v2 — "CLR-Bench: Evaluating Large Language Models in College-level Reasoning", 2024, mixed): qwen2.5-32b-instruct trails qwen2.5-72b-instruct on Q→A but outperforms it on Q→AR.
- `arxiv:2410.17558v2#c03` (arxiv:2410.17558v2 — "CLR-Bench: Evaluating Large Language Models in College-level Reasoning", 2024, negative): Across all three models, performance drops by over 30% on average from Q→A to Q→AR.
- `arxiv:2410.17558v2#c04` (arxiv:2410.17558v2 — "CLR-Bench: Evaluating Large Language Models in College-level Reasoning", 2024, negative): GPT-4-turbo performs strongly on Q→A but drops substantially on Q→AR.
- `arxiv:2306.08952v2#c01` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): Most models perform significantly worse on L3 than L2 questions in the ReasonQA setting, except ChatGPT.
- `arxiv:2306.08952v2#c02` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): F1 scores before 1900 and after 2020 are significantly worse than other time periods.
- `arxiv:2306.08952v2#c03` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): Performance on intra-year questions is significantly worse than on inter-year questions.
- `arxiv:2306.08952v2#c04` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): Performance in 2020-2040 is significantly worse than in other time periods for the first two LMs, indicating failure to generalize to the future.
- `arxiv:2306.08952v2#c05` (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): FLAN-T5-L has significantly lower EM scores than T5-L-NQ.
- `arxiv:2403.02615v2#c02` (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, positive): GPT-4 achieves nearly three times the accuracy of Llama2 7B.
- `arxiv:2403.02615v2#c04` (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, negative): LLaMA-7B and GPT-4 disagree on over 60% of questions.
- `arxiv:2403.02615v2#c05` (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, positive): Mistral 7B, ChatGPT, and GPT-4 show improved accuracy.
- `arxiv:2403.13315v3#c01` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, positive): Human participants achieved an average baseline score of 91.6% on sampled PuzzleVQA puzzles.
- `arxiv:2403.13315v3#c02` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, positive): GPT-4V scored 47.5% on the sampled puzzle set and was the highest-performing model there.
- `arxiv:2403.13315v3#c03` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, mixed): On single-concept abstract patterns in PuzzleVQA, model accuracy ranged from random-baseline level to 67.5%, with GPT-4V highest on average.
- `arxiv:2403.13315v3#c04` (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, mixed): On dual-concept abstract patterns in PuzzleVQA, model accuracy ranged from 26.4 to 45.5 on average, with GPT-4V highest on average.
- `arxiv:2411.12580v2#c04` (arxiv:2411.12580v2 — "Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models", 2024, negative): When reasoning, the model relies less on each individual document and shows less volatile total influence than when answering factual questions.
- `arxiv:2411.12580v2#c06` (arxiv:2411.12580v2 — "Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models", 2024, negative): Models rely less on individual documents for generating reasoning traces than for answering factual questions.
- `arxiv:2407.11963v3#c01` (arxiv:2407.11963v3 — "NeedleBench: Evaluating LLM Retrieval and Reasoning Across Varying Information Densities", 2024, mixed): Reasoning scores increase with model scale within some model series, but not universally.
- `arxiv:2407.11963v3#c04` (arxiv:2407.11963v3 — "NeedleBench: Evaluating LLM Retrieval and Reasoning Across Varying Information Densities", 2024, positive): Qwen-2.5-72B achieves the best result on the Multi-Needle Reasoning task, but remains below 50%.
- `arxiv:2411.01307v1#c03` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): MLLMs show zero-shot multimodal analogical reasoning capability on MBARD, with ChatGPT-4 achieving the best accuracy of 68.0%.
- `arxiv:2411.01307v1#c06` (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, mixed): An ablation variant removes the textual description of the relation from Explainer.
- `arxiv:2411.14465v1#c04` (arxiv:2411.14465v1 — "Testing Uncertainty of Large Language Models for Physics Knowledge and Reasoning", 2024, negative): For single-step and multi-step reasoning questions, GPT-3.5-turbo produces few correct replies despite high diversity.
- `arxiv:2407.01046v2#c04` (arxiv:2407.01046v2 — "FRoG: Evaluating Fuzzy Reasoning of Generalized Quantifiers in Large Language Models", 2024, negative): Models larger than 10B parameters have a larger average accuracy drop than models smaller than 10B on FRoG with fuzziness introduced by generalized quantifiers.
- `arxiv:2310.05845v1#c01` (arxiv:2310.05845v1 — "GraphLLM: Boosting Graph Reasoning Ability of Large Language Model", 2023, positive): With LLaMA 2 7B as backbone, the method shows relative improvements over GPT-4 few-shot CoT on four fundamental graph reasoning tasks.
- `arxiv:2310.05845v1#c02` (arxiv:2310.05845v1 — "GraphLLM: Boosting Graph Reasoning Ability of Large Language Model", 2023, positive): The method achieves 98.19% Exact Match Accuracy on average across four graph reasoning tasks, versus 47.35% for the top Graph2Text-based method.
- `arxiv:2310.05845v1#c03` (arxiv:2310.05845v1 — "GraphLLM: Boosting Graph Reasoning Ability of Large Language Model", 2023, positive): The method substantially reduces context length across graph reasoning tasks.
- `arxiv:2410.02338v2#c01` (arxiv:2410.02338v2 — "How Much Can RAG Help the Reasoning of LLM?", 2024, negative): RAG fails to improve reasoning capability when the effort required to filter noise exceeds its benefit.
- `arxiv:2410.02338v2#c04` (arxiv:2410.02338v2 — "How Much Can RAG Help the Reasoning of LLM?", 2024, positive): RAG can retrieve some information about a layer in the modeled reasoning process.
- `arxiv:2410.02338v2#c05` (arxiv:2410.02338v2 — "How Much Can RAG Help the Reasoning of LLM?", 2024, mixed): RAG can bypass the reasoning path and directly provide the corresponding answer.
- `arxiv:2402.11626v1#c06` (arxiv:2402.11626v1 — "Metacognitive Retrieval-Augmented Large Language Models", 2024, positive): MetaRAG significantly boosts reasoning accuracy in the two scenarios compared with the two other methods.
- `arxiv:2304.14732v7#c05` (arxiv:2304.14732v7 — "Search-in-the-Chain: Interactively Enhancing Large Language Models with Search for Knowledge-intensive Tasks", 2023, negative): IR can mislead the LLM when retrieved information conflicts with correctly memorized knowledge.
- `arxiv:2412.15235v1#c05` (arxiv:2412.15235v1 — "OG-RAG: Ontology-Grounded Retrieval-Augmented Generation For Large Language Models", 2024, positive): OG-RAG enables faster attribution to context and improves fact-based reasoning accuracy versus baseline methods.
- `arxiv:2412.12881v1#c04` (arxiv:2412.12881v1 — "RAG-Star: Enhancing Deliberative Reasoning with Retrieval Augmented Verification and Refinement", 2024, positive): RAG-Star significantly outperforms previous RAG and reasoning methods with Llama-3.1-8B-Instruct and GPT-4o.
- `arxiv:2412.02830v4#c03` (arxiv:2412.02830v4 — "RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models", 2024, positive): RARE shows its largest improvement over CoT on StrategyQA, a task where multi-step reasoning is crucial.
- `arxiv:2311.14740v1#c03` (arxiv:2311.14740v1 — "AutoKG: Efficient Automated Knowledge Graph Generation for Language Models", 2023, positive): Providing weather clues retrieved by AutoKG-based hybrid search enables GPT-4 to answer the question correctly.
- `arxiv:2312.01714v2#c01` (arxiv:2312.01714v2 — "Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models", 2023, positive): GPT-4V CoT-MM-Retrieval outperforms GPT-4V by 2.7% in zero-shot average score when using the problem image.
- `arxiv:2312.01714v2#c02` (arxiv:2312.01714v2 — "Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models", 2023, positive): GPT-4 CoT-MM-Retrieval surpasses Chameleon (GPT-4) by 6%, achieving an average score of 92.5%.
- `arxiv:2312.01714v2#c03` (arxiv:2312.01714v2 — "Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models", 2023, positive): ChatGPT CoT-MM-Retrieval outperforms the previous state-of-the-art Chameleon by 4.8%, reaching 84.7% average accuracy.
- `arxiv:2312.01714v2#c05` (arxiv:2312.01714v2 — "Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models", 2023, positive): The approach improves GPT-4 performance by 6% on ScienceQA and 12.9% on MathVista, and improves GPT-4V by 2.7% across two datasets.

### h084 — Importing perturbation-based evaluation such as masking and distractor sensitivity from reasoning benchmarks into RAG will identify robustness regimes where retrieval quality matters less than the model's ability to ignore semantically plausible but reasoning-irrelevant evidence.

**Mechanism.** Evaluation-method work uses masking or adversarial condition changes to stress latent reasoning; applied to RAG, analogous perturbations on retrieved context should quantify whether systems truly integrate evidence or are overly susceptible to noisy passages, conflicting snippets, and surface cues.

**Predictions:**
- Adversarially injected distractor passages will cause larger drops for vanilla RAG than for retrieval-filtered variants.
- Document influence variance will rise sharply on perturbed contexts for factual QA but only modestly for reasoning traces.

**Minimal test.** On FRoG and StrategyQA, build a RAG benchmark with three retrieval-context settings: clean top-5, top-5 plus two semantically similar distractors, and top-5 with one contradiction-injecting passage. Evaluate closed-book CoT, vanilla RAG, and a filtered RAG variant such as reranker-based or self-consistency filtered retrieval. Measure answer Accuracy/EM and document influence statistics per generated token using influence analysis; baseline is clean-context vanilla RAG.

**Scope.** method=masking/distractor perturbation evaluation and influence analysis transferred onto RAG context construction, task=multi-step reasoning with retrieved text where distractors or conflicting evidence can be programmatically injected

**Evidence gap.** RAG papers note noise sensitivity and some benefits from retrieval, while evaluation papers show perturbation-induced failures in reasoning, but no standardized perturbation benchmark directly connects these two lines.

**Graph bridge.** evaluation-method → RAG

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.89 | 0.90 | 0.00 | 0.05 | 0.00 | 0.82 |

## Evidence claims

- **arxiv:2302.00923v5#c06** (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, positive): With vision features, rationale generation RougeL rises to 93.46% and this corresponds to better answer accuracy of 85.31%.
- **arxiv:2304.14732v7#c05** (arxiv:2304.14732v7 — "Search-in-the-Chain: Interactively Enhancing Large Language Models with Search for Knowledge-intensive Tasks", 2023, negative): IR can mislead the LLM when retrieved information conflicts with correctly memorized knowledge.
- **arxiv:2305.12001v2#c01** (arxiv:2305.12001v2 — "OPT-R: Exploring the Role of Explanations in Finetuning and Prompting for Reasoning Skills of Large Language Models", 2023, positive): Incorporating explanations during finetuning and prompting benefits Numerical reasoning the most, with a +20.4% gain.
- **arxiv:2305.12001v2#c02** (arxiv:2305.12001v2 — "OPT-R: Exploring the Role of Explanations in Finetuning and Prompting for Reasoning Skills of Large Language Models", 2023, positive): Incorporating explanations during finetuning and prompting improves Analogical reasoning by +13.9%.
- **arxiv:2305.12001v2#c03** (arxiv:2305.12001v2 — "OPT-R: Exploring the Role of Explanations in Finetuning and Prompting for Reasoning Skills of Large Language Models", 2023, mixed): Some skills exhibit negligible or negative effects from incorporating explanations during finetuning and prompting.
- **arxiv:2306.08952v2#c01** (arxiv:2306.08952v2 — "Towards Benchmarking and Improving the Temporal Reasoning Capability of Large Language Models", 2023, negative): Most models perform significantly worse on L3 than L2 questions in the ReasonQA setting, except ChatGPT.
- **arxiv:2306.17820v4#c01** (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): Meta-Reasoning improves LLM performance by 27.0% on the Letter task with 1/2 demonstrations compared to Chain-of-Thought.
- **arxiv:2306.17820v4#c02** (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): Using only one demonstration on the Track(7) task leads to a 39.6% performance boost.
- **arxiv:2306.17820v4#c03** (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): Meta-Reasoning achieves higher accuracy on tasks where pure CoT struggles, with gains of 27.0% on Letter and 37.7% on Track.
- **arxiv:2306.17820v4#c04** (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): GPT-3 text-davinci-002 outperforms the other three LLMs on five datasets after adopting Meta-Reasoning.
- **arxiv:2306.17820v4#c05** (arxiv:2306.17820v4 — "Meta-Reasoning: Semantics-Symbol Deconstruction for Large Language Models", 2023, positive): Meta-Reasoning has lower error rates than Chain-of-Thought across all datasets shown in Table 4.
- **arxiv:2309.06275v4#c01** (arxiv:2309.06275v4 — "Re-Reading Improves Reasoning in Large Language Models", 2023, positive): davinci-003 with Vanilla+Re2 improves average performance on arithmetic, commonsense, and symbolic tasks over Vanilla.
- **arxiv:2309.06275v4#c03** (arxiv:2309.06275v4 — "Re-Reading Improves Reasoning in Large Language Models", 2023, positive): LLMs with Re2 achieve consistent improvements across davinci-003 and ChatGPT under both Vanilla and CoT prompting.
- **arxiv:2309.06275v4#c04** (arxiv:2309.06275v4 — "Re-Reading Improves Reasoning in Large Language Models", 2023, positive): On Llama-2 models, the re-reading mechanism enhances Vanilla and CoT performance across most tasks.
- **arxiv:2309.13339v4#c01** (arxiv:2309.13339v4 — "Enhancing Zero-Shot Chain-of-Thought Reasoning in Large Language Models through Logic", 2023, positive): GPT-4 showed improved accuracy on Date Understanding, LastLetter, and OddOneOut tasks with LoT revisions over default outputs.
- **arxiv:2310.01446v2#c04** (arxiv:2310.01446v2 — "Adaptive-Solver Framework for Dynamic Strategy Selection in Large Language Model Reasoning", 2023, mixed): Switching to a stronger LLM can improve performance but raises costs much more than prompting adaptation.
- **arxiv:2310.07018v1#c01** (arxiv:2310.07018v1 — "NEWTON: Are Large Language Models Capable of Physical Reasoning?", 2023, negative): GPT-4 shows strong scenario-based reasoning but is less consistent than humans on object-attribute reasoning.
- **arxiv:2310.07018v1#c04** (arxiv:2310.07018v1 — "NEWTON: Are Large Language Models Capable of Physical Reasoning?", 2023, positive): GPT-4 performs well on both boolean and multiple-choice questions in NEWTON.
- **arxiv:2311.08562v3#c02** (arxiv:2311.08562v3 — "MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration", 2023, positive): GPT-4 scored 90% in Judgment and 75.0% in Deception, indicating superiority in those scenarios.
- **arxiv:2311.17667v2#c02** (arxiv:2311.17667v2 — "TimeBench: A Comprehensive Evaluation of Temporal Reasoning Abilities in Large Language Models", 2023, negative): Duration-conversion performance decreases compared to other atomic tasks.
- **arxiv:2312.01714v2#c04** (arxiv:2312.01714v2 — "Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models", 2023, positive): Adding demonstration examples in the context improves overall accuracy, especially on ScienceQA and MathVista.
- **arxiv:2312.01714v2#c05** (arxiv:2312.01714v2 — "Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models", 2023, positive): The approach improves GPT-4 performance by 6% on ScienceQA and 12.9% on MathVista, and improves GPT-4V by 2.7% across two datasets.
- **arxiv:2312.14890v4#c03** (arxiv:2312.14890v4 — "NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes", 2023, mixed): Open-source models such as Yi-34b and Mistral-7b generalize better from harder few-shot examples than from simpler ones.
- **arxiv:2312.14890v4#c04** (arxiv:2312.14890v4 — "NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes", 2023, positive): Phi-1.5 on EDP can generalize better from easier few-shot examples than from harder ones at some difficulty levels.
- **arxiv:2401.06805v2#c01** (arxiv:2401.06805v2 — "Exploring the Reasoning Abilities of Multimodal Large Language Models (MLLMs): A Comprehensive Survey on Emerging Trends in Multimodal Reasoning", 2024, positive): InfiMM-LLaMA-13B shows improved reasoning abilities over LLaVA-1.5 across various benchmarks despite using the same instruction dataset.
- **arxiv:2401.10529v2#c01** (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, positive): GPT-4V with sequential input demonstrates the best reasoning capability among evaluated MLLMs for understanding image sequences, except being on par with Gemini and LLaVA-1.5 in behavior precision.
- **arxiv:2401.10529v2#c02** (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, negative): Video-LLaMA-2 and Chat-UniVi do not show an advantage over LLaVA-1.5 on this benchmark, and Video-LLaMA-2 performs notably worse.
- **arxiv:2401.10529v2#c03** (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, mixed): GPT-4V performs much better on reasoning about objects in image sequences than on reasoning about behaviors.
- **arxiv:2401.10529v2#c04** (arxiv:2401.10529v2 — "Mementos: A Comprehensive Benchmark for Multimodal Large Language Model Reasoning over Image Sequences", 2024, negative): GPT-4V scores only around 30% on behavior reasoning, with best recall barely exceeding 40%.
- **arxiv:2402.04236v3#c03** (arxiv:2402.04236v3 — "CogCoM: A Visual Language Model with Chain-of-Manipulations Reasoning", 2024, mixed): 5-shot prompting to GPT-4 yields stable solving steps, but makes manipulation and general-thinking descriptions similar.
- **arxiv:2402.08955v1#c05** (arxiv:2402.08955v1 — "Using Counterfactual Tasks to Evaluate the Generality of Analogical Reasoning in Large Language Models", 2024, mixed): GPT-4 achieved mean accuracy of 0.452 across all alphabets and problem types.
- **arxiv:2402.11574v1#c01** (arxiv:2402.11574v1 — "Visual In-Context Learning for Large Vision-Language Models", 2024, positive): VICL improves performance over zero-shot and standard ICL baselines, especially for LLaVA-13B.
- **arxiv:2402.11574v1#c03** (arxiv:2402.11574v1 — "Visual In-Context Learning for Large Vision-Language Models", 2024, positive): Token reduction enables including more demonstrations within the LVLM token budget.
- **arxiv:2402.11574v1#c04** (arxiv:2402.11574v1 — "Visual In-Context Learning for Large Vision-Language Models", 2024, mixed): In LVLM ICL, the model must infer task intent and an image parsing strategy from reference triplets before analyzing the target image.
- **arxiv:2402.13577v1#c01** (arxiv:2402.13577v1 — "BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models", 2024, positive): Bba surpasses all compared baseline methods on three tasks with relative performance gains.
- **arxiv:2402.13577v1#c02** (arxiv:2402.13577v1 — "BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models", 2024, positive): Bba improves GPT-4V(ision) performance on geometry problem solving from 28.34% to 34.22%.
- **arxiv:2402.13577v1#c03** (arxiv:2402.13577v1 — "BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models", 2024, positive): The Bba prompting method enhances GPT-4V(ision)'s multimodal reasoning by integrating DSL.
- **arxiv:2403.00816v3#c01** (arxiv:2403.00816v3 — "Read and Think: An Efficient Step-wise Multimodal Language Model for Document Understanding and Reasoning", 2024, positive): Step-wise generation improves performance over directly generated answers on InfoVQA and ChartQA.
- **arxiv:2403.00827v1#c01** (arxiv:2403.00827v1 — "Self-Refinement of Language Models from External Proxy Metrics Feedback", 2024, positive): Fine-tuning llama-2-13b-chat on synthetic dialogue data generated by ProMiSe improves performance over zero-shot and supervised human-annotated baselines.
- **arxiv:2403.02615v2#c01** (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, positive): GPT-4's accuracy increases by 13.9% with a step-by-step instruction.
- **arxiv:2403.02615v2#c03** (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, positive): Zero-shot chain-of-thought yields higher average accuracy than standard zero-shot across all models.
- **arxiv:2403.02615v2#c05** (arxiv:2403.02615v2 — "Exploring the Limitations of Large Language Models in Compositional Relation Reasoning", 2024, positive): Mistral 7B, ChatGPT, and GPT-4 show improved accuracy.
- **arxiv:2403.03864v3#c03** (arxiv:2403.03864v3 — "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning", 2024, positive): Guided vision substantially improves GPT-4V performance on some puzzles, indicating a visual perception bottleneck.
- **arxiv:2403.03864v3#c04** (arxiv:2403.03864v3 — "Are Language Models Puzzle Prodigies? Algorithmic Puzzles Unveil Serious Challenges in Multimodal Reasoning", 2024, positive): GPT-4V eCoT performs better on combinatorics, graphs, and sets than on optimization and search.
- **arxiv:2403.11381v2#c05** (arxiv:2403.11381v2 — "Can LLM-Augmented autonomous agents cooperate?, An evaluation of their cooperative capabilities through Melting Pot", 2024, negative): GPT-3.5 and GPT-4 found it challenging to interpret and reason about spatial information in a matrix-based state representation.
- **arxiv:2403.13315v3#c02** (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, positive): GPT-4V scored 47.5% on the sampled puzzle set and was the highest-performing model there.
- **arxiv:2403.13315v3#c06** (arxiv:2403.13315v3 — "PuzzleVQA: Diagnosing Multimodal Reasoning Challenges of Language Models with Abstract Visual Patterns", 2024, positive): LLaVA-13B and Gemini Pro improved most when guided jointly by visual perception, inductive reasoning, and deductive reasoning.
- **arxiv:2403.18405v3#c01** (arxiv:2403.18405v3 — "Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval", 2024, positive): Using the proposed workflow with a small amount of GPT-3.5-generated synthetic data significantly improves two 7B LLMs for legal relevance judgment, reaching or surpassing GPT-3.5.
- **arxiv:2404.05221v2#c02** (arxiv:2404.05221v2 — "LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models", 2024, positive): Breadth of search is generally more important than depth for efficient reasoning-space search on most tasks.
- **arxiv:2404.05221v2#c04** (arxiv:2404.05221v2 — "LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models", 2024, negative): Inappropriate prompt format design can lead to false-positive reasoning chains.
- **arxiv:2404.05221v2#c06** (arxiv:2404.05221v2 — "LLM Reasoners: New Evaluation, Library, and Analysis of Step-by-Step Reasoning with Large Language Models", 2024, negative): Manual evaluation found up to 39% false-positive reasoning chains for Llama-2-70B on StrategyQA.
- **arxiv:2404.06479v5#c01** (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, negative): GPT-4o has substantial failure rates on simple vector-graphics reasoning tasks.
- **arxiv:2404.06479v5#c02** (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, positive): GPT-4o performs better than VDLM-mm on some tasks such as Shapeworld Spatial Reasoning.
- **arxiv:2404.06479v5#c03** (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, negative): Perception errors contribute to VDLM-mm performing worse than GPT-4o on Shapeworld tasks.
- **arxiv:2404.06479v5#c04** (arxiv:2404.06479v5 — "Visually Descriptive Language Model for Vector Graphics Reasoning", 2024, positive): VDLM-mm yields consistent overall improvements over GPT-4V and GPT-4o on high-level visual reasoning tasks.
- **arxiv:2404.13985v2#c06** (arxiv:2404.13985v2 — "Information Re-Organization Improves Reasoning in Large Language Models", 2024, negative): Using GPT-3.5 for information re-organization with GPT-4 reasoning decreases reasoning ability.
- **arxiv:2405.00402v1#c04** (arxiv:2405.00402v1 — "Self-Refine Instruction-Tuning for Aligning Reasoning in Language Models", 2024, positive): GPT-3.5 has a more robust baseline performance among the teacher models.
- **arxiv:2405.01533v2#c01** (arxiv:2405.01533v2 — "OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning", 2024, positive): Using GPT-4 for counterfactual reasoning based on high-level decision making achieves good accuracy and interpretability.
- **arxiv:2405.05508v2#c04** (arxiv:2405.05508v2 — "Redefining Information Retrieval of Structured Database via Large Language Models", 2024, negative): Few-shot learning with black-box LLMs like GPT-4 may yield unsatisfactory accuracy.
- **arxiv:2405.07229v2#c01** (arxiv:2405.07229v2 — "MM-InstructEval: Zero-Shot Evaluation of (Multimodal) Large Language Models on Multimodal Reasoning Tasks", 2024, mixed): Flan-T5 series and Qwen2.5-VL open-source models show strong multimodal reasoning ability but still trail closed-source models.
- **arxiv:2405.11640v1#c02** (arxiv:2405.11640v1 — "Inquire, Interact, and Integrate: A Proactive Agent Collaborative Framework for Zero-Shot Multimodal Medical Reasoning", 2024, positive): With 5% of the training data, augmented training data outperforms MultiMedRes zero-shot prediction.
- **arxiv:2405.11640v1#c03** (arxiv:2405.11640v1 — "Inquire, Interact, and Integrate: A Proactive Agent Collaborative Framework for Zero-Shot Multimodal Medical Reasoning", 2024, mixed): With only 1% training data, the EKAID model with augmented training data achieves performance comparable to training on all data without chatlog.
- **arxiv:2405.13872v2#c01** (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, positive): Compared to GPT-4o, the model shows higher performance in Physical Relation, Object Location, and Spatial Relation categories.
- **arxiv:2405.13872v2#c02** (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, negative): Removing visual rationales and keeping only text rationales reduces performance on MMVet, especially in Knowledge and Spatial Awareness.
- **arxiv:2405.13872v2#c03** (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, positive): IoT improves performance on Cognition tasks, with gains reported for GPT-4o and Gemini-pro-1.5.
- **arxiv:2405.13872v2#c04** (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, positive): Compared with Gemini, IoT improves performance across more categories, with especially large gains in Action Recognition, Image Emotion, and Image Topic.
- **arxiv:2405.13872v2#c05** (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, mixed): Gemini-Pro's limited input image size makes it difficult to input a vision rationale series during MMVet experiments.
- **arxiv:2405.13872v2#c06** (arxiv:2405.13872v2 — "Image-of-Thought Prompting for Visual Reasoning Refinement in Multimodal Large Language Models", 2024, positive): For GPT-4o, IoT improves performance across many categories, especially when comparing different objects through reasoning.
- **arxiv:2406.01584v3#c01** (arxiv:2406.01584v3 — "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models", 2024, positive): SpatialRGPT outperforms the state of the art by over 20% accuracy compared to GPT-4V-Turbo.
- **arxiv:2406.01584v3#c02** (arxiv:2406.01584v3 — "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models", 2024, positive): SpatialRGPT-7B performs better than the SpatialRGPT-7B(rgb) variant, especially when relative depth helps resolve ambiguities.
- **arxiv:2406.01584v3#c03** (arxiv:2406.01584v3 — "SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models", 2024, positive): On BLINK Relative Depth results, SpatialRGPT-7B scores 82.3% and SpatialRGPT-VILA-1.5-8B scores 87.9%, above GPT-4V-Turbo's 66.9%.
- **arxiv:2406.02537v1#c02** (arxiv:2406.02537v1 — "TopViewRS: Vision-Language Models as Top-View Spatial Reasoners", 2024, negative): GPT-4V performs worse than Idefics-9B on both Static and Dynamic Spatial Reasoning tasks.
- **arxiv:2406.02537v1#c03** (arxiv:2406.02537v1 — "TopViewRS: Vision-Language Models as Top-View Spatial Reasoners", 2024, negative): Current state-of-the-art VLMs perform unsatisfactorily on TopViewRS, with model-wise average EM and PM below 50% over all tasks.
- **arxiv:2406.02537v1#c05** (arxiv:2406.02537v1 — "TopViewRS: Vision-Language Models as Top-View Spatial Reasoners", 2024, negative): Humans outperform GPT-4V by 47.78% on average.
- **arxiv:2406.03843v3#c02** (arxiv:2406.03843v3 — "POEM: Interactive Prompt Optimization for Enhancing Multimodal Reasoning of Large Language Models", 2024, mixed): The paper primarily focuses on visual and language modalities in multimodal LLM research.
- **arxiv:2406.09403v3#c03** (arxiv:2406.09403v3 — "Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models", 2024, positive): Sketchpad substantially improves performance over strong base models with no sketching on both math and vision tasks.
- **arxiv:2406.10621v3#c01** (arxiv:2406.10621v3 — "StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in Structure-Rich Text", 2024, negative): Qwen2-7B-Instruct has lower accuracy with Self-CoT and PS-CoT prompts than with other prompting methods, especially on XML and Tree structures.
- **arxiv:2406.10621v3#c02** (arxiv:2406.10621v3 — "StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in Structure-Rich Text", 2024, positive): Meta-Llama-3.1-70B-Instruct-Turbo improves accuracy when using the w/ Hint prompt instead of the Naive prompt.
- **arxiv:2406.10621v3#c05** (arxiv:2406.10621v3 — "StrucText-Eval: Evaluating Large Language Model's Reasoning Ability in Structure-Rich Text", 2024, positive): In a 3-shot scenario, GPT-4 substantially outperforms Gemini-Pro-Flash and Mistral.
- **arxiv:2406.10923v1#c01** (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, positive): Replacing BLIP-2 with Gemini improves F1 on TiM by 4.5.
- **arxiv:2406.10923v1#c02** (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, positive): Higher frame rate consistently improves performance over sparse sampling on TiM.
- **arxiv:2406.10923v1#c03** (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, positive): Current methods only marginally outperform a random baseline on TiM for abstract perception and long-range compositional reasoning.
- **arxiv:2406.10923v1#c04** (arxiv:2406.10923v1 — "Investigating Video Reasoning Capability of Large Language Models with Tropes in Movies", 2024, mixed): SOTA methods achieve at most 25 F1 on TiM and remain far below human performance.
- **arxiv:2406.11698v1#c01** (arxiv:2406.11698v1 — "Meta Reasoning for Large Language Models", 2024, positive): MRP attains the highest overall performance across 7 tasks with an average of 0.772.
- **arxiv:2406.11698v1#c02** (arxiv:2406.11698v1 — "Meta Reasoning for Large Language Models", 2024, positive): MRP outperforms comparison methods on BigToM and Code tasks.
- **arxiv:2406.11698v1#c04** (arxiv:2406.11698v1 — "Meta Reasoning for Large Language Models", 2024, mixed): MRP effectiveness depends on base model capability, performing satisfactorily with GPT-4 but suboptimally with GPT-3.5.
- **arxiv:2406.11698v1#c06** (arxiv:2406.11698v1 — "Meta Reasoning for Large Language Models", 2024, positive): MRP improves LLM ability on tasks requiring mixed reasoning strategies, especially for larger models like GPT-4.
- **arxiv:2406.11776v1#c03** (arxiv:2406.11776v1 — "Improving Multi-Agent Debate with Sparse Communication Topology", 2024, positive): Sparse MAD reduces token usage by 40.6% in multimodal reasoning, excluding input image tokens.
- **arxiv:2406.11776v1#c04** (arxiv:2406.11776v1 — "Improving Multi-Agent Debate with Sparse Communication Topology", 2024, positive): Placing the stronger LLM at a higher-centrality node leads to better performance than placing it at a lower-centrality node.
- **arxiv:2407.01046v2#c02** (arxiv:2407.01046v2 — "FRoG: Evaluating Fuzzy Reasoning of Generalized Quantifiers in Large Language Models", 2024, negative): GPT-4-turbo achieves below 50% accuracy on FRoG across masking settings.
- **arxiv:2407.01046v2#c04** (arxiv:2407.01046v2 — "FRoG: Evaluating Fuzzy Reasoning of Generalized Quantifiers in Large Language Models", 2024, negative): Models larger than 10B parameters have a larger average accuracy drop than models smaller than 10B on FRoG with fuzziness introduced by generalized quantifiers.
- **arxiv:2407.01525v3#c04** (arxiv:2407.01525v3 — "ScanReason: Empowering 3D Visual Grounding with Reasoning Capabilities", 2024, positive): The LLM-based 3D reasoning grounding baseline segments scene objects and converts their categories and 3D bounding boxes into text for InternLM2-7B.
- **arxiv:2407.04212v1#c01** (arxiv:2407.04212v1 — "Smart Vision-Language Reasoners", 2024, positive): SmartestVLM lr0.0003 achieves 34.7, a +48% gain over BERT+ResNet50 on the counting skill in SMART.
- **arxiv:2407.04212v1#c02** (arxiv:2407.04212v1 — "Smart Vision-Language Reasoners", 2024, positive): Including the QF layer representation improves accuracy on all skill sets.
- **arxiv:2407.04212v1#c04** (arxiv:2407.04212v1 — "Smart Vision-Language Reasoners", 2024, negative): Baseline models from Cherian et al. (2022) struggle on the SMART task, especially when employing transformers.
- **arxiv:2407.08029v1#c01** (arxiv:2407.08029v1 — "A Critical Review of Causal Reasoning Benchmarks for Large Language Models", 2024, negative): GPT-3.5 and GPT-4 memorized the Tübingen cause-effect dataset, or a large portion of it.
- **arxiv:2407.11511v3#c03** (arxiv:2407.11511v3 — "Multi-Step Reasoning with Large Language Models, a Survey", 2024, positive): With older LLMs such as GPT-3, reasoning approaches improve over standard prompting by 20–50 percentage points.
- **arxiv:2407.14507v3#c01** (arxiv:2407.14507v3 — "Internal Consistency and Self-Feedback in Large Language Models: A Survey", 2024, negative): GPT-4 exhibited self-contradictions at a rate of 15.7% in one experiment.
- **arxiv:2407.20564v1#c01** (arxiv:2407.20564v1 — "CLR-Fact: Evaluating the Complex Logical Reasoning Capability of Large Language Models over Factual Knowledge", 2024, mixed): Chain-of-thought prompting improves Precision@10 more as reasoning operation variety increases on GPT-3.5-turbo.
- **arxiv:2407.20564v1#c02** (arxiv:2407.20564v1 — "CLR-Fact: Evaluating the Complex Logical Reasoning Capability of Large Language Models over Factual Knowledge", 2024, positive): Demonstration selection yields an average performance improvement of 12-25% across two datasets.
- **arxiv:2408.00754v2#c04** (arxiv:2408.00754v2 — "Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Model", 2024, positive): The training-free Coarse Correspondences approach yields gains for GPT4-V/O across four spatial-temporal reasoning benchmarks.
- **arxiv:2408.04648v1#c01** (arxiv:2408.04648v1 — "PLUGH: A Benchmark for Spatial Understanding and Reasoning in Large Language Models", 2024, positive): In Graph Reconstruction, performance generally improves with more in-context examples.
- **arxiv:2408.04648v1#c02** (arxiv:2408.04648v1 — "PLUGH: A Benchmark for Spatial Understanding and Reasoning in Large Language Models", 2024, positive): In Novel Shortest Path, more shots reduce normalized Levenshtein distance.
- **arxiv:2408.04648v1#c03** (arxiv:2408.04648v1 — "PLUGH: A Benchmark for Spatial Understanding and Reasoning in Large Language Models", 2024, positive): In Temporal Hinted Shortest Path, more shots reduce normalized Levenshtein distance.
- **arxiv:2408.05093v4#c05** (arxiv:2408.05093v4 — "Order Matters in Hallucination: Reasoning Order as Benchmark and Reflexive Prompting for Large-Language-Models", 2024, positive): The method is particularly suited to reasoning tasks because it compares outputs from an order prompt pair.
- **arxiv:2408.08105v4#c01** (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, mixed): LLaVA-Next achieves the best reported results, but performance remains near random baseline on MuCR tasks.
- **arxiv:2408.08105v4#c02** (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, negative): Form-1 and Form-2 limit MLLMs' ability to recognize and use critical visual cues compared to Form-3.
- **arxiv:2408.08105v4#c03** (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, negative): Using Form-1, GPT-4o failed to incorporate a specific visual cue and chose an effect image based on abstract textual interpretation rather than direct visual correlation.
- **arxiv:2408.08105v4#c04** (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, positive): Form-3 gives GPT-4o direct visual information that helps it identify essential details across cause-and-effect images.
- **arxiv:2408.08105v4#c05** (arxiv:2408.08105v4 — "Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Discern Causal Links Across Modalities", 2024, positive): Form-3 improves GPT-4o's ability to establish causal links by allowing freer analysis of raw visual inputs.
- **arxiv:2408.15778v4#c01** (arxiv:2408.15778v4 — "LogicGame: Benchmarking Rule-Based Reasoning Abilities of Large Language Models", 2024, positive): Stronger models such as gpt-4o and qwen2-72b-instruct gain more AP-Acc from few-shot prompting than weaker models.
- **arxiv:2409.00106v1#c01** (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): Providing scene metadata in addition to the image yields a 2% improvement over the base LLM for GPT-4, but not for BLIP2.
- **arxiv:2409.00106v1#c03** (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): BLIP-2 Flan-T5 XXL generally achieves higher accuracy than BLIP-2 Flan-T5 XL on CLEVR and PTR regardless of prompting technique.
- **arxiv:2409.00106v1#c05** (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): GPT-4 is 17% more accurate than GPT-4V on CLEVR.
- **arxiv:2409.00106v1#c06** (arxiv:2409.00106v1 — "Zero-Shot Visual Reasoning by Vision-Language Models: Benchmarking and Analysis", 2024, positive): Flan-T5-XXL (11B) with standard prompting achieves the best performance and outperforms GPT-3.5-Turbo (175B).
- **arxiv:2409.09788v1#c01** (arxiv:2409.09788v1 — "Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models", 2024, positive): Using a reference object is associated with a higher success rate than not using one on Q-Spatial Bench.
- **arxiv:2409.09788v1#c02** (arxiv:2409.09788v1 — "Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models", 2024, negative): Prompting GPT-4V and GPT-4o with detailed procedures hurts performance on Q-Spatial Bench.
- **arxiv:2409.09788v1#c03** (arxiv:2409.09788v1 — "Reasoning Paths with Reference Objects Elicit Quantitative Spatial Reasoning in Large Vision-Language Models", 2024, positive): SpatialPrompt improves success rate across almost all VLMs on Q-Spatial Bench.
- **arxiv:2409.12437v2#c04** (arxiv:2409.12437v2 — "Enhancing Logical Reasoning in Large Language Models through Graph-based Synthetic Data", 2024, negative): A task-specific prompt causes performance drops in the few-shot setting.
- **arxiv:2409.13980v1#c01** (arxiv:2409.13980v1 — "Enhancing Advanced Visual Reasoning Ability of Large Language Models", 2024, positive): CVR-LLMLlama3 surpasses BLIP2 on WinoGAViL in the SWOW setting with 88.7% accuracy and a +17.1 improvement.
- **arxiv:2409.13980v1#c02** (arxiv:2409.13980v1 — "Enhancing Advanced Visual Reasoning Ability of Large Language Models", 2024, positive): The model outperforms MiniGPT4 with 62.0% accuracy and a +13.8 improvement on the GPT4-rated evaluation by Bitton-Guetta et al.
- **arxiv:2409.13980v1#c03** (arxiv:2409.13980v1 — "Enhancing Advanced Visual Reasoning Ability of Large Language Models", 2024, mixed): CVR-LLMLlama3 scores higher than LLaVA 1.5 on Whoops, VCR (Q->A), and NYCCC (Match).
- **arxiv:2409.14051v2#c02** (arxiv:2409.14051v2 — "GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion", 2024, positive): The 3+3 group strategy achieves the best accuracy in the experiments.
- **arxiv:2409.14051v2#c04** (arxiv:2409.14051v2 — "GroupDebate: Enhancing the Efficiency of Multi-Agent Debate Using Group Discussion", 2024, positive): Different group strategies improve accuracy and reduce token cost compared to not grouping.
- **arxiv:2409.17906v1#c01** (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo+1-shot increases accuracy over 0-shot on Edge count and Node degree tasks for small graphs.
- **arxiv:2409.17906v1#c02** (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo+1-shot improves accuracy over 0-shot on Connected components, MST, and Shortest path tasks for small graphs.
- **arxiv:2409.17906v1#c03** (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo+1-shot achieves the highest accuracy on Node count, Edge count, and Node degree tasks.
- **arxiv:2409.17906v1#c04** (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo+1-shot yields a large accuracy increase on the Edge count task for medium-sized graphs.
- **arxiv:2409.17906v1#c05** (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo significantly outperforms 0-shot on the Cycle check task when using GPT-3.5.
- **arxiv:2409.17906v1#c06** (arxiv:2409.17906v1 — "Graph Reasoning with Large Language Models via Pseudo-code Prompting", 2024, positive): Pseudo-code prompts improve performance across various graph tasks with GPT-3.5 and Mixtral.
- **arxiv:2410.01952v2#c01** (arxiv:2410.01952v2 — "TypedThinker: Diversify Large Language Model Reasoning with Typed Thinking", 2024, positive): TypedThinker improves performance across multiple benchmarks for Mistral 7B, LLaMA3 8B, and Qwen 2 7B on logical and mathematical reasoning tasks.
- **arxiv:2410.02203v4#c02** (arxiv:2410.02203v4 — "GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning", 2024, positive): GraphIC surpasses all training-free baselines and outperforms all training-based methods under GPT-4o-mini and Llama-3.
- **arxiv:2410.02203v4#c05** (arxiv:2410.02203v4 — "GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning", 2024, positive): Even when the reasoning process is incorrect, GraphIC still maintains high accuracy.
- **arxiv:2410.02203v4#c06** (arxiv:2410.02203v4 — "GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning", 2024, positive): GraphIC outperforms 10 baseline methods across mathematical reasoning, code generation, and logical reasoning tasks.
- **arxiv:2410.05983v1#c04** (arxiv:2410.05983v1 — "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG", 2024, positive): Increasing training data consistently improves accuracy for fine-tuned LLMs in RAG applications.
- **arxiv:2410.05983v1#c05** (arxiv:2410.05983v1 — "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG", 2024, positive): Data-augmented RAG fine-tuning leads to further improvements over implicit RAG fine-tuning, and implicit RAG fine-tuning outperforms no RAG-specific tuning and direct fine-tuning on Mistral-Nemo-12B-Chat.
- **arxiv:2410.09489v1#c01** (arxiv:2410.09489v1 — "Towards Efficient Visual-Language Alignment of the Q-Former for Visual Reasoning Tasks", 2024, positive): Applying LoRA to both the Q-Former and LLM achieves superior performance on both benchmarks with fewer than 12% of trainable parameters.
- **arxiv:2410.09489v1#c04** (arxiv:2410.09489v1 — "Towards Efficient Visual-Language Alignment of the Q-Former for Visual Reasoning Tasks", 2024, mixed): Applying LoRA to the Q-Former yields competitive performance, matching or surpassing full fine-tuning while using less than 2% of the original trainable parameters.
- **arxiv:2411.00387v3#c02** (arxiv:2411.00387v3 — "STEM-POM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing", 2024, mixed): GPT-4o outperformed smaller models, but errors remained high on challenging symbols.
- **arxiv:2411.01307v1#c01** (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Explainer+FLAVA improves over FLAVA by 6.9%-9.5% across all metrics.
- **arxiv:2411.01307v1#c02** (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Explainer+MKGformer performs better than MKGformer by 1.6%-2.6% in all five metrics.
- **arxiv:2411.01307v1#c04** (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Predictor(LLaVA) has accuracy close to ChatGPT-4 and significantly outperforms other methods on MBARD.
- **arxiv:2411.01307v1#c05** (arxiv:2411.01307v1 — "Can Multimodal Large Language Model Think Analogically?", 2024, positive): Predictor(VisualGLM) achieves 47.2% accuracy on MARS, significantly higher than other methods.
- **arxiv:2411.12580v2#c06** (arxiv:2411.12580v2 — "Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models", 2024, negative): Models rely less on individual documents for generating reasoning traces than for answering factual questions.
- **arxiv:2411.14465v1#c04** (arxiv:2411.14465v1 — "Testing Uncertainty of Large Language Models for Physics Knowledge and Reasoning", 2024, negative): For single-step and multi-step reasoning questions, GPT-3.5-turbo produces few correct replies despite high diversity.
- **arxiv:2412.08317v1#c04** (arxiv:2412.08317v1 — "Large Language Models Still Face Challenges in Multi-Hop Reasoning with External Knowledge", 2024, negative): Performance in non-sequential reasoning cases is lower than in sequential reasoning cases, with 56% accuracy versus 78%.
- **arxiv:2412.13540v3#c02** (arxiv:2412.13540v3 — "Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning", 2024, positive): Adding MCDGraph improves performance on QA samples across various tasks.
- **arxiv:2412.13540v3#c03** (arxiv:2412.13540v3 — "Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning", 2024, positive): Adding MCDGraph improves performance on FC samples across various tasks.
- **arxiv:2412.15238v2#c01** (arxiv:2412.15238v2 — "Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks", 2024, positive): Dipper with n=9 improves accuracy by about 10 percentage points over a single LLM baseline.
- **arxiv:2412.15238v2#c02** (arxiv:2412.15238v2 — "Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks", 2024, positive): An ensemble using seven different prompts outperforms both a self-ensemble without prompt variation and the average of seven single-prompt self-ensembles.
- **arxiv:2412.15238v2#c03** (arxiv:2412.15238v2 — "Dipper: Diversity in Prompts for Producing Large Language Model Ensembles in Reasoning tasks", 2024, positive): The full Dipper implementation with FASV achieves the highest accuracy among the self-ensemble baseline and other Dipper variants across ensemble sizes.
- **arxiv:2412.15296v1#c01** (arxiv:2412.15296v1 — "Confidence in the Reasoning of Large Language Models", 2024, negative): Second-answer accuracy is often worse than first-answer accuracy when models are asked to reconsider.
- **arxiv:2412.15296v1#c02** (arxiv:2412.15296v1 — "Confidence in the Reasoning of Large Language Models", 2024, negative): Changing their mind is associated with significantly worse accuracy for Mistral, reaching 36% and 32%, below the target value.
- **arxiv:2412.16599v1#c01** (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, negative): LLaVA-7B and LLaVA-13B achieve very low accuracy on relative compass direction reasoning, lower than on other tasks.
- **arxiv:2412.16599v1#c02** (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, negative): Claude-3 series and GPT-4o-mini perform poorly on relative compass direction reasoning, with accuracy mostly between 10% and 20%.
- **arxiv:2412.16599v1#c05** (arxiv:2412.16599v1 — "Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning", 2024, positive): On absolute spatial direction reasoning, LLaVA-13B performs slightly better than LLaVA-7B, improving from 14.08% to 25.72% accuracy.
- **arxiv:2412.17970v1#c01** (arxiv:2412.17970v1 — "CARL-GT: Evaluating Causal Reasoning Capabilities of Large Language Models", 2024, positive): Llama3-8B and Mistral-7B outperform Qwen2-7B and Gemma2-9B on d-separation estimation.
- **arxiv:2412.17970v1#c02** (arxiv:2412.17970v1 — "CARL-GT: Evaluating Causal Reasoning Capabilities of Large Language Models", 2024, positive): All methods achieve at least AUC 0.5 on d-separation estimation.
