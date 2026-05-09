# Selected Hypotheses

Selected **10** hypotheses across **9** anomalies.

Exploratory report: no synthesized insights were supported yet, so the sections below show claim-level evidence and candidate explanations rather than a settled takeaway.

## Conflict Hypotheses

### Anomaly a280 — community_disconnect

**Central question:** Could safety and web be connected by a shared mechanism?

**Shared entities:** community_from=safety, community_to=web, shared_concepts=evaluation protocol, modality alignment, reliability risk, safety risk, text, text + image

**Evidence claims:**
- `arxiv:2603.22862v2#c01` (arxiv:2603.22862v2 — "The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration", 2026, positive): Adversarial tool chains used as negative samples in alignment fine-tuning enhance LLM resilience against interference and improve robustness of parameter extraction.
- `arxiv:2503.21460v1#c03` (arxiv:2503.21460v1 — "Large Language Model Agent: A Survey on Methodology, Applications and Challenges", 2025, positive): Netsafe identifies safety phenomena and topological properties that affect the safety of multi-agent networks under adversarial attacks.
- `arxiv:2402.08567#c01` (arxiv:2402.08567 — "Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast", 2024, mixed): The paper evaluates infectious jailbreak using cumulative/current infection ratio over chat rounds.
- `arxiv:2402.08567#c02` (arxiv:2402.08567 — "Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast", 2024, mixed): The study measures when infection reaches 90% and the infection ratio at the 16th chat round.
- `arxiv:2402.08567#c03` (arxiv:2402.08567 — "Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast", 2024, mixed): The infection ratio is analyzed under different ensemble sample sizes M.
- `arxiv:2402.08567#c04` (arxiv:2402.08567 — "Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast", 2024, mixed): The infection ratio is evaluated under image corruptions including Flip, Resize, and JPEG.
- `arxiv:2402.08567#c05` (arxiv:2402.08567 — "Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast", 2024, mixed): The infection dynamics are evaluated when using InstructBLIP 7B as the multimodal LLM.
- `arxiv:2404.07362#c03` (arxiv:2404.07362 — ""We Need Structured Output": Towards User-centered Constraints on Large Language Model Output", 2024, positive): Semantic constraints that exclude specific terms, items, or actions accounted for 8.2% of cases.
- `arxiv:2405.06211v3#c03` (arxiv:2405.06211v3 — "A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models", 2024, negative): Trustworthy RA-LLM systems should be robust to malicious or inadvertent perturbations from attackers.
- `arxiv:2410.07283#c01` (arxiv:2410.07283 — "Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems", 2024, mixed): GPT-4o is more resistant to prompt injections than GPT-3.5, but becomes a stronger attacker once compromised.
- `arxiv:2410.07283#c02` (arxiv:2410.07283 — "Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems", 2024, positive): Self-Replicating infection has a higher success rate for GPT-4o and is much more effective for GPT-3.5.
- `arxiv:2410.07283#c03` (arxiv:2410.07283 — "Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems", 2024, positive): GPT-4o ignores 66% of self-replicating attacks and 54% of non-replicating attacks, indicating greater robustness.
- `arxiv:2410.07283#c04` (arxiv:2410.07283 — "Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems", 2024, negative): GPT-3.5 ignores only 9% and 20% of attacks, respectively.
- `arxiv:2410.07283#c05` (arxiv:2410.07283 — "Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems", 2024, positive): GPT-4o’s higher precision makes it more dangerous once compromised.
- `arxiv:2307.06435#c05` (arxiv:2307.06435 — "A Comprehensive Overview of Large Language Models", 2023, positive): Model safety improves with fine-tuning on safe demonstrations, and an additional RLHF step further improves safety and reduces jailbreak susceptibility.
- `arxiv:2308.11432v7#c03` (arxiv:2308.11432v7 — "A Survey on Large Language Model based Autonomous Agents", 2023, positive): Existing powerful LLMs including ChatGPT and GPT-4 are mostly aligned with unified human values.
- `arxiv:2309.05463#c01` (arxiv:2309.05463 — "Textbooks Are All You Need II: phi-1.5 technical report", 2023, mixed): phi-1.5's textbook-like synthetic training data appears to reduce toxic content generation compared with internet-only trained models.
- `arxiv:2309.05463#c03` (arxiv:2309.05463 — "Textbooks Are All You Need II: phi-1.5 technical report", 2023, positive): The absence of web data is associated with improvement in toxic and biased generation behavior for phi-1.5.
- `arxiv:2401.01614#c01` (arxiv:2401.01614 — "GPT-4V(ision) is a Generalist Web Agent, if Grounded", 2024, positive): Choice with GPT-4V outperforms GPT-4 and FLAN-T5-XL by over 20% whole task success rate across all three settings.
- `arxiv:2401.01614#c02` (arxiv:2401.01614 — "GPT-4V(ision) is a Generalist Web Agent, if Grounded", 2024, positive): GPT-4 outperforms FLAN-T5-XL by 4.4% whole task success rate in online evaluation, despite worse offline step success rate.
- `arxiv:2401.01614#c03` (arxiv:2401.01614 — "GPT-4V(ision) is a Generalist Web Agent, if Grounded", 2024, positive): Choice with GPT-4V has a substantial performance advantage over text-only GPT-4 under all three metrics across all three test splits.
- `arxiv:2401.01614#c04` (arxiv:2401.01614 — "GPT-4V(ision) is a Generalist Web Agent, if Grounded", 2024, positive): GPT-4V can successfully complete 51.1% of live website tasks when its textual plans are manually grounded into website actions.
- `arxiv:2401.13919#c01` (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, positive): WebVoyager achieves a 59.1% task success rate on the benchmark and outperforms GPT-4 (All Tools) and the text-only setting.
- `arxiv:2401.13919#c02` (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, positive): WebVoyager outperforms text-only and GPT-4 (All Tools) baselines by large margins in most website tasks.
- `arxiv:2401.13919#c03` (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, mixed): WebVoyager is slightly worse than the text-only setting on Allrecipes and similar on Github, ESPN, Cambridge Dictionary, and Wolfram Alpha.
- `arxiv:2401.13919#c04` (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, positive): The three backbone models GPT-4V, Claude 3 Opus, and GPT-4o have relatively close automatic evaluation results, and all perform significantly better than the text-only setting with GPT-4 backbone.
- `arxiv:2401.13919#c05` (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, positive): The evaluation protocol achieves 85.3% agreement with human judges, suggesting GPT-4V is a reliable evaluator for online agents.
- `arxiv:2404.03648v2#c01` (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, negative): The system falters on advanced web applications such as maps, animations, and video browsing.
- `arxiv:2404.03648v2#c02` (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, negative): Hallucinations are the largest error category in web task automation at 44%.
- `arxiv:2404.03648v2#c03` (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, mixed): Prediction is the largest component of system execution time at 47.89%.
- `arxiv:2404.03648v2#c05` (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, positive): AutoWebGLM is claimed in the abstract to outperform GPT-4 as an automated web navigation agent.
- `arxiv:2405.20309v2#c01` (arxiv:2405.20309v2 — "Large Language Models Can Self-Improve At Web Agent Tasks", 2024, mixed): Self-improvement may reinforce biases in a base model.
- `arxiv:2405.20309v2#c02` (arxiv:2405.20309v2 — "Large Language Models Can Self-Improve At Web Agent Tasks", 2024, mixed): A rapidly changing environment is a challenging scenario for self-improvement because the model may need to adapt online to evolving tasks.
- `arxiv:2412.05467#c01` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, positive): GPT-4o performance improved over workarena-plus-plus measurements on WebArena and WorkArena L2.
- `arxiv:2412.05467#c02` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, positive): Claude improves substantially on WorkArena-L2 and achieves a 39.1% average task success rate.
- `arxiv:2412.05467#c03` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, mixed): Task validation uses logic rules, exact matching, or semantic matching with GPT-3.5 as a judge, which implies some costs.
- `arxiv:2412.05467#c04` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, positive): Claude-3.5-Sonnet achieved 39.1% success on WorkArena L2, compared with 8.5% for GPT-4o.
- `arxiv:2412.05467#c05` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, mixed): Claude-3.5-Sonnet leads on almost all benchmarks, except vision-related tasks where GPT-4o is superior.
- `arxiv:2307.13854v4#c01` (arxiv:2307.13854v4 — "WebArena: A Realistic Web Environment for Building Autonomous Agents", 2023, negative): GPT-4 with chain-of-thought prompting achieves 11.70% end-to-end task success, far below human performance of 78.24%.
- `arxiv:2307.13854v4#c02` (arxiv:2307.13854v4 — "WebArena: A Realistic Web Environment for Building Autonomous Agents", 2023, negative): GPT-3.5 with chain-of-thought prompting successfully performs only 8.75% of tasks.
- `arxiv:2307.13854v4#c03` (arxiv:2307.13854v4 — "WebArena: A Realistic Web Environment for Building Autonomous Agents", 2023, negative): text-bison-001 underperforms GPT-3.5, with a success rate of 5.05%.
- `arxiv:2307.13854v4#c04` (arxiv:2307.13854v4 — "WebArena: A Realistic Web Environment for Building Autonomous Agents", 2023, negative): Across 61 templates, GPT-4 achieves 100% task success on only four templates, while GPT-3.5 achieves full completion on none.

### h074 — Venue separation persists because safety emphasizes adversarial stress-testing and alignment interventions, whereas web work emphasizes capability and benchmark progress; cross-pollination would look like importing adversarial red-teaming fine-tuning into web agents and importing web-style online evaluators into safety agent benchmarks.

**Mechanism.** High recent activity with balanced impact suggests both communities are active, but each optimizes different leaderboard targets and reviewer expectations, so tools that look central in one venue appear auxiliary in the other.

**Predictions:**
- Adversarial fine-tuning lowers web-agent cascading error rate
- Web-style judge protocols increase consistency of safety agent evaluation

**Minimal test.** Take a web agent baseline and fine-tune or preference-tune it on adversarial tool-chain or injected webpage negatives drawn from safety setups, then measure task success and attack robustness; separately, apply GPT-4V/GPT-3.5 judge-based online evaluation protocols from web benchmarks to a safety multi-agent prompt-injection benchmark and compare agreement with human raters.

**Scope.** method=fine-tuning or RLHF plus judge-based evaluation, task=agentic benchmarks with observable trajectories and action outcomes

**Evidence gap.** There are few direct replications showing whether alignment data that improves safety also improves web robustness without harming capability, or whether web evaluator protocols validly score safety incidents.

**Graph bridge.** evaluation protocol → text + image

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.92 | 0.96 | 1.00 | 0.87 | 0.00 | 0.98 |

### Anomaly a037 — impact_conflict

**Central question:** Why do high-impact papers disagree about RAG on long-context-QA?

**Shared entities:** method=RAG, task=long-context-QA

**Evidence claims:**
- `arxiv:2502.01549v1#c01` (arxiv:2502.01549v1 — "VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos", 2025, positive): VideoRAG has a higher overall win rate than NaiveRAG across all videos.
- `arxiv:2502.01549v1#c02` (arxiv:2502.01549v1 — "VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos", 2025, positive): VideoRAG has a higher overall win rate than GraphRAG-l across all videos.
- `arxiv:2502.01549v1#c03` (arxiv:2502.01549v1 — "VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos", 2025, positive): VideoRAG has a higher overall win rate than GraphRAG-g across all videos.
- `arxiv:2502.01549v1#c04` (arxiv:2502.01549v1 — "VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos", 2025, positive): VideoRAG has a higher overall win rate than LightRAG across all videos.
- `arxiv:2502.01549v1#c06` (arxiv:2502.01549v1 — "VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos", 2025, positive): The system outperforms state-of-the-art large vision models for long-context video understanding on overall score.
- `arxiv:2403.05530#c03` (arxiv:2403.05530 — "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context", 2024, positive): Full-context provides better answers than retrieval-augmented generation with 4k tokens in 78% of cases.
- `arxiv:2407.02485v1#c03` (arxiv:2407.02485v1 — "RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs", 2024, negative): A shorter top-k context list usually yields higher generation accuracy than reading too many chunked contexts such as top-100.
- `arxiv:2501.00663#c01` (arxiv:2501.00663 — "Titans: Learning to Memorize at Test Time", 2024, negative): Titans outperforms a RAG-augmented Llama3.1-8B model on the BABILong benchmark despite having about 70 fewer parameters.

### h164 — The contradiction is moderated by retrieval recall and top-k setting: RAG appears strong when the retriever surfaces a small high-recall set, but weak when relevant evidence is missed or when too many chunks are passed to the generator.

**Mechanism.** Generation quality depends non-monotonically on retrieved context size. If top-k is too small, key evidence is absent; if too large, the model is distracted by redundant/noisy chunks. A tuned retrieval stack in VideoRAG can outperform alternatives, while generic RAG with poor recall or oversized top-k can underperform full-context or non-RAG baselines.

**Predictions:**
- Accuracy peaks at moderate top-k
- Contradiction fades when recall@k is matched

**Minimal test.** Re-run the compared systems with a shared retriever and identical chunking, then sweep top-k (e.g., 1, 5, 10, 20, 100) while measuring retrieval recall@k and answer accuracy; if the papers disagree only because they operated at different recall/top-k points, their performance ordering should converge at the same recall-controlled setting.

**Scope.** method=RAG, task=long-context-QA with retrieved chunk inputs

**Evidence gap.** The claim set lacks consistent reporting of retriever quality, chunk granularity, and top-k, preventing direct normalization of the retrieval operating point.

**Graph bridge.** arxiv:2407.02485v1#c03 → arxiv:2502.01549v1#c02

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.95 | 1.00 | 0.83 | 0.00 | 0.97 |

### Anomaly a014 — impact_conflict

**Central question:** Why do high-impact papers disagree about chain-of-thought on multi-step-reasoning?

**Shared entities:** method=chain-of-thought, task=multi-step-reasoning

**Evidence claims:**
- `arxiv:2601.07780v1#c06` (arxiv:2601.07780v1 — "Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection", 2026, positive): The increased computational cost of the approach is justified in applications requiring accurate, robust, and trustworthy reasoning.
- `arxiv:2501.06186v1#c01` (arxiv:2501.06186v1 — "LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs", 2025, positive): Curriculum learning combined with multi-step chain-of-thought gives a 9.14% absolute gain over the base Llama-3.2-11B-Vision-Inst model.
- `arxiv:2502.20808v6#c01` (arxiv:2502.20808v6 — "MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts", 2025, positive): Adding these strategies improved GPT-4o performance on multi-step free-form questions from 25.4% to 32.6%.
- `arxiv:2503.16419v4#c01` (arxiv:2503.16419v4 — "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models", 2025, positive): Pretraining recipes for reasoning-capable models often encourage extended reasoning steps to improve accuracy, making the overthinking challenge difficult to address.
- `arxiv:2503.16419v4#c03` (arxiv:2503.16419v4 — "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models", 2025, positive): Models tend to perform better with extended reasoning steps, implying CoT length is important for effective problem-solving.
- `arxiv:2504.15279v1#c01` (arxiv:2504.15279v1 — "VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models", 2025, positive): GPT-4o-mini gains only 1.2 points from chain-of-thought prompting versus direct-answer prompts.
- `arxiv:2401.04925v4#c01` (arxiv:2401.04925v4 — "The Impact of Reasoning Step Length on Large Language Models", 2024, positive): For few-shot chain-of-thought prompting, step count has a direct linear correlation with accuracy.
- `arxiv:2401.04925v4#c02` (arxiv:2401.04925v4 — "The Impact of Reasoning Step Length on Large Language Models", 2024, positive): Chain length appears more important than factual accuracy for effective problem-solving.
- `arxiv:2401.03991v1#c01` (arxiv:2401.03991v1 — "Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark", 2024, positive): On revised StepGame, GPT-4 with CoT outperforms GPT-4 base across most hop settings.
- `arxiv:2401.03991v1#c04` (arxiv:2401.03991v1 — "Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark", 2024, positive): The customized CoT method shows greater advantages in larger models such as GPT-4 and Davinci, maintaining accuracy as task complexity increases.
- `arxiv:2411.10440v6#c03` (arxiv:2411.10440v6 — "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step", 2024, positive): LLaVA-CoT uses sequential stages of summarization, visual interpretation, logical reasoning, and conclusion generation, unlike chain-of-thought prompting.
- `arxiv:2302.00923v5#c01` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, positive): Multimodal-CoT Base achieves higher accuracy than the language-only baseline.
- `arxiv:2302.00923v5#c02` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, positive): Two-Stage Multimodal reaches higher accuracy than Two-Stage Baseline by the final reported epoch.
- `arxiv:2302.00923v5#c03` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, positive): Multimodal-CoT demonstrates effective generalization to MMMU, outperforming some larger ~8B models.
- `arxiv:2303.13988#c03` (arxiv:2303.13988 — "Machine Psychology", 2023, positive): Zero-shot chain-of-thought prompting improves reasoning performance.
- `arxiv:2307.06435#c04` (arxiv:2307.06435 — "A Comprehensive Overview of Large Language Models", 2023, positive): Fine-tuning with chain-of-thought improves performance on held-out tasks.
- `arxiv:2309.04461v2#c01` (arxiv:2309.04461v2 — "Measuring and Improving Chain-of-Thought Reasoning in Vision-Language Models", 2023, positive): CoTBLIP achieves the highest reasoning performance among listed VLMs on the reported Ro metric.
- `arxiv:2309.04461v2#c02` (arxiv:2309.04461v2 — "Measuring and Improving Chain-of-Thought Reasoning in Vision-Language Models", 2023, positive): CoTBLIP improves reasoning performance and consistency by approximately 4% relative to the SOTA.
- `arxiv:2309.04461v2#c05` (arxiv:2309.04461v2 — "Measuring and Improving Chain-of-Thought Reasoning in Vision-Language Models", 2023, positive): CoTBLIP can generate reasoning chains that help high-level visual inference and lead to correct answers in cases where BLIP-2 initially fails.
- `arxiv:2601.07780v1#c05` (arxiv:2601.07780v1 — "Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection", 2026, negative): Chain-of-Thought prompting still faces consistency, accuracy, and self-correction challenges on complex or ethically sensitive tasks.
- `arxiv:2502.17419v6#c02` (arxiv:2502.17419v6 — "From System 1 to System 2: A Survey of Reasoning Large Language Models", 2025, negative): Foundational LLMs excel at fast decision-making but lack depth for complex reasoning.
- `arxiv:2502.20808v6#c04` (arxiv:2502.20808v6 — "MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts", 2025, mixed): Claude improves with CoT and with 2-shot based on CoT, while GPT4o performs best under CoT with 2-shot and Qwen-vl-max and gpt-4v perform best under CoT.
- `arxiv:2401.03991v1#c03` (arxiv:2401.03991v1 — "Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark", 2024, mixed): For the Turbo model, CoT improves performance as k increases, but less than for Davinci and GPT-4.
- `arxiv:2407.11373#c02` (arxiv:2407.11373 — "Reliable Reasoning Beyond Natural Language", 2024, negative): GPT-4 text-only chain-of-thought performance drops as variable interdependence increases, reaching 0% on problems with four interdependent variables.
- `arxiv:2411.00836v2#c05` (arxiv:2411.00836v2 — "DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models", 2024, mixed): 3-shot chain-of-thought slightly improves GPT-4o but reduces performance for Claude-3.5 and Gemini Pro 1.5.
- `arxiv:2411.10442v2#c05` (arxiv:2411.10442v2 — "Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization", 2024, negative): Chain-of-Thought reasoning reduces InternVL2-8B performance on MathVista compared with direct answers.
- `arxiv:2302.00923v5#c05` (arxiv:2302.00923v5 — "Multimodal Chain-of-Thought Reasoning in Language Models", 2023, negative): Generated rationale in the two-stage framework does not improve answer accuracy compared with the QCM A variant.
- `arxiv:2302.07842#c02` (arxiv:2302.07842 — "Augmented Language Models: a Survey", 2023, mixed): Chain-of-thought reasoning from a few in-context examples emerges only at sufficient model scale.
- `arxiv:2302.07842#c03` (arxiv:2302.07842 — "Augmented Language Models: a Survey", 2023, mixed): Chain-of-thought few-shot performance depends strongly on example format, choice, and order.

### h262 — The sign of chain-of-thought effects flips with dataset composition, especially whether benchmarks reward decomposable stepwise reasoning versus tightly interdependent or ambiguity-sensitive reasoning.

**Mechanism.** On tasks whose latent solution can be factorized into mostly independent substeps, explicit intermediate reasoning scaffolds search and raises accuracy; on datasets with high variable interdependence, misleading cues, or ethical ambiguity, longer chains amplify early mistakes and consistency failures, turning gains into losses.

**Predictions:**
- CoT gains are positive on low-interdependence subsets
- CoT gains vanish or reverse on high-interdependence subsets

**Minimal test.** Construct matched evaluation slices within one benchmark family that differ only in interdependence/compositional depth while keeping model, prompt template, decoding, and scorer fixed; compare direct-answer versus CoT deltas on each slice and check whether papers' contradictory signs disappear after conditioning on slice type.

**Scope.** method=chain-of-thought, task=multi-step-reasoning

**Evidence gap.** Most claims do not report a common decomposition of item difficulty, interdependence, or ambiguity across datasets, so cross-paper sign changes cannot yet be normalized by composition.

**Graph bridge.** chain-of-thought → multi-step-reasoning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.96 | 1.00 | 0.80 | 0.00 | 0.97 |

### h263 — The contradiction is driven by prompt format differences—zero-shot trigger phrases, few-shot exemplars, and rationale structure can switch CoT from helpful guidance to harmful prompt overconstraint.

**Mechanism.** CoT performance is highly sensitive to whether prompts use zero-shot 'think step by step', 2-shot/3-shot exemplars, customized reasoning templates, or multimodal staged instructions; good formats elicit latent reasoning, while mismatched exemplars or verbose templates consume context, bias solution paths, or induce imitation of wrong reasoning, reversing the effect relative to direct answers.

**Predictions:**
- A single standardized CoT template reduces between-paper variance
- Reordering or replacing exemplars changes the CoT delta sign

**Minimal test.** Run the same model-task pairs from the conflicting papers under a factorial prompt ablation: direct answer, zero-shot CoT trigger, fixed 2-shot CoT exemplars, fixed 3-shot CoT exemplars, and the paper-specific customized template; keep dataset, decoding, and evaluation fixed, then test whether contradiction disappears once prompt format is held constant.

**Scope.** method=chain-of-thought, task=multi-step-reasoning

**Evidence gap.** Conflicting papers rarely release fully matched prompt text, exemplar order, or rationale formatting, preventing direct attribution of sign changes to prompt structure alone.

**Graph bridge.** chain-of-thought → GPT-4o

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.85 | 0.97 | 1.00 | 0.80 | 0.00 | 0.97 |

### Anomaly a065 — impact_conflict

**Central question:** Why do high-impact papers disagree about long-context on long-context-QA?

**Shared entities:** method=long-context, task=long-context-QA

**Evidence claims:**
- `arxiv:2401.07883#c05` (arxiv:2401.07883 — "The Chronicles of RAG: The Retriever, the Chunk and the Generator", 2024, positive): GPT-4-1106-preview can process up to 128k input tokens, compared with GPT-1 and GPT-2 models handling up to 1024 tokens.
- `arxiv:2403.05530#c02` (arxiv:2403.05530 — "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context", 2024, positive): Gemini 1.5 shows better recall than GPT-4 Turbo up to 128K tokens.
- `arxiv:2501.00663#c02` (arxiv:2501.00663 — "Titans: Learning to Memorize at Test Time", 2024, positive): Titans can scale beyond a 2M context window and achieve better accuracy than baselines on needle-in-haystack tasks.
- `arxiv:2402.19473#c03` (arxiv:2402.19473 — "Retrieval-Augmented Generation for AI-Generated Content: A Survey", 2024, mixed): Prompt compression and long-context support partially mitigate lengthy-context challenges, with a slight trade-off in accuracy or costs.
- `arxiv:2407.11005#c03` (arxiv:2407.11005 — "RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems", 2024, negative): For CUAD, only Claude 3 Haiku was used because the context length exceeds the GPT-3.5 16k token limit.
- `arxiv:2408.08632v2#c02` (arxiv:2408.08632v2 — "A Survey on Benchmarks of Multimodal Large Language Models", 2024, mixed): MileBench and MMNeedle explore long-context recall abilities of MLLMs using the needle-in-a-haystack method and image retrieval tasks.
- `arxiv:2308.12950#c01` (arxiv:2308.12950 — "Code Llama: Open Foundation Models for Code", 2023, mixed): On a synthetic key retrieval task with 16K-token context, models are compared against GPT-3.5-turbo using accuracy.

### h243 — The conflict is partly an evaluation-protocol mismatch: some papers measure context-window capacity or recall-at-length, while others judge end-task answer accuracy or practical usability under token-limit constraints.

**Mechanism.** A model can accept 128K-2M tokens and retrieve a planted fact, yet still fail to improve end-to-end QA accuracy because generation, reasoning, or truncation policies differ; conversely, papers emphasizing token-limit exclusions or answer accuracy can report negative results even when raw recall is strong.

**Predictions:**
- Capacity claims correlate weakly with answer accuracy
- After harmonizing metric and truncation rules, effect signs align more often

**Minimal test.** Re-score the same model outputs under a unified protocol: fixed effective context length, identical truncation policy, and both recall and answer-accuracy metrics reported side by side. Check whether the reported contradiction disappears once evaluation criteria are standardized.

**Scope.** method=long-context, task=long-context-QA

**Evidence gap.** The available claims conflate maximum supported tokens, recall, and task accuracy, with little evidence from shared evaluation scripts or common metrics.

**Graph bridge.** 128k input tokens → Accuracy

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.89 | 0.94 | 1.00 | 0.80 | 0.00 | 0.97 |

### Anomaly a160 — metric_mismatch

**Central question:** Do different metrics explain why agent appears inconsistent on agentic-reasoning?

**Shared entities:** method=agent, task=agentic-reasoning, metrics=accuracy, all metrics except answer correctness, average return, average success rate, averaged sr, ce, cgt, emr score; tf1 score, navigation steps, number of prompts the user needs to type, overall score, overall success rate, payoff, percent, precision-recall balance, precision; tf1, recall, schema compliance and valid tool naming, task success rate, tf1; emr, tool precision (tp), win rate
**Varying settings:** metric

**Evidence claims:**
- `arxiv:2603.24943v1#c01` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, negative): Tool Precision is lower on single-tool samples because models often over-predict and generate multiple tools when only one is needed.
- `arxiv:2603.24943v1#c02` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Model size does not consistently correlate with performance on FinMCP-Bench.
- `arxiv:2603.24943v1#c03` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Easy cases penalize over-calling with lower precision, while harder cases reward better recall and planning, leading to higher TF1 for models with balanced tool selection.
- `arxiv:2603.24943v1#c04` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): Top models lead by maintaining a better precision-recall balance across diverse scenarios rather than excelling in only a few.
- `arxiv:2603.24943v1#c06` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): On FinMCP-Bench overall, Qwen3-235B-A22B-Thinking attains the highest TF1 and EMR among the listed models.
- `arxiv:2502.04180v2#c01` (arxiv:2502.04180v2 — "Multi-agent Architecture Search via Agentic Supernet", 2025, positive): The proposed agentic supernet surpasses existing handcrafted or automated multi-agent systems.
- `arxiv:2505.12371#c04` (arxiv:2505.12371 — "MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks", 2025, mixed): In clinical workflow automation data tasks on MIMIC-IV, SmolAgents has a higher correct rate than a single LLM, while Owl is lower.
- `arxiv:2505.18079v4#c04` (arxiv:2505.18079v4 — "Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding", 2025, negative): Switching the reasoning model in agentic search to OpenAI o4-mini causes a 5.8% performance drop.
- `arxiv:2505.18079v4#c05` (arxiv:2505.18079v4 — "Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding", 2025, negative): Switching the reasoning model in agentic search to GPT-4o causes a 13.7% performance decline.
- `arxiv:2509.25140v2#c03` (arxiv:2509.25140v2 — "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory", 2025, positive): Compared with No memory, the method boosts overall performance by 20.5% while increasing total token consumption by around 4.3%.
- `arxiv:2401.10727v3#c01` (arxiv:2401.10727v3 — "MLLM-Tool: A Multimodal Large Language Model For Tool Agent Learning", 2024, negative): One-to-one scenarios perform significantly worse than one-to-many scenarios for the 7B and 13B models.
- `arxiv:2402.08189#c01` (arxiv:2402.08189 — "Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs", 2024, mixed): SingleLLM-4 was inconsistent across personality pairs, achieving human-like gameplay in all Fair-Fair simulations but only a small fraction of Greedy-Greedy conditions.
- `arxiv:2402.08189#c02` (arxiv:2402.08189 — "Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs", 2024, mixed): In SingleLLM-3.5, errors occurred more often in strategy creation than in gameplay.
- `arxiv:2402.08189#c03` (arxiv:2402.08189 — "Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs", 2024, negative): MultiAgent-3.5 performed slightly worse, producing complete and personality-consistent strategies for both players in 80% of simulations.
- `arxiv:2402.08189#c04` (arxiv:2402.08189 — "Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs", 2024, mixed): A fully greedy society may not survive, while a society with 10-25% greedy people may thrive because fair people uphold the system.
- `arxiv:2404.03648v2#c02` (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, negative): Hallucinations are the largest error category in web task automation at 44%.
- `arxiv:2407.01511v4#c01` (arxiv:2407.01511v4 — "CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents", 2024, positive): Single-agent GPT-4o achieves the best overall completion ratio of 38.01%.
- `arxiv:2407.01511v4#c02` (arxiv:2407.01511v4 — "CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents", 2024, mixed): GPT-4o single-agent and by-environment multi-agent structures can have the same success rate but completion ratios differing by up to 4.67%.
- `arxiv:2407.01511v4#c04` (arxiv:2407.01511v4 — "CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents", 2024, positive): On cross-platform tasks, GPT-4 Turbo (Single) attains a completion ratio of 52.61%.
- `arxiv:2410.24024v2#c01` (arxiv:2410.24024v2 — "AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents", 2024, positive): In XML mode, GPT-4-1106-Preview outperforms other models with the highest success rate and sub-goal success rate.
- `arxiv:2410.24024v2#c02` (arxiv:2410.24024v2 — "AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents", 2024, mixed): The two GPT-4 series models perform comparably on reasonable operation ratio, with around 86% of operations being reasonable.
- `arxiv:2410.24024v2#c03` (arxiv:2410.24024v2 — "AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents", 2024, positive): GPT-4o achieves the highest reversed redundancy ratio despite having a lower success rate, suggesting stronger reduction of unnecessary operations.
- `arxiv:2412.05467#c01` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, positive): GPT-4o performance improved over workarena-plus-plus measurements on WebArena and WorkArena L2.
- `arxiv:2412.05467#c02` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, positive): Claude improves substantially on WorkArena-L2 and achieves a 39.1% average task success rate.
- `arxiv:2412.05467#c04` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, positive): Claude-3.5-Sonnet achieved 39.1% success on WorkArena L2, compared with 8.5% for GPT-4o.
- `arxiv:2412.09082v3#c01` (arxiv:2412.09082v3 — "Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method", 2024, positive): GPT-4+NaviLLM shows improved performance over pre-trained and fine-tuned NaviLLM, with a 23% gain in ISR over fine-tuned NaviLLM.
- `arxiv:2308.08155#c05` (arxiv:2308.08155 — "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", 2023, negative): MiniWobChat achieves a 52.8% success rate on 49 available tasks, which is 3.6% lower than RCI on the MiniWob++ benchmark.
- `arxiv:2311.17227v2#c01` (arxiv:2311.17227v2 — "War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars", 2023, positive): In GPT-4 WWI simulations, alliances between Britain-France, German Empire-Austria-Hungary, and Serbia-Russia were consistently formed in all simulation results.
- `arxiv:2503.05659v2#c01` (arxiv:2503.05659v2 — "A Survey of Large Language Model Empowered Agents for Recommendation and Search: Towards Next-Generation Information Retrieval", 2025, mixed): Existing language model agents achieve high success on basic tasks but much lower success on composite tasks.
- `arxiv:2508.20453v1#c01` (arxiv:2508.20453v1 — "MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers", 2025, positive): Strong models maintain very high schema understanding, surpassing 98% in schema compliance and valid tool naming.
- `arxiv:2508.20453v1#c02` (arxiv:2508.20453v1 — "MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers", 2025, negative): llama-3-1-8b-instruct performs worse with multiple servers than in the single-server case.
- `arxiv:2508.20453v1#c03` (arxiv:2508.20453v1 — "MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers", 2025, negative): nova-micro-v1 performs worse with multiple servers than in the single-server case.
- `arxiv:2509.25140v2#c01` (arxiv:2509.25140v2 — "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory", 2025, positive): ReasoningBank improves overall success rate on WebArena compared to memory-free agents across three backbone LLMs.
- `arxiv:2509.25140v2#c02` (arxiv:2509.25140v2 — "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory", 2025, positive): On the WebArena Multi subset, ReasoningBank gains +4.6 averaged SR over the strongest baseline.
- `arxiv:2509.25140v2#c04` (arxiv:2509.25140v2 — "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory", 2025, positive): Using past reasoning hints reduces navigation steps from 29 to 10 compared to a baseline without memory.
- `arxiv:2401.07324v3#c01` (arxiv:2401.07324v3 — "Small LLMs Are Weak Tool Learners: A Multi-LLM Agent", 2024, positive): $ $-UMi (7B) surpasses ChatGPT and ToolLLaMA in both pass rate and win rate.
- `arxiv:2401.07324v3#c02` (arxiv:2401.07324v3 — "Small LLMs Are Weak Tool Learners: A Multi-LLM Agent", 2024, mixed): $ $-UMi underperforms GPT-4 in win rate but matches or exceeds GPT-4 in pass rate for some test groups such as I1-Inst.
- `arxiv:2401.07324v3#c04` (arxiv:2401.07324v3 — "Small LLMs Are Weak Tool Learners: A Multi-LLM Agent", 2024, mixed): $ $-UMi outperforms ChatGPT and ToolLLama on all metrics except answer correctness on ToolAlpaca.
- `arxiv:2401.13919#c01` (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, positive): WebVoyager achieves a 59.1% task success rate on the benchmark and outperforms GPT-4 (All Tools) and the text-only setting.
- `arxiv:2401.16158v2#c01` (arxiv:2401.16158v2 — "Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception", 2024, positive): Mobile-Agent achieved strong accuracy and completion rates in experiments.
- `arxiv:2305.18323#c01` (arxiv:2305.18323 — "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models", 2023, positive): ReWOO has higher accuracy than ReAct.
- `arxiv:2310.04406v3#c06` (arxiv:2310.04406v3 — "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models", 2023, positive): The abstract states that LATS reaches state-of-the-art 92.7% pass@1 on HumanEval with GPT-4 and 75.9 average score on WebShop with GPT-3.5, comparable to gradient-based fine-tuning.
- `arxiv:2311.17227v2#c02` (arxiv:2311.17227v2 — "War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars", 2023, positive): Under the default setting, GPT-4 achieves 77.78 alliance accuracy, 54.60 war declaration accuracy, and 92.09 mobilization accuracy on WWI simulation.
- `arxiv:2311.17227v2#c03` (arxiv:2311.17227v2 — "War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars", 2023, positive): GPT-4 outperforms Claude-2 and GPT-3.5 on WWI alliance and war declaration accuracy under the default setting.
- `arxiv:2402.19446#c01` (arxiv:2402.19446 — "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL", 2024, positive): ArCHer reaches an average return just above -17 with fewer than 1000 samples, whereas PPO needs more than 100k samples, indicating at least 100x higher sample efficiency.
- `arxiv:2407.01511v4#c03` (arxiv:2407.01511v4 — "CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents", 2024, positive): GPT-4o has a higher CE value than GPT-4 Turbo, indicating better cost-effectiveness.
- `arxiv:2412.09082v3#c02` (arxiv:2412.09082v3 — "Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method", 2024, negative): GPT-4+NaviLLM performs slightly worse than MGDM on complex tasks, especially in CGT.
- `arxiv:2412.09082v3#c03` (arxiv:2412.09082v3 — "Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method", 2024, negative): The CGT metric of GPT-4+NaviLLM is lower than that of fine-tuned NaviLLM.
- `arxiv:2308.05960v1#c01` (arxiv:2308.05960v1 — "BOLAA: Benchmarking and Orchestrating LLM-augmented Autonomous Agents", 2023, mixed): Llama-2-70b attains nearly 0.3344 recall on the ZS LAA setting, comparable to the best LAA.
- `arxiv:2308.08155#c03` (arxiv:2308.08155 — "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", 2023, positive): The system integrates multiple agents to reduce user interactions by 3–5 times on average.
- `arxiv:2312.11970v1#c03` (arxiv:2312.11970v1 — "Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives", 2023, negative): In the Battle of the Sexes, GPT-4 cannot coordinate well enough to obtain maximum payoff.

### h202 — Outcome metrics that require exact end-state correctness diverge from tool-call overlap metrics because they score different parts of the same agent trajectory.

**Mechanism.** EMR/success/pass-style metrics collapse the whole trajectory into a strict all-or-nothing final outcome, while TF1/precision/recall/tool-precision reward partial correctness in tool selection even when execution or final answer fails; over-calling can improve recall-based scores while hurting exact success.

**Predictions:**
- High-TF1 low-EMR cases will contain extra but plausible tool calls.
- Rankings by EMR will penalize over-calling more than rankings by TF1.

**Minimal test.** On a shared tool-use eval set such as FinMCP-Bench or MCP-Bench, collect raw per-example outputs from the same agent runs: full tool-call traces and final answers. Cross-score each identical output with both exact outcome metrics (EMR, answer correctness, task success) and tool overlap metrics (TP/TR/TF1). Then inspect examples where TF1 is high but EMR/success is zero, and compare model rankings under both scorings.

**Scope.** method=agent, task=agentic-reasoning

**Evidence gap.** We still need paired per-example traces with both tool annotations and final-task labels, not just aggregate tables.

**Graph bridge.** TF1; EMR → all metrics except answer correctness

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.94 | 1.00 | 0.82 | 0.00 | 0.97 |

### Anomaly a325 — community_disconnect

**Central question:** Could code-reasoning and pharmaceutical regulatory compliance be connected by a shared mechanism?

**Shared entities:** community_from=code-reasoning, community_to=pharmaceutical regulatory compliance, shared_concepts=evaluation protocol, text

**Evidence claims:**
- `arxiv:2503.19633v1#c06` (arxiv:2503.19633v1 — "1.4 Million Open-Source Distilled Reasoning Dataset to Empower Large Language Model Training", 2025, positive): Accuracy improved on LiveCodeBench from 57.5% to 59.7%.
- `arxiv:2410.11005#c03` (arxiv:2410.11005 — "Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks", 2024, negative): GPT-4o misinterprets an AAVE instruction as a general statement rather than a directive to name the function.
- `arxiv:2303.17760#c03` (arxiv:2303.17760 — "CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society", 2023, positive): CAMEL agents beat GPT-3.5-turbo on code evaluations judged by GPT-4.
- `arxiv:2309.07864#c04` (arxiv:2309.07864 — "The Rise and Potential of Large Language Model Based Agents: A Survey", 2023, negative): Too many agents in a step like coding can raise communication costs without substantial performance gains over a smaller agent count, so some agents may need to be removed dynamically.
- `arxiv:2402.01717#c01` (arxiv:2402.01717 — "From RAG to QA-RAG: Integrating Generative AI for Pharmaceutical Regulatory Compliance Process", 2024, positive): Using GPT-4 can achieve similarity up to 80%.
- `arxiv:2402.01717#c02` (arxiv:2402.01717 — "From RAG to QA-RAG: Integrating Generative AI for Pharmaceutical Regulatory Compliance Process", 2024, positive): ChatGPT-4 has higher recall than ChatGPT-3.5 Finetuned and Mistral 7B Finetuned on BertScore evaluation.
- `arxiv:2402.01717#c03` (arxiv:2402.01717 — "From RAG to QA-RAG: Integrating Generative AI for Pharmaceutical Regulatory Compliance Process", 2024, positive): ChatGPT-3.5 Finetuned has higher BertScore F1 than ChatGPT-4 and Mistral 7B Finetuned.
- `arxiv:2402.01717#c04` (arxiv:2402.01717 — "From RAG to QA-RAG: Integrating Generative AI for Pharmaceutical Regulatory Compliance Process", 2024, positive): The Only question approach achieves slightly higher context recall than the Only hypothetical answer method.
- `arxiv:2402.01717#c05` (arxiv:2402.01717 — "From RAG to QA-RAG: Integrating Generative AI for Pharmaceutical Regulatory Compliance Process", 2024, mixed): The final answer agent used in all cases was ChatGPT-3.5 Turbo with consistent answer-generation prompts.

### h053 — Venue separation and methodological priors sustain the disconnect: code-reasoning work favors agentic decomposition and distilled reasoning gains, whereas pharmaceutical regulatory compliance appears optimized around single-model answer generation with fixed prompts, so each side underestimates the other's tooling.

**Mechanism.** Code researchers publish around coding/reasoning benchmarks where multi-agent or distilled workflows are natural, while compliance studies use QA-style evaluation setups with a fixed final answer model, reducing incentives to test agentic coordination or reasoning distillation on regulatory text tasks.

**Predictions:**
- Agentic code methods improve compliance task completeness
- Fixed-prompt compliance setups underuse distilled reasoning gains

**Minimal test.** Import a small CAMEL-style or distilled-reasoning pipeline from code-reasoning into a pharmaceutical compliance benchmark currently using a fixed final answer agent, and compare completeness/consistency against the original single-model setup under equal token budgets.

**Scope.** method=agent or distillation versus prompting, task=regulatory answer generation or code task evaluation

**Evidence gap.** There is little direct evidence on whether agent communication overhead or distilled reasoning benefits transfer to long-form compliance text tasks under realistic cost constraints.

**Graph bridge.** code-reasoning → pharmaceutical regulatory compliance

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.92 | 0.90 | 1.00 | 0.88 | 0.00 | 0.97 |

### Anomaly a026 — impact_conflict

**Central question:** Why do high-impact papers disagree about agent on agentic-reasoning?

**Shared entities:** method=agent, task=agentic-reasoning

**Evidence claims:**
- `arxiv:2603.24943v1#c04` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): Top models lead by maintaining a better precision-recall balance across diverse scenarios rather than excelling in only a few.
- `arxiv:2603.24943v1#c05` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): Qwen3-30B-A3B-Thinking and Qwen3-235B-A22B-Thinking improve from Easy to Hard, suggesting they leverage richer constraints and multi-tool opportunities in harder queries.
- `arxiv:2603.24943v1#c06` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): On FinMCP-Bench overall, Qwen3-235B-A22B-Thinking attains the highest TF1 and EMR among the listed models.
- `arxiv:2502.04180v2#c01` (arxiv:2502.04180v2 — "Multi-agent Architecture Search via Agentic Supernet", 2025, positive): The proposed agentic supernet surpasses existing handcrafted or automated multi-agent systems.
- `arxiv:2502.19411v1#c06` (arxiv:2502.19411v1 — "Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs", 2025, positive): Agentless achieves the highest SWE-Bench (Lite) score among the listed agentic methods.
- `arxiv:2504.12477#c01` (arxiv:2504.12477 — "Towards Conversational AI for Human-Machine Collaborative MLOps", 2025, positive): The conversational MLOps assistant reduces complexity and lowers barriers to entry for users across diverse technical skill levels.
- `arxiv:2504.12477#c02` (arxiv:2504.12477 — "Towards Conversational AI for Human-Machine Collaborative MLOps", 2025, positive): The system uses a hierarchical modular architecture with specialized agents for pipeline orchestration, data management, and domain-specific knowledge integration.
- `arxiv:2504.12477#c03` (arxiv:2504.12477 — "Towards Conversational AI for Human-Machine Collaborative MLOps", 2025, positive): The LLM-based conversational agent system is designed to enhance human-machine collaboration in MLOps.
- `arxiv:2504.12477#c04` (arxiv:2504.12477 — "Towards Conversational AI for Human-Machine Collaborative MLOps", 2025, positive): Through iterative reasoning loops and context-aware processing, the system enables users with varying technical backgrounds to discover, execute, and monitor ML pipelines, manage datasets and artifacts, and access documentation via conversational interfaces.
- `arxiv:2508.20453v1#c01` (arxiv:2508.20453v1 — "MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers", 2025, positive): Strong models maintain very high schema understanding, surpassing 98% in schema compliance and valid tool naming.
- `arxiv:2509.25140v2#c01` (arxiv:2509.25140v2 — "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory", 2025, positive): ReasoningBank improves overall success rate on WebArena compared to memory-free agents across three backbone LLMs.
- `arxiv:2509.25140v2#c02` (arxiv:2509.25140v2 — "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory", 2025, positive): On the WebArena Multi subset, ReasoningBank gains +4.6 averaged SR over the strongest baseline.
- `arxiv:2509.25140v2#c03` (arxiv:2509.25140v2 — "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory", 2025, positive): Compared with No memory, the method boosts overall performance by 20.5% while increasing total token consumption by around 4.3%.
- `arxiv:2509.25140v2#c04` (arxiv:2509.25140v2 — "ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory", 2025, positive): Using past reasoning hints reduces navigation steps from 29 to 10 compared to a baseline without memory.
- `arxiv:2401.07324v3#c01` (arxiv:2401.07324v3 — "Small LLMs Are Weak Tool Learners: A Multi-LLM Agent", 2024, positive): $ $-UMi (7B) surpasses ChatGPT and ToolLLaMA in both pass rate and win rate.
- `arxiv:2401.07324v3#c03` (arxiv:2401.07324v3 — "Small LLMs Are Weak Tool Learners: A Multi-LLM Agent", 2024, positive): Agents with a 13B backbone perform better than their 7B counterparts.
- `arxiv:2401.07324v3#c05` (arxiv:2401.07324v3 — "Small LLMs Are Weak Tool Learners: A Multi-LLM Agent", 2024, positive): $ $-UMi with a 7B backbone can outperform a Single-LLM with a 13B backbone.
- `arxiv:2401.13919#c01` (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, positive): WebVoyager achieves a 59.1% task success rate on the benchmark and outperforms GPT-4 (All Tools) and the text-only setting.
- `arxiv:2401.13919#c02` (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, positive): WebVoyager outperforms text-only and GPT-4 (All Tools) baselines by large margins in most website tasks.
- `arxiv:2401.16158v2#c01` (arxiv:2401.16158v2 — "Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception", 2024, positive): Mobile-Agent achieved strong accuracy and completion rates in experiments.
- `arxiv:2401.16158v2#c02` (arxiv:2401.16158v2 — "Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception", 2024, positive): Mobile-Agent is more adaptable across diverse mobile operating environments and avoids system-specific customizations compared with XML- or metadata-based solutions.
- `arxiv:2401.16158v2#c04` (arxiv:2401.16158v2 — "Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception", 2024, positive): Mobile-Agent can complete challenging instructions such as multi-app operations.
- `arxiv:2402.19446#c01` (arxiv:2402.19446 — "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL", 2024, positive): ArCHer reaches an average return just above -17 with fewer than 1000 samples, whereas PPO needs more than 100k samples, indicating at least 100x higher sample efficiency.
- `arxiv:2402.19446#c02` (arxiv:2402.19446 — "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL", 2024, positive): On WebShop, online RL training of a GPT2 base model with ArCHer outperforms prompting strategies on GPT-3.5, including an expert-written prompt and ReAct.
- `arxiv:2402.19446#c03` (arxiv:2402.19446 — "ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL", 2024, positive): In ArCHer, only the high level interacts with the environment, while the low level is trained entirely in silico against the high-level value function.
- `arxiv:2403.15137#c02` (arxiv:2403.15137 — "CACA Agent: Capability Collaboration based AI Agent", 2024, positive): The framework enables rapid expansion of tool services and uses LLM inference for tool selection.
- `arxiv:2403.15137#c06` (arxiv:2403.15137 — "CACA Agent: Capability Collaboration based AI Agent", 2024, positive): Methodology capability stores processing knowledge for many application requests across scenarios.
- `arxiv:2404.03648v2#c05` (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, positive): AutoWebGLM is claimed in the abstract to outperform GPT-4 as an automated web navigation agent.
- `arxiv:2407.01511v4#c01` (arxiv:2407.01511v4 — "CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents", 2024, positive): Single-agent GPT-4o achieves the best overall completion ratio of 38.01%.
- `arxiv:2407.01511v4#c03` (arxiv:2407.01511v4 — "CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents", 2024, positive): GPT-4o has a higher CE value than GPT-4 Turbo, indicating better cost-effectiveness.
- `arxiv:2407.01511v4#c04` (arxiv:2407.01511v4 — "CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents", 2024, positive): On cross-platform tasks, GPT-4 Turbo (Single) attains a completion ratio of 52.61%.
- `arxiv:2410.24024v2#c01` (arxiv:2410.24024v2 — "AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents", 2024, positive): In XML mode, GPT-4-1106-Preview outperforms other models with the highest success rate and sub-goal success rate.
- `arxiv:2410.24024v2#c03` (arxiv:2410.24024v2 — "AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents", 2024, positive): GPT-4o achieves the highest reversed redundancy ratio despite having a lower success rate, suggesting stronger reduction of unnecessary operations.
- `arxiv:2412.05467#c01` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, positive): GPT-4o performance improved over workarena-plus-plus measurements on WebArena and WorkArena L2.
- `arxiv:2412.05467#c02` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, positive): Claude improves substantially on WorkArena-L2 and achieves a 39.1% average task success rate.
- `arxiv:2412.05467#c04` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, positive): Claude-3.5-Sonnet achieved 39.1% success on WorkArena L2, compared with 8.5% for GPT-4o.
- `arxiv:2412.09082v3#c01` (arxiv:2412.09082v3 — "Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method", 2024, positive): GPT-4+NaviLLM shows improved performance over pre-trained and fine-tuned NaviLLM, with a 23% gain in ISR over fine-tuned NaviLLM.
- `arxiv:2412.09082v3#c05` (arxiv:2412.09082v3 — "Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method", 2024, positive): Most models perform better on Spot robot tasks than on Stretch robot tasks, except InstructNav and GLM-4v prompt which are relatively even across both task types.
- `arxiv:2302.07842#c04` (arxiv:2302.07842 — "Augmented Language Models: a Survey", 2023, positive): RT-1 can follow over 700 natural language instructions and generalize to new tasks, environments, and objects.
- `arxiv:2305.18323#c01` (arxiv:2305.18323 — "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models", 2023, positive): ReWOO has higher accuracy than ReAct.
- `arxiv:2308.05960v1#c03` (arxiv:2308.05960v1 — "BOLAA: Benchmarking and Orchestrating LLM-augmented Autonomous Agents", 2023, positive): Llama-2-13b yields better performance than longchat-13b-16k despite having less context length.
- `arxiv:2308.05960v1#c04` (arxiv:2308.05960v1 — "BOLAA: Benchmarking and Orchestrating LLM-augmented Autonomous Agents", 2023, positive): fastchat-t5-3b performs above average on the ZS LAA architecture.
- `arxiv:2308.11432v7#c02` (arxiv:2308.11432v7 — "A Survey on Large Language Model based Autonomous Agents", 2023, positive): Integrating advanced LLMs like GPT-4 into a multi-agent system enables agents to adapt, perform complex tasks, and communicate effectively, leading to self-driven evolution in environment interactions.
- `arxiv:2308.10848v3#c03` (arxiv:2308.10848v3 — "AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors", 2023, positive): The framework's efficacy is illustrated across tasks requiring multi-agent decision-making, especially for GPT-4-based agents.
- `arxiv:2308.08155#c03` (arxiv:2308.08155 — "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", 2023, positive): The system integrates multiple agents to reduce user interactions by 3–5 times on average.
- `arxiv:2310.04406v3#c06` (arxiv:2310.04406v3 — "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models", 2023, positive): The abstract states that LATS reaches state-of-the-art 92.7% pass@1 on HumanEval with GPT-4 and 75.9 average score on WebShop with GPT-3.5, comparable to gradient-based fine-tuning.
- `arxiv:2311.17227v2#c01` (arxiv:2311.17227v2 — "War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars", 2023, positive): In GPT-4 WWI simulations, alliances between Britain-France, German Empire-Austria-Hungary, and Serbia-Russia were consistently formed in all simulation results.
- `arxiv:2311.17227v2#c02` (arxiv:2311.17227v2 — "War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars", 2023, positive): Under the default setting, GPT-4 achieves 77.78 alliance accuracy, 54.60 war declaration accuracy, and 92.09 mobilization accuracy on WWI simulation.
- `arxiv:2311.17227v2#c03` (arxiv:2311.17227v2 — "War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars", 2023, positive): GPT-4 outperforms Claude-2 and GPT-3.5 on WWI alliance and war declaration accuracy under the default setting.
- `arxiv:2312.11970v1#c01` (arxiv:2312.11970v1 — "Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives", 2023, positive): GPT-4 shows better strategic reasoning ability and converges to Nash equilibrium faster than GPT-3.5 and text-davinci.
- `arxiv:2603.24943v1#c01` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, negative): Tool Precision is lower on single-tool samples because models often over-predict and generate multiple tools when only one is needed.
- `arxiv:2603.24943v1#c02` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Model size does not consistently correlate with performance on FinMCP-Bench.
- `arxiv:2603.24943v1#c03` (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Easy cases penalize over-calling with lower precision, while harder cases reward better recall and planning, leading to higher TF1 for models with balanced tool selection.
- `arxiv:2503.05659v2#c01` (arxiv:2503.05659v2 — "A Survey of Large Language Model Empowered Agents for Recommendation and Search: Towards Next-Generation Information Retrieval", 2025, mixed): Existing language model agents achieve high success on basic tasks but much lower success on composite tasks.
- `arxiv:2505.12371#c04` (arxiv:2505.12371 — "MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks", 2025, mixed): In clinical workflow automation data tasks on MIMIC-IV, SmolAgents has a higher correct rate than a single LLM, while Owl is lower.
- `arxiv:2505.18079v4#c04` (arxiv:2505.18079v4 — "Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding", 2025, negative): Switching the reasoning model in agentic search to OpenAI o4-mini causes a 5.8% performance drop.
- `arxiv:2505.18079v4#c05` (arxiv:2505.18079v4 — "Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding", 2025, negative): Switching the reasoning model in agentic search to GPT-4o causes a 13.7% performance decline.
- `arxiv:2508.20453v1#c02` (arxiv:2508.20453v1 — "MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers", 2025, negative): llama-3-1-8b-instruct performs worse with multiple servers than in the single-server case.
- `arxiv:2508.20453v1#c03` (arxiv:2508.20453v1 — "MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers", 2025, negative): nova-micro-v1 performs worse with multiple servers than in the single-server case.
- `arxiv:2401.07324v3#c02` (arxiv:2401.07324v3 — "Small LLMs Are Weak Tool Learners: A Multi-LLM Agent", 2024, mixed): $ $-UMi underperforms GPT-4 in win rate but matches or exceeds GPT-4 in pass rate for some test groups such as I1-Inst.
- `arxiv:2401.07324v3#c04` (arxiv:2401.07324v3 — "Small LLMs Are Weak Tool Learners: A Multi-LLM Agent", 2024, mixed): $ $-UMi outperforms ChatGPT and ToolLLama on all metrics except answer correctness on ToolAlpaca.
- `arxiv:2401.10727v3#c01` (arxiv:2401.10727v3 — "MLLM-Tool: A Multimodal Large Language Model For Tool Agent Learning", 2024, negative): One-to-one scenarios perform significantly worse than one-to-many scenarios for the 7B and 13B models.
- `arxiv:2401.13919#c03` (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, mixed): WebVoyager is slightly worse than the text-only setting on Allrecipes and similar on Github, ESPN, Cambridge Dictionary, and Wolfram Alpha.
- `arxiv:2402.08189#c01` (arxiv:2402.08189 — "Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs", 2024, mixed): SingleLLM-4 was inconsistent across personality pairs, achieving human-like gameplay in all Fair-Fair simulations but only a small fraction of Greedy-Greedy conditions.
- `arxiv:2402.08189#c02` (arxiv:2402.08189 — "Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs", 2024, mixed): In SingleLLM-3.5, errors occurred more often in strategy creation than in gameplay.
- `arxiv:2402.08189#c03` (arxiv:2402.08189 — "Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs", 2024, negative): MultiAgent-3.5 performed slightly worse, producing complete and personality-consistent strategies for both players in 80% of simulations.
- `arxiv:2402.08189#c04` (arxiv:2402.08189 — "Simulating Human Strategic Behavior: Comparing Single and Multi-agent LLMs", 2024, mixed): A fully greedy society may not survive, while a society with 10-25% greedy people may thrive because fair people uphold the system.
- `arxiv:2403.15137#c04` (arxiv:2403.15137 — "CACA Agent: Capability Collaboration based AI Agent", 2024, negative): Existing methods are limited by finite tool definitions, ML-API-focused datasets, required data cleaning and annotation for tool expansion, and inability to meet complex diverse needs.
- `arxiv:2404.01230v1#c01` (arxiv:2404.01230v1 — "LLM as a Mastermind: A Survey of Strategic Reasoning with Large Language Models", 2024, mixed): CompeteAI introduces a GPT-4-simulated competitive environment centered on restaurant and customer agent interactions to illustrate business competition dynamics.
- `arxiv:2404.03648v2#c01` (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, negative): The system falters on advanced web applications such as maps, animations, and video browsing.
- `arxiv:2404.03648v2#c02` (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, negative): Hallucinations are the largest error category in web task automation at 44%.
- `arxiv:2406.05804v6#c02` (arxiv:2406.05804v6 — "A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning", 2024, mixed): Tool use, planning, and feedback learning are three prominent paradigms for developing LLM-based agents.
- `arxiv:2407.01511v4#c02` (arxiv:2407.01511v4 — "CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents", 2024, mixed): GPT-4o single-agent and by-environment multi-agent structures can have the same success rate but completion ratios differing by up to 4.67%.
- `arxiv:2410.24024v2#c02` (arxiv:2410.24024v2 — "AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents", 2024, mixed): The two GPT-4 series models perform comparably on reasonable operation ratio, with around 86% of operations being reasonable.
- `arxiv:2412.05467#c05` (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, mixed): Claude-3.5-Sonnet leads on almost all benchmarks, except vision-related tasks where GPT-4o is superior.
- `arxiv:2412.09082v3#c02` (arxiv:2412.09082v3 — "Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method", 2024, negative): GPT-4+NaviLLM performs slightly worse than MGDM on complex tasks, especially in CGT.
- `arxiv:2412.09082v3#c03` (arxiv:2412.09082v3 — "Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method", 2024, negative): The CGT metric of GPT-4+NaviLLM is lower than that of fine-tuned NaviLLM.
- `arxiv:2412.09082v3#c04` (arxiv:2412.09082v3 — "Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method", 2024, negative): In the Spot robot configuration, MGDM performs slightly worse than NaviLLM+GPT-4.
- `arxiv:2412.10380#c03` (arxiv:2412.10380 — "Challenges in Human-Agent Communication", 2024, negative): A lack of safeguards against system disruptions is a limitation even when the primary task is achieved.
- `arxiv:2412.10380#c04` (arxiv:2412.10380 — "Challenges in Human-Agent Communication", 2024, negative): Agents with greater capacity to act in the open world and complete goals bring a wider range of failure modes and associated costs.
- `arxiv:2305.18323#c02` (arxiv:2305.18323 — "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models", 2023, negative): GPT-3.5 can hit its 4096-token limit due to too many reasoning steps.
- `arxiv:2305.18323#c03` (arxiv:2305.18323 — "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models", 2023, negative): A failure mode is that the model can miss the answer even when reasoning and tool responses are correct.
- `arxiv:2308.05960v1#c01` (arxiv:2308.05960v1 — "BOLAA: Benchmarking and Orchestrating LLM-augmented Autonomous Agents", 2023, mixed): Llama-2-70b attains nearly 0.3344 recall on the ZS LAA setting, comparable to the best LAA.
- `arxiv:2308.05960v1#c02` (arxiv:2308.05960v1 — "BOLAA: Benchmarking and Orchestrating LLM-augmented Autonomous Agents", 2023, mixed): Llama-2-13b performs best under the PlanAct LAA architecture, while Llama-2-70b performs best under the BOLAA architecture.
- `arxiv:2308.08155#c05` (arxiv:2308.08155 — "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", 2023, negative): MiniWobChat achieves a 52.8% success rate on 49 available tasks, which is 3.6% lower than RCI on the MiniWob++ benchmark.
- `arxiv:2311.17227v2#c04` (arxiv:2311.17227v2 — "War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars", 2023, negative): In GPT-4 war-declaration simulations, mistakes mostly come from whether Britain and France choose to declare war against Austria-Hungary or the German Empire.
- `arxiv:2312.11970v1#c02` (arxiv:2312.11970v1 — "Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives", 2023, mixed): In the prisoner's dilemma, GPT-4 cooperates with cooperative opponents but defects after a single opponent defection.
- `arxiv:2312.11970v1#c03` (arxiv:2312.11970v1 — "Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives", 2023, negative): In the Battle of the Sexes, GPT-4 cannot coordinate well enough to obtain maximum payoff.

### h218 — The disagreement is moderated by prompt/interface format: agents look strong when action spaces are exposed in structured or multimodal formats, but weaker when the same tasks are presented through lossy text/HTML or differently serialized tool schemas.

**Mechanism.** Structured XML, schema-constrained tool calls, screenshots, or multimodal grounding reduce action ambiguity and hallucinated operations, whereas text-only or HTML-heavy interfaces force latent parsing and increase planning noise; this can reverse whether an agentic wrapper helps or hurts.

**Predictions:**
- Structured/XML/schema prompts improve agent success relative to text-only
- Format-matched evaluations reduce cross-paper sign disagreement

**Minimal test.** Take one shared family of tasks and run the same backbone agent under two strictly controlled interface conditions: structured/XML/schema-constrained or multimodal input versus text/HTML serialization, with identical decoding and budgets. Then compare agent-minus-baseline effects within each format. If the contradiction is format-driven, papers using the same interface should converge in sign.

**Scope.** method=agent, task=agentic-reasoning

**Evidence gap.** Most claims compare different interfaces and benchmarks simultaneously, so format is not isolated from task difficulty.

**Graph bridge.** agent → tool use

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.92 | 0.94 | 1.00 | 0.81 | 0.00 | 0.97 |

### Anomaly a146 — metric_mismatch

**Central question:** Do different metrics explain why evaluation-method appears inconsistent on domain-QA?

**Shared entities:** method=evaluation-method, task=domain-QA, metrics=accuracy, percent, win rate, win rate; macro-average
**Varying settings:** metric

**Evidence claims:**
- `arxiv:2404.10981#c05` (arxiv:2404.10981 — "A Survey on Retrieval-Augmented Text Generation for Large Language Models", 2026, mixed): MIRAGE evaluates RAG in medical QA with accuracy on MMLU-Med variants.
- `arxiv:2604.04325v1#c01` (arxiv:2604.04325v1 — "Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction", 2026, negative): Concatenating sharded evidence stays close to the full single-turn setting, suggesting little information loss from sharding.
- `arxiv:2604.04325v1#c05` (arxiv:2604.04325v1 — "Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction", 2026, positive): False-to-true revisions are more common than true-to-false revisions on average.
- `arxiv:2604.04325v1#c06` (arxiv:2604.04325v1 — "Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction", 2026, positive): Lab results strongly trigger earlier first answers for lab-dependent diseases compared with non-lab-dependent cases.
- `arxiv:2505.23802#c01` (arxiv:2505.23802 — "MedHELM: Holistic Evaluation of Large Language Models for Medical Tasks", 2025, positive): Claude models achieved 63-64% win rates with identical macro-averages of 0.73.
- `arxiv:2505.23802#c02` (arxiv:2505.23802 — "MedHELM: Holistic Evaluation of Large Language Models for Medical Tasks", 2025, positive): GPT-4o achieved a 57% win rate, higher than Gemini 2.0 Flash and GPT-4o mini.

### h185 — Win rate is measuring pairwise preference or relative dominance against comparators, whereas accuracy/macro-average measure absolute task performance, so a model can win many head-to-head judgments without having the best absolute correctness profile.

**Mechanism.** Pairwise win rate depends on the comparison pool, judge thresholds, and stylistic or partial-quality advantages; absolute metrics aggregate correctness per item or per subtask, which can flatten or reorder models even when one model wins more direct comparisons.

**Predictions:**
- Higher win-rate models need not have higher exact accuracy.
- Identical macro-averages can coexist with different pairwise win rates.

**Minimal test.** Take one shared domain-QA evaluation set and collect a single frozen set of model answers from the compared systems. Cross-score those same answers with both metrics: (1) exact accuracy or macro-average over task categories, and (2) pairwise win rate via blinded head-to-head judging on the same questions. Compare rank correlations and identify models whose pairwise win rate exceeds peers despite similar or lower absolute accuracy/macro-average.

**Scope.** method=evaluation-method, task=domain-QA

**Evidence gap.** The current claims do not provide a shared answer set with both pairwise judgments and absolute correctness labels for the same model outputs.

**Graph bridge.** win rate; macro-average → accuracy

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.88 | 0.90 | 1.00 | 0.82 | 0.00 | 0.96 |

### Anomaly a131 — setting_mismatch

**Central question:** Which settings flip the effect of long-context on long-context-QA?

**Shared entities:** method=long-context, task=long-context-QA
**Varying settings:** context_length

**Evidence claims:**
- `arxiv:2401.07883#c05` (arxiv:2401.07883 — "The Chronicles of RAG: The Retriever, the Chunk and the Generator", 2024, positive): GPT-4-1106-preview can process up to 128k input tokens, compared with GPT-1 and GPT-2 models handling up to 1024 tokens.
- `arxiv:2403.05530#c02` (arxiv:2403.05530 — "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context", 2024, positive): Gemini 1.5 shows better recall than GPT-4 Turbo up to 128K tokens.
- `arxiv:2501.00663#c02` (arxiv:2501.00663 — "Titans: Learning to Memorize at Test Time", 2024, positive): Titans can scale beyond a 2M context window and achieve better accuracy than baselines on needle-in-haystack tasks.
- `arxiv:2402.19473#c03` (arxiv:2402.19473 — "Retrieval-Augmented Generation for AI-Generated Content: A Survey", 2024, mixed): Prompt compression and long-context support partially mitigate lengthy-context challenges, with a slight trade-off in accuracy or costs.
- `arxiv:2407.11005#c03` (arxiv:2407.11005 — "RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems", 2024, negative): For CUAD, only Claude 3 Haiku was used because the context length exceeds the GPT-3.5 16k token limit.
- `arxiv:2408.08632v2#c02` (arxiv:2408.08632v2 — "A Survey on Benchmarks of Multimodal Large Language Models", 2024, mixed): MileBench and MMNeedle explore long-context recall abilities of MLLMs using the needle-in-a-haystack method and image retrieval tasks.
- `arxiv:2308.12950#c01` (arxiv:2308.12950 — "Code Llama: Open Foundation Models for Code", 2023, mixed): On a synthetic key retrieval task with 16K-token context, models are compared against GPT-3.5-turbo using accuracy.

### h244 — The conflicting long-context-QA results are primarily driven by the absolute context_length regime, with gains appearing at medium-long windows but degrading or saturating at very short or ultra-long windows.

**Mechanism.** As context_length increases, models first benefit from fitting all evidence, but beyond a model-specific range attention dilution, position bias, and optimization mismatch reduce effective retrieval and answer accuracy.

**Predictions:**
- Performance peaks at an intermediate context window on at least one benchmark.
- A 16K-cap model underperforms or fails once the same benchmark is expanded beyond its limit.

**Minimal test.** On one original benchmark such as the synthetic key retrieval task from arxiv:2308.12950#c01 or CUAD from arxiv:2407.11005#c03, hold model, prompt, metric, and data fixed while sweeping only context_length across 8K, 16K, 32K, 64K, 128K, and max-supported values; measure accuracy/recall and fit a non-monotonic curve.

**Scope.** method=long-context, task=long-context-QA

**Evidence gap.** The current claims mix capacity-limit statements with quality statements, so matched benchmark curves across multiple window sizes are missing.

**Graph bridge.** long-context → context_length

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.90 | 0.92 | 1.00 | 0.80 | 0.00 | 0.97 |

## Evidence claims

- **arxiv:2302.07842#c03** (arxiv:2302.07842 — "Augmented Language Models: a Survey", 2023, mixed): Chain-of-thought few-shot performance depends strongly on example format, choice, and order.
- **arxiv:2303.13988#c03** (arxiv:2303.13988 — "Machine Psychology", 2023, positive): Zero-shot chain-of-thought prompting improves reasoning performance.
- **arxiv:2303.17760#c03** (arxiv:2303.17760 — "CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society", 2023, positive): CAMEL agents beat GPT-3.5-turbo on code evaluations judged by GPT-4.
- **arxiv:2307.06435#c05** (arxiv:2307.06435 — "A Comprehensive Overview of Large Language Models", 2023, positive): Model safety improves with fine-tuning on safe demonstrations, and an additional RLHF step further improves safety and reduces jailbreak susceptibility.
- **arxiv:2308.12950#c01** (arxiv:2308.12950 — "Code Llama: Open Foundation Models for Code", 2023, mixed): On a synthetic key retrieval task with 16K-token context, models are compared against GPT-3.5-turbo using accuracy.
- **arxiv:2309.07864#c04** (arxiv:2309.07864 — "The Rise and Potential of Large Language Model Based Agents: A Survey", 2023, negative): Too many agents in a step like coding can raise communication costs without substantial performance gains over a smaller agent count, so some agents may need to be removed dynamically.
- **arxiv:2401.03991v1#c01** (arxiv:2401.03991v1 — "Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark", 2024, positive): On revised StepGame, GPT-4 with CoT outperforms GPT-4 base across most hop settings.
- **arxiv:2401.03991v1#c04** (arxiv:2401.03991v1 — "Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark", 2024, positive): The customized CoT method shows greater advantages in larger models such as GPT-4 and Davinci, maintaining accuracy as task complexity increases.
- **arxiv:2401.07324v3#c04** (arxiv:2401.07324v3 — "Small LLMs Are Weak Tool Learners: A Multi-LLM Agent", 2024, mixed): $ $-UMi outperforms ChatGPT and ToolLLama on all metrics except answer correctness on ToolAlpaca.
- **arxiv:2401.07883#c05** (arxiv:2401.07883 — "The Chronicles of RAG: The Retriever, the Chunk and the Generator", 2024, positive): GPT-4-1106-preview can process up to 128k input tokens, compared with GPT-1 and GPT-2 models handling up to 1024 tokens.
- **arxiv:2401.13919#c01** (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, positive): WebVoyager achieves a 59.1% task success rate on the benchmark and outperforms GPT-4 (All Tools) and the text-only setting.
- **arxiv:2401.13919#c05** (arxiv:2401.13919 — "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models", 2024, positive): The evaluation protocol achieves 85.3% agreement with human judges, suggesting GPT-4V is a reliable evaluator for online agents.
- **arxiv:2402.01717#c05** (arxiv:2402.01717 — "From RAG to QA-RAG: Integrating Generative AI for Pharmaceutical Regulatory Compliance Process", 2024, mixed): The final answer agent used in all cases was ChatGPT-3.5 Turbo with consistent answer-generation prompts.
- **arxiv:2402.19473#c03** (arxiv:2402.19473 — "Retrieval-Augmented Generation for AI-Generated Content: A Survey", 2024, mixed): Prompt compression and long-context support partially mitigate lengthy-context challenges, with a slight trade-off in accuracy or costs.
- **arxiv:2403.05530#c02** (arxiv:2403.05530 — "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context", 2024, positive): Gemini 1.5 shows better recall than GPT-4 Turbo up to 128K tokens.
- **arxiv:2403.05530#c03** (arxiv:2403.05530 — "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context", 2024, positive): Full-context provides better answers than retrieval-augmented generation with 4k tokens in 78% of cases.
- **arxiv:2404.03648v2#c01** (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, negative): The system falters on advanced web applications such as maps, animations, and video browsing.
- **arxiv:2404.03648v2#c02** (arxiv:2404.03648v2 — "AutoWebGLM: A Large Language Model-based Web Navigating Agent", 2024, negative): Hallucinations are the largest error category in web task automation at 44%.
- **arxiv:2404.10981#c05** (arxiv:2404.10981 — "A Survey on Retrieval-Augmented Text Generation for Large Language Models", 2026, mixed): MIRAGE evaluates RAG in medical QA with accuracy on MMLU-Med variants.
- **arxiv:2407.02485v1#c03** (arxiv:2407.02485v1 — "RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs", 2024, negative): A shorter top-k context list usually yields higher generation accuracy than reading too many chunked contexts such as top-100.
- **arxiv:2407.11005#c03** (arxiv:2407.11005 — "RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems", 2024, negative): For CUAD, only Claude 3 Haiku was used because the context length exceeds the GPT-3.5 16k token limit.
- **arxiv:2407.11373#c02** (arxiv:2407.11373 — "Reliable Reasoning Beyond Natural Language", 2024, negative): GPT-4 text-only chain-of-thought performance drops as variable interdependence increases, reaching 0% on problems with four interdependent variables.
- **arxiv:2410.24024v2#c01** (arxiv:2410.24024v2 — "AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents", 2024, positive): In XML mode, GPT-4-1106-Preview outperforms other models with the highest success rate and sub-goal success rate.
- **arxiv:2411.00836v2#c05** (arxiv:2411.00836v2 — "DynaMath: A Dynamic Visual Benchmark for Evaluating Mathematical Reasoning Robustness of Vision Language Models", 2024, mixed): 3-shot chain-of-thought slightly improves GPT-4o but reduces performance for Claude-3.5 and Gemini Pro 1.5.
- **arxiv:2412.05467#c03** (arxiv:2412.05467 — "The BrowserGym Ecosystem for Web Agent Research", 2024, mixed): Task validation uses logic rules, exact matching, or semantic matching with GPT-3.5 as a judge, which implies some costs.
- **arxiv:2501.00663#c02** (arxiv:2501.00663 — "Titans: Learning to Memorize at Test Time", 2024, positive): Titans can scale beyond a 2M context window and achieve better accuracy than baselines on needle-in-haystack tasks.
- **arxiv:2502.01549v1#c02** (arxiv:2502.01549v1 — "VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos", 2025, positive): VideoRAG has a higher overall win rate than GraphRAG-l across all videos.
- **arxiv:2502.01549v1#c03** (arxiv:2502.01549v1 — "VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos", 2025, positive): VideoRAG has a higher overall win rate than GraphRAG-g across all videos.
- **arxiv:2502.20808v6#c01** (arxiv:2502.20808v6 — "MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts", 2025, positive): Adding these strategies improved GPT-4o performance on multi-step free-form questions from 25.4% to 32.6%.
- **arxiv:2503.19633v1#c06** (arxiv:2503.19633v1 — "1.4 Million Open-Source Distilled Reasoning Dataset to Empower Large Language Model Training", 2025, positive): Accuracy improved on LiveCodeBench from 57.5% to 59.7%.
- **arxiv:2505.23802#c01** (arxiv:2505.23802 — "MedHELM: Holistic Evaluation of Large Language Models for Medical Tasks", 2025, positive): Claude models achieved 63-64% win rates with identical macro-averages of 0.73.
- **arxiv:2505.23802#c02** (arxiv:2505.23802 — "MedHELM: Holistic Evaluation of Large Language Models for Medical Tasks", 2025, positive): GPT-4o achieved a 57% win rate, higher than Gemini 2.0 Flash and GPT-4o mini.
- **arxiv:2508.20453v1#c01** (arxiv:2508.20453v1 — "MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers", 2025, positive): Strong models maintain very high schema understanding, surpassing 98% in schema compliance and valid tool naming.
- **arxiv:2601.07780v1#c05** (arxiv:2601.07780v1 — "Enhancing Self-Correction in Large Language Models through Multi-Perspective Reflection", 2026, negative): Chain-of-Thought prompting still faces consistency, accuracy, and self-correction challenges on complex or ethically sensitive tasks.
- **arxiv:2603.22862v2#c01** (arxiv:2603.22862v2 — "The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration", 2026, positive): Adversarial tool chains used as negative samples in alignment fine-tuning enhance LLM resilience against interference and improve robustness of parameter extraction.
- **arxiv:2603.24943v1#c01** (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, negative): Tool Precision is lower on single-tool samples because models often over-predict and generate multiple tools when only one is needed.
- **arxiv:2603.24943v1#c03** (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, mixed): Easy cases penalize over-calling with lower precision, while harder cases reward better recall and planning, leading to higher TF1 for models with balanced tool selection.
- **arxiv:2603.24943v1#c06** (arxiv:2603.24943v1 — "FinMCP-Bench: Benchmarking LLM Agents for Real-World Financial Tool Use under the Model Context Protocol", 2026, positive): On FinMCP-Bench overall, Qwen3-235B-A22B-Thinking attains the highest TF1 and EMR among the listed models.
- **arxiv:2604.04325v1#c01** (arxiv:2604.04325v1 — "Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction", 2026, negative): Concatenating sharded evidence stays close to the full single-turn setting, suggesting little information loss from sharding.
