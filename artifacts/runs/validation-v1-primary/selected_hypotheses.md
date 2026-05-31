# Selected Hypotheses

Selected **10** hypotheses across **8** anomalies.

Exploratory report: no synthesized insights were supported yet, so the sections below show claim-level evidence and candidate explanations rather than a settled takeaway.

## Conflict Hypotheses

### Anomaly a515 — community_disconnect

**Central question:** Could natural language understanding and resource allocation with censored semi-bandit feedback be connected by a shared mechanism?

**Shared entities:** community_from=natural language understanding, community_to=resource allocation with censored semi-bandit feedback, shared_concepts=evaluation protocol, text

**Evidence claims:**
- `arxiv:1804.07461v1#c01` (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, positive): Using attention improves multi-task BiLSTM performance over a vanilla BiLSTM on sentence-pair tasks.
- `arxiv:1804.07461v1#c02` (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, negative): Existing models do not substantially outperform training a separate model for each task on the main GLUE benchmark.
- `arxiv:1804.07461v1#c03` (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, mixed): Naive multi-task learning with standard models performs no better overall than training separate models per task.
- `arxiv:1804.07461v1#c04` (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, positive): Multi-task learning improves over a single-task model for tasks with less training data.
- `arxiv:1909.01504v1#c01` (arxiv:1909.01504v1 — "Censored Semi-Bandits: A Framework for Resource Allocation with Censored Feedback", 2019, mixed): CSB-DT is empirically compared against CSB-DT-UCB in experiments.
- `arxiv:1909.01504v1#c03` (arxiv:1909.01504v1 — "Censored Semi-Bandits: A Framework for Resource Allocation with Censored Feedback", 2019, positive): Once the threshold is estimated accurately enough to be allocation equivalent, the problem reduces to solving a knapsack problem.
- `arxiv:1909.01504v1#c04` (arxiv:1909.01504v1 — "Censored Semi-Bandits: A Framework for Resource Allocation with Censored Feedback", 2019, negative): Problem instances with δ = 0 are hopeless because optimal allocation requires full-precision threshold estimates.

### h068 — Venue separation persists because natural language understanding emphasizes benchmark-level aggregate scores, whereas censored semi-bandit work emphasizes algorithmic instance conditions; cross-pollination would be most visible if NLU adopted instance-level threshold diagnostics and semi-bandits adopted benchmark-style ablations on representation sharing.

**Mechanism.** Different evaluation priors obscure analogous failure modes: aggregate benchmark reporting can hide that multi-task learning only helps some tasks, while regret/instance analyses can hide whether gains depend on shared features analogous to text encoders or attention modules.

**Predictions:**
- Task-level breakdowns in NLU reveal threshold-like regimes of transfer benefit.
- Ablating shared context features in semi-bandits changes performance more than optimizer choice.

**Minimal test.** Re-evaluate an NLU multi-task benchmark with per-task data-regime and margin diagnostics inspired by allocation-equivalence thresholds, and re-run a small censored semi-bandit benchmark with benchmark-style ablations comparing separate models, shared models, and attention-like context weighting.

**Scope.** method=diagnostic ablations and regime-stratified evaluation, task=multi-task sentence-pair benchmarks and contextualized resource allocation simulations

**Evidence gap.** Few papers directly align reporting granularity, making it unclear whether the apparent disagreement is just an artifact of evaluation style.

**Graph bridge.** natural language understanding → resource allocation with censored semi-bandit feedback

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.93 | 0.87 | 1.00 | 0.70 | 0.00 | 0.96 |

### Anomaly a007 — impact_conflict

**Central question:** Why do high-impact papers disagree about evaluation-method on image classification?

**Shared entities:** method=evaluation-method, task=image classification

**Evidence claims:**
- `arxiv:2002.10365v1#c01` (arxiv:2002.10365v1 — "The Early Phase of Neural Network Training", 2020, positive): Evaluation accuracy rises rapidly early in training, reaching 55% after the first epoch and exceeding half of the final 91.5% accuracy.
- `arxiv:2002.10648v1#c01` (arxiv:2002.10648v1 — "I Am Going MAD: Maximum Discrepancy Competition for Comparing Classifiers Adaptively", 2020, positive): VGG16BN outperforms ResNet34 and ResNet101 under similar computation budgets.
- `arxiv:1804.08838v1#c06` (arxiv:1804.08838v1 — "Measuring the Intrinsic Dimension of Objective Landscapes", 2018, mixed): The threshold used differs by dataset: 90% for MNIST and 45% for CIFAR-10.
- `arxiv:1802.03796v1#c04` (arxiv:1802.03796v1 — "Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks", 2018, mixed): Evaluation used the CIFAR-100 and STL-10 datasets.
- `arxiv:2002.10648v1#c03` (arxiv:2002.10648v1 — "I Am Going MAD: Maximum Discrepancy Competition for Comparing Classifiers Adaptively", 2020, negative): PNASNet-5-Large's rank drops substantially under MAD.

### h132 — The contradiction reflects an evaluation protocol moderator: thresholding and scoring rules differ across papers, and these protocol choices can flip whether the evaluation-method appears favorable.

**Mechanism.** If one study uses dataset-specific thresholds or relative-rank criteria while another uses raw accuracy or different decision cutoffs, small score differences get transformed nonlinearly into opposite conclusions about superiority.

**Predictions:**
- Using one common threshold removes some sign flips
- Raw-score comparisons disagree less than thresholded ranks

**Minimal test.** Standardize the evaluation protocol by applying one shared scoring rule to all reported model outputs: same threshold definition, same ranking aggregation, and same metric transformation; then test whether the conflicting claims converge under the unified protocol.

**Scope.** method=evaluation-method, task=image classification

**Evidence gap.** The claims do not provide enough side-by-side raw outputs under multiple scoring rules to separate protocol effects from true model differences.

**Graph bridge.** intrinsic dimension thresholding → Accuracy

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.90 | 0.92 | 1.00 | 0.56 | 0.00 | 0.95 |

### h130 — The contradiction arises because the evaluation-method changes sign across dataset regimes, favoring some models or conclusions on simple/low-diversity datasets but reversing on harder or more diverse image classification datasets.

**Mechanism.** Dataset composition moderates the metric: class count, image complexity, and domain diversity alter which inductive biases are rewarded, so a method that looks favorable on MNIST/CIFAR-10 can rank models differently on CIFAR-100/STL-10 or in early-training analyses.

**Predictions:**
- Rankings align within a single fixed dataset
- Sign reversals concentrate between CIFAR-10 and CIFAR-100/STL-10

**Minimal test.** Re-run the same evaluation-method on the same set of architectures using a harmonized protocol across MNIST, CIFAR-10, CIFAR-100, and STL-10; then compare whether the positive and negative claims disappear when analysis is restricted to one dataset at a time.

**Scope.** method=evaluation-method, task=image classification

**Evidence gap.** The claims do not report matched cross-dataset reruns with identical models and hyperparameters, so direct evidence for dataset-driven sign flips is missing.

**Graph bridge.** MNIST; CIFAR-10 → CIFAR-100; STL-10

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.86 | 0.90 | 1.00 | 0.56 | 0.00 | 0.94 |

### Anomaly a020 — impact_conflict

**Central question:** Why do high-impact papers disagree about search-based on planning?

**Shared entities:** method=search-based, task=planning

**Evidence claims:**
- `arxiv:1906.05253v1#c01` (arxiv:1906.05253v1 — "Search on the Replay Buffer: Bridging Planning and Reinforcement Learning", 2019, positive): SoRB generalizes to new houses and reaches almost 80% of goals 10 steps away, outperforming goal-conditioned RL.
- `arxiv:1906.05253v1#c02` (arxiv:1906.05253v1 — "Search on the Replay Buffer: Bridging Planning and Reinforcement Learning", 2019, positive): SoRB reaches almost 80% of goals 10 steps away, about four times more than the goal-conditioned RL agent.
- `arxiv:1906.05253v1#c03` (arxiv:1906.05253v1 — "Search on the Replay Buffer: Bridging Planning and Reinforcement Learning", 2019, positive): SoRB achieves 22% higher average AUC than SPTM across five random seeds.
- `arxiv:2006.15820v1#c01` (arxiv:2006.15820v1 — "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search", 2020, positive): Retro* solves 31% more test molecules than DFPN-E under a budget of 500 one-step calls.
- `arxiv:2006.15820v1#c03` (arxiv:2006.15820v1 — "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search", 2020, positive): Retro* outperforms baseline algorithms early and improves faster as the time limit increases.
- `arxiv:2006.15820v1#c04` (arxiv:2006.15820v1 — "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search", 2020, positive): Using a value function helps MCTS because it provides a lower-variance value estimate than rollout.
- `arxiv:2006.15820v1#c02` (arxiv:2006.15820v1 — "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search", 2020, mixed): Retro* achieves higher success rate and better route quality than several baselines in the performance summary.

### h138 — The disagreement is moderated by value-function quality and training-data coverage: search-based planning helps when the learned heuristic/value is accurate on the test distribution, but becomes only mixed when heuristic estimates are weaker or distribution shift lowers guidance quality.

**Mechanism.** Methods like SoRB and Retro* rely on learned distance or value estimates to prioritize expansions; if those estimates generalize well, search amplifies them into large gains, whereas poor calibration or coverage gaps make search explore misleading branches, reducing advantages relative to simpler baselines.

**Predictions:**
- With in-distribution test targets, heuristic-guided search improves strongly.
- After degrading heuristic accuracy, the advantage over baselines shrinks.

**Minimal test.** Hold the planner and evaluation metric fixed while varying only heuristic quality: compare search-based performance using the original learned value function, a retrained value function on reduced or shifted data, and an oracle/ablated heuristic, then test whether the positive versus mixed discrepancy tracks heuristic accuracy.

**Scope.** method=search-based with learned value or distance estimates, task=planning

**Evidence gap.** The claims suggest a value-function mechanism but do not quantify cross-setting heuristic calibration or distribution-shift sensitivity.

**Graph bridge.** arxiv:2006.15820v1#c04 → arxiv:1906.05253v1#c01

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.86 | 0.90 | 1.00 | 0.54 | 0.00 | 0.94 |

### Anomaly a215 — metric_mismatch

**Central question:** Do different metrics explain why planning appears inconsistent on planning?

**Shared entities:** method=planning, task=planning, metrics=accuracy, all metrics, iterations, ne, sr, spl, osr, planning loss, planning time, sr
**Varying settings:** metric

**Evidence claims:**
- `arxiv:2006.15085v1#c01` (arxiv:2006.15085v1 — "What can I do here? A Theory of Affordances in Reinforcement Learning", 2020, positive): Using affordances significantly reduces planning time compared to using the full model, especially in larger environments.
- `arxiv:2006.15085v1#c02` (arxiv:2006.15085v1 — "What can I do here? A Theory of Affordances in Reinforcement Learning", 2020, mixed): In the small data regime, minimum planning loss is achieved at intermediate values, corresponding to an intermediate affordance set size.
- `arxiv:2007.05655v1#c01` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): EGP improves navigation metrics over the best prior models on the test set.
- `arxiv:2007.05655v1#c02` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): Using the EGP module without synthetic data augmentation improves performance over the baseline agent on Val Unseen.
- `arxiv:2007.05655v1#c03` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): The EGP agent with synthetic data augmentation outperforms prior art on all metrics on validation-unseen and test sets.
- `arxiv:2007.05655v1#c04` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, negative): Reducing top-K makes paths shorter but consistently lowers model accuracy.
- `arxiv:2007.05655v1#c05` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): Information propagation and more planning channels increase success rate in ablations.
- `arxiv:1802.03654v1#c02` (arxiv:1802.03654v1 — "Beyond the One-Step Greedy Approach in Reinforcement Learning", 2018, positive): The h-PI algorithm converges within a stated finite iteration bound.
- `arxiv:1802.03654v1#c03` (arxiv:1802.03654v1 — "Beyond the One-Step Greedy Approach in Reinforcement Learning", 2018, positive): The λ-PI algorithm converges within a stated finite iteration bound.

### h147 — Optimization-surrogate metrics are measuring model fit to planning signals, whereas evaluation metrics are measuring realized decision performance, so better planning loss need not imply better external planning behavior and vice versa.

**Mechanism.** Planning loss is a training or proxy objective over predicted transitions, affordances, or action values, while accuracy, NE, SR, and planning time evaluate downstream behavior after decoding/search; intermediate-capacity models can minimize surrogate loss yet fail to produce the best executed plans, and conversely a biased but fast planner can act well despite worse loss.

**Predictions:**
- Outputs from the lowest-loss checkpoint will not always have the best SR/NE on the same episodes.
- Intermediate affordance sizes can minimize planning loss while a different size yields better executed accuracy or planning time.

**Minimal test.** Train or checkpoint the same planning model across epochs or affordance-size settings, then for one shared eval set generate and freeze the actual planned outputs from each checkpoint/setting. Cross-score those same outputs with planning loss under the model, plus downstream metrics such as accuracy, NE, SR, SPL, and planning time. Test whether rankings by loss diverge from rankings by executed performance.

**Scope.** method=planning, task=planning

**Evidence gap.** The claims do not provide checkpoint-level paired evaluations linking surrogate-loss minima to downstream trajectory outcomes on the same examples.

**Graph bridge.** planning loss → NE, SR, SPL, OSR

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.87 | 0.88 | 1.00 | 0.60 | 0.00 | 0.95 |

### Anomaly a063 — impact_conflict

**Central question:** Why do high-impact papers disagree about search-based on code-generation?

**Shared entities:** method=search-based, task=code-generation

**Evidence claims:**
- `arxiv:1802.04335v1#c02` (arxiv:1802.04335v1 — "Neural Program Search: Solving Programming Tasks from Description and Examples", 2018, positive): The final Seq2Tree + Search model achieves the best result of 85.8%.
- `arxiv:1802.04335v1#c03` (arxiv:1802.04335v1 — "Neural Program Search: Solving Programming Tasks from Description and Examples", 2018, positive): Seq2Tree + Search attains 85.8% test accuracy, higher than Attentional Seq2Seq at 54.1% and Attentional Seq2Seq + Search at 72.3%.
- `arxiv:1802.04335v1#c04` (arxiv:1802.04335v1 — "Neural Program Search: Solving Programming Tasks from Description and Examples", 2018, positive): Model accuracy increases when search explores more trees.
- `arxiv:1809.02840v1#c05` (arxiv:1809.02840v1 — "Neural Guided Constraint Logic Programming for Program Synthesis", 2018, positive): RobustFill Attention-C with a large beam size solved one more problem than RNN+Constraints on flattened tree manipulation problems.
- `arxiv:1902.06349v1#c05` (arxiv:1902.06349v1 — "Learning to Infer Program Sketches", 2019, mixed): With beam size 10 on the full dataset, the Generator only RNN baseline far exceeds previously reported state of the art and achieves near-perfect accuracy, while the synthesis-only model fails to achieve high performance.

### h121 — The sign of search-based gains flips with dataset composition: constrained synthetic program-synthesis benchmarks reward search, while fuller or easier datasets let strong generators solve most cases without search.

**Mechanism.** Search helps most when the target space is combinatorial and outputs must satisfy strict execution constraints; if the dataset contains many shorter, more stereotyped, or generator-friendly programs, beam-decoded generation alone can already reach near-perfect accuracy, making added search look unnecessary or harmful.

**Predictions:**
- Search gains grow on harder compositional subsets
- Contradiction shrinks after matching task difficulty

**Minimal test.** Re-evaluate a common search-based model and a generator-only beam baseline on matched subsets from the compared benchmarks, stratified by program length, tree depth, and constraint complexity; if both papers' results align within each stratum, dataset composition is the moderator.

**Scope.** method=search-based, task=code-generation

**Evidence gap.** The claims do not provide harmonized difficulty statistics or overlapping subset evaluations across datasets.

**Graph bridge.** search-based → code-generation

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.91 | 0.94 | 1.00 | 0.58 | 0.00 | 0.96 |

### Anomaly a040 — impact_conflict

**Central question:** Why do high-impact papers disagree about planning on planning?

**Shared entities:** method=planning, task=planning

**Evidence claims:**
- `arxiv:2006.15085v1#c01` (arxiv:2006.15085v1 — "What can I do here? A Theory of Affordances in Reinforcement Learning", 2020, positive): Using affordances significantly reduces planning time compared to using the full model, especially in larger environments.
- `arxiv:2007.05655v1#c01` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): EGP improves navigation metrics over the best prior models on the test set.
- `arxiv:2007.05655v1#c02` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): Using the EGP module without synthetic data augmentation improves performance over the baseline agent on Val Unseen.
- `arxiv:2007.05655v1#c03` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): The EGP agent with synthetic data augmentation outperforms prior art on all metrics on validation-unseen and test sets.
- `arxiv:2007.05655v1#c05` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): Information propagation and more planning channels increase success rate in ablations.
- `arxiv:1802.03654v1#c01` (arxiv:1802.03654v1 — "Beyond the One-Step Greedy Approach in Reinforcement Learning", 2018, positive): The h-greedy policy yields component-wise value improvement, with equality only for an optimal policy.
- `arxiv:1802.03654v1#c02` (arxiv:1802.03654v1 — "Beyond the One-Step Greedy Approach in Reinforcement Learning", 2018, positive): The h-PI algorithm converges within a stated finite iteration bound.
- `arxiv:1802.03654v1#c03` (arxiv:1802.03654v1 — "Beyond the One-Step Greedy Approach in Reinforcement Learning", 2018, positive): The λ-PI algorithm converges within a stated finite iteration bound.
- `arxiv:2006.15085v1#c02` (arxiv:2006.15085v1 — "What can I do here? A Theory of Affordances in Reinforcement Learning", 2020, mixed): In the small data regime, minimum planning loss is achieved at intermediate values, corresponding to an intermediate affordance set size.
- `arxiv:2007.05655v1#c04` (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, negative): Reducing top-K makes paths shorter but consistently lowers model accuracy.

### h112 — The contradiction arises because planning helps when the evaluation distribution contains large or unseen environments, but can hurt or become non-monotonic in small-data or simpler regimes where planning abstractions discard useful detail.

**Mechanism.** Dataset composition moderates the sign: larger/unseen environments increase the value of pruning and lookahead, while small-data regimes or simpler state spaces make coarse affordance sets or restricted planning channels underfit, producing mixed or negative effects.

**Predictions:**
- Planning gains grow with environment size or unseen splits
- Intermediate planning capacity is best in low-data settings

**Minimal test.** Re-run the same planning method and baseline across matched bins of environment size and training-set size, holding architecture and optimizer fixed; if both papers are evaluated on the same large-vs-small and high-data-vs-low-data slices, the sign disagreement should shrink or become a consistent interaction rather than a contradiction.

**Scope.** method=planning, task=planning

**Evidence gap.** A cross-paper stratified comparison using identical size and data-regime bins is missing.

**Graph bridge.** planning → planning

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.86 | 0.94 | 1.00 | 0.60 | 0.00 | 0.95 |

### h114 — The papers appear to disagree because they evaluate different outcome variables and protocols, where planning can improve computation or convergence guarantees while worsening task accuracy under another protocol.

**Mechanism.** Evaluation protocol differences moderate the apparent sign. One line measures planning time, iteration bounds, or theoretical value improvement, while another measures accuracy-oriented navigation metrics; planning can reduce computation or guarantee improvement yet still lower accuracy when search truncation or split-specific evaluation changes the operational objective.

**Predictions:**
- Matched metrics reduce apparent sign disagreement
- Accuracy-cost frontiers reveal planning helps one metric but hurts another

**Minimal test.** Evaluate the same planner variants under a unified protocol that reports both accuracy metrics and computational metrics on the same tasks and splits; if the contradiction is protocol-driven, the papers should agree once compared on the same metric family and stopping criteria.

**Scope.** method=planning, task=planning

**Evidence gap.** No joint benchmark reports planning time, convergence-style metrics, and task accuracy for the same methods under the same stopping rules.

**Graph bridge.** planning time → accuracy

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.85 | 0.92 | 1.00 | 0.60 | 0.00 | 0.95 |

### Anomaly a194 — metric_mismatch

**Central question:** Do different metrics explain why evaluation-method appears inconsistent on evaluation?

**Shared entities:** method=evaluation-method, task=evaluation, metrics=# train, # test, # vocab, # tokens, $10\%$, 0-1 error, 0-1 loss, 15\% of trials, 2\%, 95% confidence intervals, absolute error, accuracy, accuracy; unfairness, adversarial risk, all metrics, amortization gap, approximation gap, auc, average $ _2$-distance error, average cumulative error, average document-level topic entropy, average oracle bleu, average regret, average return, average reward; average success rate, average value of the learned policy on the test set, bleu and nsbleu, bleu and self-bleu, bleu-nsbleu, bound, bounds, capacity and geometry, classification error, correlation, correlations, covering number, cr-nrr, d_w, differential privacy, eigenvalue, eigenvalues; rank, empirical risk; empirical advrisk, estimate difference, execution time, expected calibration error, fairnessloss, fcr, fid, fields distribution peak, generalization bound, generalization loss, geodesic distance, h( _ ) and i( _ ; _ -1 ), hyperparameter search results, intersection over union (iou), intersection over union (iou); mean class-wise accuracy, krackhardt hierarchy score, l_0 / training loss, lb / ub, least-squares estimate, loss and accuracy, loss value, lower and upper bounds, manifold capacity, max norm value change, mean auc, mean average error, mean dendrogram purity, median performance, meteor, cider, rouge-l, bleu@n, metrics, mse, mutual informations, negative log-likelihood, norm, p, p-value, pages, pearson correlation between logits, percent, performance, prec@k, precision, precision@1, predictive log likelihood, probability, probability at least $1- $, probability bound, probability of error, provably robust accuracy; robust accuracy, rank correlations, rate, rate of our bound, regret, robust accuracy, root mean squares error (mse), s, sample complexity improvement, scores, search cost, params, test error, significance test, significant influence (p > 0.05); small levels (0.03--0.04), spectral radius smaller than $1/ $, stable rank / mp soft rank, standard deviation, test accuracy, test accuracy (%), test error, test return, the supervised metrics, time, top-1 accuracy, top-1 accuracy on the test set, unsupervised clustering accuracy, upper bound, variance, word error rate (wer); normalised discounted cumulative gain (ndcg)
**Varying settings:** metric

**Evidence claims:**
- `arxiv:1711.05144v1#c02` (arxiv:1711.05144v1 — "Preventing Fairness Gerrymandering: Auditing and Learning for Subgroup Fairness", 2018, negative): A subgroup covering 67.4% of datapoints had a false positive rate 0.26 above the base rate, a 61% increase.
- `arxiv:2002.07962v1#c04` (arxiv:2002.07962v1 — "Inductive representation learning on temporal graphs", 2020, mixed): For unseen-node evaluation, 10% of nodes are sampled, masked during training, and treated as unseen nodes whose interactions are only considered in validation and testing.
- `arxiv:2002.07962v1#c05` (arxiv:2002.07962v1 — "Inductive representation learning on temporal graphs", 2020, mixed): The evaluation uses a chronological train-validation-test split of 70%-15%-15% based on node interaction timestamps.
- `arxiv:1904.00760v1#c05` (arxiv:1904.00760v1 — "Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet", 2019, negative): VGG-16 performance is only modestly reduced on texturised images relative to clean images.
- `arxiv:1711.00138v1#c04` (arxiv:1711.00138v1 — "Visualizing and Understanding Atari Agents", 2018, negative): Without saliency videos, respondents mainly identified the ball.
- `arxiv:1801.02613v1#c04` (arxiv:1801.02613v1 — "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality", 2018, mixed): Only a small fraction of BIM-b adversarial examples exhibit higher Bayesian uncertainties than normal examples, unlike other attack strategies.
- `arxiv:1803.00047v1#c05` (arxiv:1803.00047v1 — "Analyzing Uncertainty in Neural Machine Translation", 2018, negative): On the En-De dataset, humans prefer the reference over the model output 80% of the time.
- `arxiv:2006.14610v1#c03` (arxiv:2006.14610v1 — "A causal view of compositional zero-shot recognition", 2020, negative): A 70% label noise level is too noisy for evaluating noun-attribute compositionality.
- `arxiv:1909.06161v1#c04` (arxiv:1909.06161v1 — "Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs", 2019, mixed): Below 70% ImageNet top-1 performance, models show strong correlation with Brain-Score, but above 70% the correlation is not significant.
- `arxiv:1909.06674v1#c01` (arxiv:1909.06674v1 — "A Step Toward Quantifying Independently Reproducible Machine Learning Research", 2019, positive): Among the 255 selected papers, 162 (63.5%) were successfully replicated.
- `arxiv:1909.06674v1#c02` (arxiv:1909.06674v1 — "A Step Toward Quantifying Independently Reproducible Machine Learning Research", 2019, positive): The study defined a paper as reproducible when at least 75% of its claims could be confirmed using independently written code.
- `arxiv:1909.06674v1#c03` (arxiv:1909.06674v1 — "A Step Toward Quantifying Independently Reproducible Machine Learning Research", 2019, mixed): As a last-resort criterion, replication was considered successful if results were within 10% of the paper's reported numbers, while non-quantitative results required subjective comparison.
- `arxiv:1909.06674v1#c04` (arxiv:1909.06674v1 — "A Step Toward Quantifying Independently Reproducible Machine Learning Research", 2019, positive): The observed reproducibility rate was significantly better than a 26% reproducibility estimate from prior work.
- `arxiv:1905.13344v1#c01` (arxiv:1905.13344v1 — "Deterministic PAC-Bayesian generalization bounds for deep networks via generalizing noise-resilience", 2019, positive): For larger depth, the bound with 5%-$ $ outperforms all other bounds, including existing bounds.
- `arxiv:1905.13344v1#c02` (arxiv:1905.13344v1 — "Deterministic PAC-Bayesian generalization bounds for deep networks via generalizing noise-resilience", 2019, positive): Hypothetical variants using 5%-$ $ or median-$ $ perform orders of magnitude better than the actual bound.
- `arxiv:1904.10120v1#c05` (arxiv:1904.10120v1 — "Semi-Cyclic Stochastic Gradient Descent", 2019, mixed): The tweet dataset was partitioned into six time-based components and ten day-based cycles with a 90/10 train-test split.
- `arxiv:1910.08264v1#c04` (arxiv:1910.08264v1 — "Learning Compositional Koopman Operators for Model-Based Control", 2020, mixed): The experiments use a 90% training and 10% testing split.
- `arxiv:1703.05698v1#c03` (arxiv:1703.05698v1 — "Neural Sketch Learning for Conditional Program Generation", 2018, negative): Metrics are worse for both models when evaluating the worst 5% of cases.
- `arxiv:1808.02651v1#c03` (arxiv:1808.02651v1 — "Beyond pixel norm-balls: Parametric adversaries using an analytically differentiable renderer", 2019, mixed): Random and adversarial attacks were compared using success rate under small perturbation constraints.
- `arxiv:1905.12614v1#c05` (arxiv:1905.12614v1 — "Unsupervised Model Selection for Variational Disentangled Representation Learning", 2020, positive): The metric is more accurate than the other one.
- `arxiv:1808.09351v1#c05` (arxiv:1808.09351v1 — "3D-Aware Scene Manipulation via Inverse Graphics", 2018, mixed): Object-wise evaluations include only objects meeting size, occlusion, and truncation thresholds.
- `arxiv:1805.10915v1#c03` (arxiv:1805.10915v1 — "Dirichlet-based Gaussian Processes for Large-scale Calibrated Classification", 2018, mixed): The experiments use an 80%-20% split for training and Platt scaling calibration.
- `arxiv:2002.12174v1#c03` (arxiv:2002.12174v1 — "Optimistic Exploration even with a Pessimistic Initialisation", 2020, mixed): For the Randomised Chain experiment, results are reported as the median over 20 seeds with the 25%-75% quartile shaded.
- `arxiv:2002.12174v1#c04` (arxiv:2002.12174v1 — "Optimistic Exploration even with a Pessimistic Initialisation", 2020, mixed): For the Maze experiment, results are reported as the median over 8 seeds with the 25%-75% quartile shaded.
- `arxiv:2002.12174v1#c05` (arxiv:2002.12174v1 — "Optimistic Exploration even with a Pessimistic Initialisation", 2020, mixed): For the Montezuma's Revenge experiment, results are reported as the median over 4 seeds with the 25%-75% quartile shaded.
- `arxiv:1802.04381v1#c05` (arxiv:1802.04381v1 — "Classification from Pairwise Similarity and Unlabeled Data", 2018, positive): Some methods outperform others in benchmark comparisons under a one-sided t-test at 5% significance.
- `arxiv:1906.00588v1#c02` (arxiv:1906.00588v1 — "Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel", 2020, negative): Achieving accurate 68% CI coverage is harder than achieving accurate 95% or 90% CI coverage.
- `arxiv:1906.00588v1#c03` (arxiv:1906.00588v1 — "Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel", 2020, mixed): The estimated 90% CI can cover slightly more than 90% of test outcomes because mean and variance estimation errors cancel each other by chance.
- `arxiv:1906.00588v1#c04` (arxiv:1906.00588v1 — "Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel", 2020, negative): Residuals between observed coverage percentages and target coverage levels remain biased.
- `arxiv:1906.00588v1#c05` (arxiv:1906.00588v1 — "Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel", 2020, positive): Z-scores are used to reduce bias in coverage evaluation under a Gaussian outcome assumption.
- `arxiv:1805.10886v1#c05` (arxiv:1805.10886v1 — "Importance Weighted Transfer of Samples in Reinforcement Learning", 2018, mixed): Results are averaged over 20 runs and reported with 95% confidence intervals.
- `arxiv:2011.08596v1#c03` (arxiv:2011.08596v1 — "Learning outside the Black-Box: The pursuit of interpretable models", 2020, mixed): For each experiment, the dataset is split into 80% training and 20% test sets.
- `arxiv:2002.03478v1#c01` (arxiv:2002.03478v1 — "Interpretable Off-Policy Evaluation in Reinforcement Learning by Highlighting Influential Transitions", 2020, positive): In one medical simulator example, the value estimation error was less than 1%, and the method labeled the dataset as reliable.
- `arxiv:2002.03478v1#c02` (arxiv:2002.03478v1 — "Interpretable Off-Policy Evaluation in Reinforcement Learning by Highlighting Influential Transitions", 2020, mixed): Under less stochastic dynamics in case (c), evaluation achieves 8% error, and the method identifies influential transitions for domain expert review.
- `arxiv:2002.03478v1#c03` (arxiv:2002.03478v1 — "Interpretable Off-Policy Evaluation in Reinforcement Learning by Highlighting Influential Transitions", 2020, positive): Transitions that affect the FQE value estimate by more than 5% are flagged for expert review.
- `arxiv:2002.03478v1#c04` (arxiv:2002.03478v1 — "Interpretable Off-Policy Evaluation in Reinforcement Learning by Highlighting Influential Transitions", 2020, mixed): For computational convenience, the MIMIC-III cohort was down-sampled so that 20% (346) ICU stays were used for policy learning and another 20% (346) for evaluation via FQE and influence analysis.
- `arxiv:2007.05003v1#c04` (arxiv:2007.05003v1 — "Active Learning on Attributed Graphs via Graph Cognizant Logistic Regression and Preemptive Query Generation", 2020, positive): Marked methods had a statistically significant difference from the best performing baseline under a Wilcoxon ranking test at the 5% level.
- `arxiv:1905.12202v1#c02` (arxiv:1905.12202v1 — "Empirically Measuring Concentration: Fundamental Limits on Intrinsic Robustness", 2019, positive): On CIFAR-10, our estimate is 29.21% versus 52.96% reported in Madry et al.
- `arxiv:1905.12202v1#c03` (arxiv:1905.12202v1 — "Empirically Measuring Concentration: Fundamental Limits on Intrinsic Robustness", 2019, positive): Using our method on CIFAR-10, the error-region measure increases by only 0.69% after expansion with l2=0.4905.
- `arxiv:1905.12202v1#c04` (arxiv:1905.12202v1 — "Empirically Measuring Concentration: Fundamental Limits on Intrinsic Robustness", 2019, positive): On CIFAR-10, the resulting error-region measure can increase from 5.94% to 18.13% after expansion with ε=8/255.
- `arxiv:1807.00493v1#c03` (arxiv:1807.00493v1 — "Active Testing: An Efficient and Robust Framework for Estimating Accuracy", 2018, positive): MEEC sampling with a learned estimator can achieve estimates within 2-3% of real estimates at 50% vetting effort.
- `arxiv:1806.02329v1#c03` (arxiv:1806.02329v1 — "Mitigating Bias in Adaptive Data Gathering via Differential Privacy", 2018, negative): In linear contextual bandits, observed p-values frequently fell below the 0.05 threshold, indicating inflated false discoveries relative to the nominal 5% rate.
- `arxiv:1902.03227v1#c03` (arxiv:1902.03227v1 — "Minimal Images in Deep Neural Networks: Fragile Object Recognition in Natural Images", 2019, mixed): Human success rates differ by 14.5% between DNN fragile recognition images and their incorrect counterparts.
- `arxiv:1902.03227v1#c04` (arxiv:1902.03227v1 — "Minimal Images in Deep Neural Networks: Fragile Object Recognition in Natural Images", 2019, negative): DNNs show an average confidence gap of 56% for the same image pairs.
- `arxiv:1902.03227v1#c05` (arxiv:1902.03227v1 — "Minimal Images in Deep Neural Networks: Fragile Object Recognition in Natural Images", 2019, negative): Overlap between human minimal image maps and DNN FRI maps is below 6% across tested images.
- `arxiv:1902.09592v1#c01` (arxiv:1902.09592v1 — "Verification of Non-Linear Specifications for Neural Networks", 2019, negative): The gap between the adversarial and verification bound is smaller than 0.9% for all perturbations up to 6/255.
- `arxiv:1902.09592v1#c02` (arxiv:1902.09592v1 — "Verification of Non-Linear Specifications for Neural Networks", 2019, negative): For perturbation 2/255, the gap between the adversarial and verification bound is less than 1.5%.
- `arxiv:1902.09592v1#c03` (arxiv:1902.09592v1 — "Verification of Non-Linear Specifications for Neural Networks", 2019, negative): As N increases from 2 to 4, the nominal percentage satisfying the specification decreases from 72.6% to 53.3%.
- `arxiv:1905.13565v1#c03` (arxiv:1905.13565v1 — "Enhancing Simple Models by Exploiting What They Already Know", 2020, mixed): The evaluation reports averaged percentage errors with 95% confidence intervals on six real datasets.
- `arxiv:1905.13565v1#c04` (arxiv:1905.13565v1 — "Enhancing Simple Models by Exploiting What They Already Know", 2020, mixed): Results for all methods are averaged over 10 random splits and reported with 95% confidence intervals.
- `arxiv:1905.03826v1#c04` (arxiv:1905.03826v1 — "Random Function Priors for Correlation Modeling", 2019, mixed): For each test document, the evaluation uses a 90%/10% split into training and testing words.
- `arxiv:1807.01961v1#c01` (arxiv:1807.01961v1 — "A Boo(n) for Evaluating Architecture Performance", 2018, mixed): Best single model results have 95% confidence intervals when compared against Boo_5 performance across different result-pool sizes.
- `arxiv:1807.01961v1#c02` (arxiv:1807.01961v1 — "A Boo(n) for Evaluating Architecture Performance", 2018, mixed): Test performance was summarized with averages and 95% confidence intervals for three aggregation methods across different numbers of experiments.
- `arxiv:1807.01961v1#c03` (arxiv:1807.01961v1 — "A Boo(n) for Evaluating Architecture Performance", 2018, negative): For AS Reader, random hyperparameter sampling increased the interquartile range to 2.9%.
- `arxiv:1807.01961v1#c04` (arxiv:1807.01961v1 — "A Boo(n) for Evaluating Architecture Performance", 2018, mixed): Median differences between published results on the two datasets were 0.86% and 1.15%, respectively.
- `arxiv:1807.01961v1#c06` (arxiv:1807.01961v1 — "A Boo(n) for Evaluating Architecture Performance", 2018, mixed): With fixed hyperparameters, the models' accuracies had interquartile ranges of 0.98% and 1.20% absolute.
- `arxiv:1812.08255v1#c01` (arxiv:1812.08255v1 — "Automatic Classifiers as Scientific Instruments: One Step Further Away from Ground-Truth", 2019, mixed): For n=14 and n=20, the probabilities are non-negligible at around 10-15%.
- `arxiv:1802.03916v1#c01` (arxiv:1802.03916v1 — "Detecting and Correcting for Label Shift with Black Box Predictors", 2018, mixed): A performance comparison is reported across Breast, Cleveland, Glass2, Credit, Horse, Meta, Pima, and Vehicle using accuracy values with uncertainties.
- `arxiv:1802.03493v1#c02` (arxiv:1802.03493v1 — "More Robust Doubly Robust Off-policy Evaluation", 2018, positive): Increasing the sample size of evaluation trajectories improves accuracy for all estimators across experiments.
- `arxiv:1802.03493v1#c06` (arxiv:1802.03493v1 — "More Robust Doubly Robust Off-policy Evaluation", 2018, mixed): The reinforcement learning accuracy and significance results are averaged over 100 runs with different randomly generated test behavioral trajectories.
- `arxiv:2002.03518v1#c05` (arxiv:2002.03518v1 — "Multilingual Alignment of Contextual Word Representations", 2020, mixed): Keeping only word pairs found in both source-target and target-source directions trades off coverage for accuracy and yields a reasonably high-precision dataset.
- `arxiv:1901.11448v1#c04` (arxiv:1901.11448v1 — "Feature-Critic Networks for Heterogeneous Domain Generalization", 2019, mixed): All methods are evaluated using average multi-class classification accuracy and VD-Score on target domains.
- `arxiv:1901.11448v1#c05` (arxiv:1901.11448v1 — "Feature-Critic Networks for Heterogeneous Domain Generalization", 2019, mixed): Results are reported as recognition accuracy and VD scores on four held-out target datasets in Visual Decathlon using a ResNet-18 extractor.
- `arxiv:1801.03558v1#c05` (arxiv:1801.03558v1 — "Inference Suboptimality in Variational Autoencoders", 2018, positive): AIS has an advantage over the IWAE bound for evaluation when the inference network overfits to the training data.
- `arxiv:1802.07088v1#c05` (arxiv:1802.07088v1 — "i-RevNet: Deep invertible networks", 2018, negative): Classification accuracy only decreases significantly when d is 200.
- `arxiv:1810.12247v1#c02` (arxiv:1810.12247v1 — "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset", 2019, positive): In MAESTRO, audio and MIDI files are aligned with 3 ms accuracy.
- `arxiv:1810.12247v1#c04` (arxiv:1810.12247v1 — "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset", 2019, mixed): For initial alignment, a 4096-sample (90 ms) hop length was chosen as a speed-accuracy trade-off and was reasonable for most repertoire.
- `arxiv:1709.04348v1#c05` (arxiv:1709.04348v1 — "Natural Language Inference over Interaction Space", 2018, positive): The evaluation metric for all datasets is accuracy.
- `arxiv:1805.10032v1#c04` (arxiv:1805.10032v1 — "Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance", 2019, mixed): Top-1 accuracy of MLP on MNIST is evaluated under sign-flipping attack with different hyperparameters.
- `arxiv:1805.10032v1#c05` (arxiv:1805.10032v1 — "Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance", 2019, mixed): Top-1 accuracy of CNN on CIFAR-10 is evaluated under omniscient attack with different hyperparameters.
- `arxiv:1902.09432v1#c04` (arxiv:1902.09432v1 — "Scalable and Order-robust Continual Learning with Additive Parameter Decomposition", 2020, mixed): Order-robustness results are reported as mean accuracy across multiple task orders and random trials.
- `arxiv:1905.13192v1#c04` (arxiv:1905.13192v1 — "Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels", 2019, mixed): Test accuracy is correlated with the dataset and the architecture.
- `arxiv:2003.04475v1#c06` (arxiv:2003.04475v1 — "Domain Adaptation with Conditional Distribution Matching and Generalized Label Shift", 2020, positive): Outperforming is defined as having larger test accuracy than the base version for a given seed.
- `arxiv:1812.09764v1#c01` (arxiv:1812.09764v1 — "Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology", 2019, mixed): Mean normalized neural persistence was compared against validation loss for early stopping across multiple datasets, showing differences in accuracy and stopping epoch.
- `arxiv:1812.09764v1#c03` (arxiv:1812.09764v1 — "Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology", 2019, negative): A naive extension using neural persistence of a convolutional layer for stopping can cause a considerable loss of accuracy.
- `arxiv:1812.09764v1#c04` (arxiv:1812.09764v1 — "Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology", 2019, positive): For one dataset, most parameter configurations stop earlier than validation loss and can slightly increase accuracy.
- `arxiv:1812.09764v1#c06` (arxiv:1812.09764v1 — "Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology", 2019, positive): Across 625 configurations, stopping based on neural persistence stops on average half an epoch earlier than validation loss while losing virtually no accuracy.
- `arxiv:1803.06084v1#c04` (arxiv:1803.06084v1 — "A Kernel Theory of Modern Data Augmentation", 2019, positive): The authors validate a correlation between accuracy and kernel target alignment on MNIST across multiple augmentations.
- `arxiv:1810.02334v1#c05` (arxiv:1810.02334v1 — "Unsupervised Learning via Meta-Learning", 2019, mixed): Experimental performance is measured as classification accuracy averaged over 1000 tasks.
- `arxiv:2006.01095v1#c05` (arxiv:2006.01095v1 — "Emergence of Separable Manifolds in Deep Language Representations", 2020, mixed): The trend in linear separability across layer depth depends on training set size.
- `arxiv:2006.01095v1#c06` (arxiv:2006.01095v1 — "Emergence of Separable Manifolds in Deep Language Representations", 2020, negative): Accuracy decreases across layers.
- `arxiv:2002.10648v1#c02` (arxiv:2002.10648v1 — "I Am Going MAD: Maximum Discrepancy Competition for Comparing Classifiers Adaptively", 2020, positive): MAD tracks steady progress in image classification, with high rank correlation to ImageNet validation accuracy rankings.
- `arxiv:2002.10648v1#c04` (arxiv:2002.10648v1 — "I Am Going MAD: Maximum Discrepancy Competition for Comparing Classifiers Adaptively", 2020, mixed): MAD should be considered complementary to conventional accuracy comparison rather than a replacement.
- `arxiv:2009.00666v1#c03` (arxiv:2009.00666v1 — "Robust, Accurate Stochastic Optimization for Variational Inference", 2020, positive): For the Radon model, the ELBO criterion never reaches convergence and yields better posterior mean accuracy and a smaller k statistic than using MCSE.
- `arxiv:1906.00930v1#c01` (arxiv:1906.00930v1 — "A Necessary and Sufficient Stability Notion for Adaptive Generalization", 2019, mixed): In adaptive processes, accuracy is defined with respect to the adversary.
- `arxiv:1906.00930v1#c04` (arxiv:1906.00930v1 — "A Necessary and Sufficient Stability Notion for Adaptive Generalization", 2019, negative): If the sample-accuracy exceedance probability is bounded, then the probability of at least one failure in T independent iterations is less than T times that bound.
- `arxiv:2006.14748v1#c05` (arxiv:2006.14748v1 — "Proper Network Interpretability Helps Adversarial Robustness in Classification", 2020, mixed): The experiments evaluate 200-step PGD accuracy across different perturbation sizes on MNIST, CIFAR-10, and Restricted ImageNet.
- `arxiv:1806.03276v1#c03` (arxiv:1806.03276v1 — "Approximate message passing for amplitude based optimization", 2018, positive): Simulation results verify the accuracy of SE predictions with the proposed initialization.
- `arxiv:1810.11551v1#c04` (arxiv:1810.11551v1 — "Estimators for Multivariate Information Measures in General Probability Spaces", 2018, mixed): A low-dimensional manifold condition in a high-dimensional space was examined for its effect on estimator accuracy.
- `arxiv:1902.02783v1#c03` (arxiv:1902.02783v1 — "Humor in Word Embeddings: Cockamamie Gobbledegook for Nincompoops", 2019, positive): A linear SVM achieves mean accuracy of 96.7% in identifying an average humor direction.
- `arxiv:1807.01961v1#c05` (arxiv:1807.01961v1 — "A Boo(n) for Evaluating Architecture Performance", 2018, positive): Resnet had mean test accuracy of 68.41% with a standard deviation of 0.67%.
- `arxiv:1907.02044v1#c06` (arxiv:1907.02044v1 — "Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack", 2020, mixed): The experiments evaluate attack effectiveness using upper bounds on robust accuracy for fixed perturbation thresholds on MNIST and CIFAR-10 adversarially trained models.
- `arxiv:1904.00760v1#c06` (arxiv:1904.00760v1 — "Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet", 2019, positive): BagNet variants show increasing Pearson correlation of logits with VGG-16 as receptive field size grows.
- `arxiv:1801.08930v1#c03` (arxiv:1801.08930v1 — "Recasting Gradient-Based Meta-Learning as Hierarchical Bayes", 2018, mixed): Results are averaged over 600 test episodes and reported with 95% confidence intervals.
- `arxiv:1802.03493v1#c01` (arxiv:1802.03493v1 — "More Robust Doubly Robust Off-policy Evaluation", 2018, positive): MRDR significantly outperforms DR in the contextual bandit experiments.
- `arxiv:1802.03493v1#c04` (arxiv:1802.03493v1 — "More Robust Doubly Robust Off-policy Evaluation", 2018, mixed): The experiments compare MRDR against DM, IS, DR, and DR0 using MSE.
- `arxiv:1907.09623v1#c04` (arxiv:1907.09623v1 — "Doubly robust off-policy evaluation with shrinkage", 2020, mixed): MSE is evaluated as the number of samples varies on the yeast dataset under a specified logging policy, for both deterministic and stochastic rewards.
- `arxiv:1802.03493v1#c05` (arxiv:1802.03493v1 — "More Robust Doubly Robust Off-policy Evaluation", 2018, mixed): Estimation accuracy in the contextual bandit experiments is evaluated with root mean squares error (MSE).
- `arxiv:1810.00760v1#c01` (arxiv:1810.00760v1 — "Riemannian Adaptive Optimization Methods", 2019, mixed): For link prediction experiments, a 2% validation set of edges is sampled.
- `arxiv:1810.00760v1#c02` (arxiv:1810.00760v1 — "Riemannian Adaptive Optimization Methods", 2019, mixed): The experiments report loss value and mean average precision (MAP).
- `arxiv:1710.11029v1#c06` (arxiv:1710.11029v1 — "Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks", 2018, positive): The diffusion matrix for CIFAR-100 has larger eigenvalues, is more non-isotropic, and has a much larger rank than that for CIFAR-10.
- `arxiv:1801.03558v1#c01` (arxiv:1801.03558v1 — "Inference Suboptimality in Variational Autoencoders", 2018, positive): On CIFAR-10, the amortization gap is much larger than the approximation gap.
- `arxiv:1801.03558v1#c04` (arxiv:1801.03558v1 — "Inference Suboptimality in Variational Autoencoders", 2018, negative): Increasing decoder capacity reduces the approximation gap.
- `arxiv:1803.08884v1#c03` (arxiv:1803.08884v1 — "Inequity aversion improves cooperation in intertemporal social dilemmas", 2018, mixed): The Cleanup game uses total contribution to the public good P as the number of waste cells cleaned.
- `arxiv:1803.08884v1#c04` (arxiv:1803.08884v1 — "Inequity aversion improves cooperation in intertemporal social dilemmas", 2018, mixed): The sustainability metric S is the average time at which rewards are collected.
- `arxiv:1811.12889v1#c05` (arxiv:1811.12889v1 — "Systematic Generalization: What Is Required and Can It Be Learned?", 2019, positive): In layout induction evaluation, the tree condition has much higher test accuracy than the chain condition.
- `arxiv:1805.08296v1#c03` (arxiv:1805.08296v1 — "Data-Efficient Hierarchical Reinforcement Learning", 2018, mixed): Evaluation reports average reward for Ant Gather and average success rate for the other tasks over 10 randomly seeded trials.
- `arxiv:1903.12436v1#c05` (arxiv:1903.12436v1 — "From variational to deterministic autoencoders", 2020, positive): Scores for random samples are evaluated against the test set.
- `arxiv:1805.10032v1#c02` (arxiv:1805.10032v1 — "Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance", 2019, mixed): The evaluation metric is top-1 accuracy on the test sets.
- `arxiv:1805.10032v1#c03` (arxiv:1805.10032v1 — "Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance", 2019, mixed): Experiments use MNIST and CIFAR10 with top-1 accuracy on the test set, and each experiment launches twenty worker processes.
- `arxiv:1905.09791v1#c03` (arxiv:1905.09791v1 — "Multi-relational Poincaré Graph Embeddings", 2019, mixed): In FB15k-237, most relations have perfect Krackhardt hierarchy score, but their subgraphs are usually shallow pairwise edges rather than trees.
- `arxiv:1711.02013v1#c02` (arxiv:1711.02013v1 — "Neural Language Modeling by Jointly Learning Syntax and Lexicon", 2018, negative): Treebank sentence flatness limits the maximum precision attainable for binary parsing evaluation.
- `arxiv:1703.01678v1#c01` (arxiv:1703.01678v1 — "Data-Dependent Stability of Stochastic Gradient Descent", 2018, mixed): In convex losses, the rate of the proposed bound is no worse than Hardt et al. up to a constant under the stated assumptions.
- `arxiv:1703.01678v1#c02` (arxiv:1703.01678v1 — "Data-Dependent Stability of Stochastic Gradient Descent", 2018, mixed): In non-convex losses, the proposed bound is no worse than Hardt et al.'s bound.
- `arxiv:1905.07773v1#c03` (arxiv:1905.07773v1 — "Online Convex Optimization in Adversarial Markov Decision Processes", 2019, mixed): The learner's performance is measured against the best stationary policy under the chosen performance criterion.
- `arxiv:1812.03849v1#c04` (arxiv:1812.03849v1 — "Weakly Supervised Dense Event Captioning in Videos", 2018, mixed): Performance is measured using METEOR, CIDEr, Rouge-L, and Bleu@N.
- `arxiv:1812.03849v1#c05` (arxiv:1812.03849v1 — "Weakly Supervised Dense Event Captioning in Videos", 2018, mixed): Captioning metrics are computed only for proposals whose overlap with ground-truth exceeds a tIoU threshold; otherwise the score is set to 0.
- `arxiv:1812.09764v1#c02` (arxiv:1812.09764v1 — "Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology", 2019, positive): Neural persistence based stopping yields slightly better test accuracy than training and validation loss when only a fraction of the data is available.
- `arxiv:1812.09764v1#c05` (arxiv:1812.09764v1 — "Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology", 2019, negative): When only a fraction of the data is available, validation-loss-based stopping gives inferior test accuracy relative to training-loss-based stopping.
- `arxiv:1902.07816v1#c05` (arxiv:1902.07816v1 — "Mixture Models for Diverse Machine Translation: Tricks of the Trade", 2019, mixed): Good average oracle BLEU together with extremely high pairwise BLEU indicates that latent values produce good but very similar outputs.
- `arxiv:1802.03063v1#c03` (arxiv:1802.03063v1 — "Learning Latent Representations in Neural Networks for Clustering through Pseudo Supervision and Graph-based Activity Regularization", 2018, mixed): The experiments evaluate test performance using unsupervised clustering accuracy.
- `arxiv:1906.05243v1#c05` (arxiv:1906.05243v1 — "When to use parametric models in reinforcement learning?", 2019, positive): These updates are guaranteed to converge under the stated positive semi-definite and spectral-radius conditions.
- `arxiv:1802.06936v1#c02` (arxiv:1802.06936v1 — "Online Learning with an Unknown Fairness Metric", 2018, negative): The ε-fairness loss of an algorithm is the total number of pairs on which it is ε-unfair over time.
- `arxiv:1901.09749v1#c06` (arxiv:1901.09749v1 — "Fairwashing: the risk of rationalization", 2019, mixed): Test sets are used to evaluate the accuracy and unfairness of the black-box models.
- `arxiv:1911.00937v1#c05` (arxiv:1911.00937v1 — "Preventing Gradient Attenuation in Lipschitz Constrained Convolutional Networks", 2019, positive): The reported evaluation includes provably robust accuracy under the certification criterion in Equation 9 and robust accuracy against two gradient-based attacks.
- `arxiv:1809.06098v1#c04` (arxiv:1809.06098v1 — "Policy Optimization via Importance Sampling", 2018, mixed): CB-POIS, PB-POIS, and TRPO are evaluated by average return as a function of the number of trajectories under a linear policy setting.
- `arxiv:1809.06098v1#c05` (arxiv:1809.06098v1 — "Policy Optimization via Importance Sampling", 2018, mixed): CB-POIS and PB-POIS are evaluated by average return as a function of the number of trajectories with deep neural policies.
- `arxiv:1909.06674v1#c05` (arxiv:1909.06674v1 — "A Step Toward Quantifying Independently Reproducible Machine Learning Research", 2019, negative): Low-readability papers were 3.17 to 5.67 pages shorter than other readability types and were the statistically significant source of the relationship.
- `arxiv:1909.06674v1#c06` (arxiv:1909.06674v1 — "A Step Toward Quantifying Independently Reproducible Machine Learning Research", 2019, negative): Author code availability did not show a statistically significant relationship in the reported chi-squared test, with p=0.184.
- `arxiv:1905.13344v1#c03` (arxiv:1905.13344v1 — "Deterministic PAC-Bayesian generalization bounds for deep networks via generalizing noise-resilience", 2019, mixed): The framework provides a bound that upper bounds test loss by training loss plus complexity terms under stated conditions.
- `arxiv:1905.13344v1#c04` (arxiv:1905.13344v1 — "Deterministic PAC-Bayesian generalization bounds for deep networks via generalizing noise-resilience", 2019, positive): The PAC-Bayesian framework yields a generalization bound on the 0-1 error of the stochastic classifier.
- `arxiv:1905.08537v1#c05` (arxiv:1905.08537v1 — "Adaptive Stochastic Natural Gradient Method for One-Shot Neural Architecture Search", 2019, mixed): The method is compared against NAS baselines using search cost, parameter count, and test error.
- `arxiv:1907.09623v1#c03` (arxiv:1907.09623v1 — "Doubly robust off-policy evaluation with shrinkage", 2020, mixed): The evaluation reports the average value of the learned policy on the test set, averaged over 10 replicates, normalized by IPS.
- `arxiv:1910.12586v1#c01` (arxiv:1910.12586v1 — "PC-Fairness: A Unified Framework for Measuring Causality-based Fairness", 2019, positive): The method can bound causal effects under unidentifiable situations and achieves tighter bounds than previous methods.
- `arxiv:2007.00717v1#c04` (arxiv:2007.00717v1 — "Adaptive Discretization for Model-Based Reinforcement Learning", 2020, positive): The stated bounds hold simultaneously with high probability for all k, h, and any partition.
- `arxiv:1910.12586v1#c02` (arxiv:1910.12586v1 — "PC-Fairness: A Unified Framework for Measuring Causality-based Fairness", 2019, positive): For indirect discrimination measurement, the method can guarantee the decision is discriminatory where prior bounds were uncertain.
- `arxiv:1910.12586v1#c03` (arxiv:1910.12586v1 — "PC-Fairness: A Unified Framework for Measuring Causality-based Fairness", 2019, positive): For counterfactual fairness of the second group, the method can guarantee the decision is fair while the comparison method remains uncertain.
- `arxiv:1910.12586v1#c04` (arxiv:1910.12586v1 — "PC-Fairness: A Unified Framework for Measuring Causality-based Fairness", 2019, positive): On the PE row for synthetic dataset D1, the method yields a much narrower bound interval than previous methods.
- `arxiv:1805.09785v1#c02` (arxiv:1805.09785v1 — "Entropy and mutual information in models of deep neural networks", 2018, mixed): If noise is assumed only at the last layer, the second-to-last layer is a deterministic mapping of the input, which lets the mutual information formula apply directly.
- `arxiv:1805.09785v1#c03` (arxiv:1805.09785v1 — "Entropy and mutual information in models of deep neural networks", 2018, mixed): The model assumes small additive white Gaussian noise before the activation at a layer to obtain entropy and mutual information while keeping the previous mapping deterministic.
- `arxiv:1802.04924v1#c03` (arxiv:1802.04924v1 — "Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks", 2018, mixed): The paper compares cost-model execution time estimates with measured per-step execution time in experiments.
- `arxiv:1905.12614v1#c01` (arxiv:1905.12614v1 — "Unsupervised Model Selection for Variational Disentangled Representation Learning", 2020, positive): The proposed unsupervised score correlates well with supervised metrics.
- `arxiv:1905.12614v1#c03` (arxiv:1905.12614v1 — "Unsupervised Model Selection for Variational Disentangled Representation Learning", 2020, mixed): Hyperparameter search results are reported for six unsupervised disentangling model classes on dSprites.
- `arxiv:1905.12614v1#c04` (arxiv:1905.12614v1 — "Unsupervised Model Selection for Variational Disentangled Representation Learning", 2020, positive): Different versions of the score show rank correlations with the metric on dSprites.
- `arxiv:1808.09351v1#c04` (arxiv:1808.09351v1 — "3D-Aware Scene Manipulation via Inverse Graphics", 2018, mixed): Geometric representation is evaluated using specific metrics for rotation, observation direction, distance, and scale.
- `arxiv:1903.05499v1#c05` (arxiv:1903.05499v1 — "DeepOBS: A Deep Learning Optimizer Benchmark Suite", 2019, positive): The authors recommend reporting both loss and accuracy on both training and test sets when presenting a new optimizer.
- `arxiv:1901.08276v1#c03` (arxiv:1901.08276v1 — "Traditional and Heavy Tailed Self Regularization in Neural Network Models", 2019, negative): Successive phases in the 5+1 phase taxonomy correspond to smaller Stable Rank/MP Soft Rank and more self-regularization.
- `arxiv:1711.02771v1#c02` (arxiv:1711.02771v1 — "On the Discrimination-Generalization Tradeoff in GANs", 2018, mixed): Using BL distance directly as the learning objective yields a similar rate of d_(μ,μ_m)=O(m^-1/d).
- `arxiv:1711.02771v1#c03` (arxiv:1711.02771v1 — "On the Discrimination-Generalization Tradeoff in GANs", 2018, negative): With high probability, d_(μ,μ_m) is bounded by d_(μ,ν) plus additional C m + terms.
- `arxiv:2007.05145v1#c05` (arxiv:2007.05145v1 — "Beyond Perturbations: Learning Guarantees with Arbitrary Adversarial Test Examples", 2020, positive): The transductive train-test analysis provides probability bounds for any classifier class over any distribution.
- `arxiv:2002.02693v1#c04` (arxiv:2002.02693v1 — "Ready Policy One: World Building Through Active Learning", 2020, positive): Performance is summarized by median performance across 10 seeds.
- `arxiv:1805.10915v1#c04` (arxiv:1805.10915v1 — "Dirichlet-based Gaussian Processes for Large-scale Calibrated Classification", 2018, mixed): Calibration is assessed using Expected Calibration Error.
- `arxiv:2006.01095v1#c02` (arxiv:2006.01095v1 — "Emergence of Separable Manifolds in Deep Language Representations", 2020, negative): In unmasked data, POS manifold capacity decreases substantially relative to NER.
- `arxiv:2006.01095v1#c03` (arxiv:2006.01095v1 — "Emergence of Separable Manifolds in Deep Language Representations", 2020, mixed): Capacity and geometry trends are similar across different Transformer architectures.
- `arxiv:2006.01095v1#c04` (arxiv:2006.01095v1 — "Emergence of Separable Manifolds in Deep Language Representations", 2020, negative): In unmasked data, the fields distribution peak shifts toward the origin across train/test split regimes, indicating reduced separability.
- `arxiv:1901.08469v1#c03` (arxiv:1901.08469v1 — "Deep Generative Learning via Variational Gradient Flow", 2019, mixed): FID evaluations are reported as mean and standard deviation over 10k generated MNIST and FashionMNIST images with five-time bootstrap sampling.
- `arxiv:1811.11979v1#c05` (arxiv:1811.11979v1 — "Unsupervised Image-to-Image Translation Using Domain-Specific Variational Information Bound", 2018, mixed): FID is computed over 10k randomly generated samples.
- `arxiv:1802.03654v1#c04` (arxiv:1802.03654v1 — "Beyond the One-Step Greedy Approach in Reinforcement Learning", 2018, mixed): In experiments, VI was stopped when the max-norm value change fell below 10^-5, and other thresholds showed the same qualitative behavior.
- `arxiv:1910.06508v1#c01` (arxiv:1910.06508v1 — "Understanding the Curse of Horizon in Off-Policy Evaluation via Conditional Importance Sampling", 2020, mixed): PDIS yields O(T^2) variance only under a stronger corollary assumption.
- `arxiv:1910.06508v1#c02` (arxiv:1910.06508v1 — "Understanding the Curse of Horizon in Off-Policy Evaluation via Conditional Importance Sampling", 2020, positive): PDIS variance can be polynomial in T under additional reward decay conditions, implying exponential improvement over crude importance sampling for long horizons.
- `arxiv:1910.06508v1#c03` (arxiv:1910.06508v1 — "Understanding the Curse of Horizon in Off-Policy Evaluation via Conditional Importance Sampling", 2020, negative): The per-decision estimator can have larger variance than the crude estimator in a counterexample.
- `arxiv:1811.09558v1#c05` (arxiv:1811.09558v1 — "Regret bounds for meta Bayesian optimization with an unknown Gaussian process prior", 2018, negative): A high-probability bound upper-bounds cumulative regret by best-sample simple regret plus an additive term.
- `arxiv:1909.01504v1#c02` (arxiv:1909.01504v1 — "Censored Semi-Bandits: A Framework for Resource Allocation with Censored Feedback", 2019, mixed): The experiment was repeated 50 times and evaluated using regret with 95% confidence intervals.
- `arxiv:1906.00569v1#c04` (arxiv:1906.00569v1 — "Distribution oblivious, risk-aware algorithms for multi-armed   bandits with unbounded rewards", 2019, mixed): The probability of error in experiments is estimated by averaging over 50000 runs for each sampled T.
- `arxiv:1805.09495v1#c04` (arxiv:1805.09495v1 — "A Practical Algorithm for Distributed Clustering and Outlier Detection", 2018, mixed): The sizes of summaries returned by different algorithms are matched to within less than 10% difference after manual tuning.
- `arxiv:1909.09141v1#c04` (arxiv:1909.09141v1 — "Causal Modeling for Fairness In Dynamical Systems", 2020, mixed): Policy evaluation methods exhibit mean average error under model mismatch conditions.
- `arxiv:1908.01517v1#c04` (arxiv:1908.01517v1 — "Adversarial Self-Defense for Cycle-Consistent GANs", 2019, mixed): For the GTA dataset evaluation, the authors compute IoU and mean class-wise accuracy using Pix2pix-based honest reconstruction.
- `arxiv:1910.02934v1#c02` (arxiv:1910.02934v1 — "Algorithm-Dependent Generalization Bounds for Overparameterized Deep Residual Networks", 2019, mixed): The theorem translates optimization over empirical loss into optimization over empirical surrogate loss related to classification error.
- `arxiv:1802.06014v1#c04` (arxiv:1802.06014v1 — "Orthogonality-Promoting Distance Metric Learning: Convex Relaxation and Theoretical Analysis", 2018, mixed): The experiments evaluate mean AUC on MIMIC, EICU, and Reuters averaged over 5 random train/test splits.
- `arxiv:2007.06168v1#c04` (arxiv:2007.06168v1 — "Model Fusion with Kullback-Leibler Divergence", 2020, mixed): Predictive log likelihood is measured on 10% of words conditioned on the remaining 90% for each test document.
- `arxiv:1905.12202v1#c01` (arxiv:1905.12202v1 — "Empirically Measuring Concentration: Fundamental Limits on Intrinsic Robustness", 2019, positive): Our method finds regions with about half the adversarial risk of Gilmer et al. on MNIST and Fashion-MNIST, while being comparable on CIFAR-10 and SVHN.
- `arxiv:1905.12202v1#c06` (arxiv:1905.12202v1 — "Empirically Measuring Concentration: Fundamental Limits on Intrinsic Robustness", 2019, mixed): Compared with prior robust classifiers, our method yields lower empirical adversarial risk on MNIST and CIFAR-10, with slightly higher empirical risk.
- `arxiv:2008.07922v1#c05` (arxiv:2008.07922v1 — "Linear Disentangled Representations and Unsupervised Action Estimation", 2020, mixed): Independence scores show strong correlations with most other metrics, except SAP and MIG.
- `arxiv:1812.08255v1#c03` (arxiv:1812.08255v1 — "Automatic Classifiers as Scientific Instruments: One Step Further Away from Ground-Truth", 2019, mixed): Test correlations stabilize over time and converge to roughly the same value across training runs and detection architectures.
- `arxiv:1910.09388v1#c01` (arxiv:1910.09388v1 — "An Unbiased Risk Estimator for Learning with Augmented Classes", 2020, positive): The LAC risk is infinite-sample consistent with the risk over the testing distribution under 0-1 loss.
- `arxiv:1801.10242v1#c05` (arxiv:1801.10242v1 — "Low-Rank Bandit Methods for High-Dimensional Dynamic Pricing", 2019, negative): The paper evaluates average regret of pricing strategies under structural shocks in the demand model at T/3 and 2T/3.
- `arxiv:1801.10242v1#c06` (arxiv:1801.10242v1 — "Low-Rank Bandit Methods for High-Dimensional Dynamic Pricing", 2019, negative): The paper evaluates average regret of various pricing strategies when the underlying demand model drifts over time.
- `arxiv:1807.00493v1#c01` (arxiv:1807.00493v1 — "Active Testing: An Efficient and Robust Framework for Estimating Accuracy", 2018, positive): Active testing reduces algorithm misranking error compared with naive performance estimates.
- `arxiv:1807.00493v1#c02` (arxiv:1807.00493v1 — "Active Testing: An Efficient and Robust Framework for Estimating Accuracy", 2018, positive): A learned estimator with uncertainty sampling further reduces absolute error and variance.
- `arxiv:1807.00493v1#c04` (arxiv:1807.00493v1 — "Active Testing: An Efficient and Robust Framework for Estimating Accuracy", 2018, positive): For Prec@K, only the top-k lists need to be fully vetted rather than the whole test set.
- `arxiv:2011.01928v1#c03` (arxiv:2011.01928v1 — "Generalization to New Actions in Reinforcement Learning", 2020, mixed): The results are averaged over 5000 episodes across 5 random seeds, with standard deviation error bars and 8 seeds for Grid World.
- `arxiv:2007.00534v1#c02` (arxiv:2007.00534v1 — "On Convergence-Diagnostic based Step Sizes for Stochastic Gradient Descent", 2020, negative): The standard deviation of S_n does not decay with the step size after leaving stationarity, but only with the averaging window as 1/n.
- `arxiv:1906.07859v1#c06` (arxiv:1906.07859v1 — "Supervised Hierarchical Clustering with Exponential Linkage", 2019, mixed): The evaluation reports mean dendrogram purity for each training method/linkage pair on four datasets, averaged over 50 randomly generated train/dev/test splits.
- `arxiv:1710.10230v1#c04` (arxiv:1710.10230v1 — "Not-So-Random Features", 2018, positive): A generalization-style bound holds with probability at least 1-δ under the stated sample-size condition.
- `arxiv:2010.12055v1#c02` (arxiv:2010.12055v1 — "A Discrete Variational Recurrent Topic Model without the Reparametrization Trick", 2020, positive): SwitchP had high positive correlation with human judgments.
- `arxiv:2010.12055v1#c04` (arxiv:2010.12055v1 — "A Discrete Variational Recurrent Topic Model without the Reparametrization Trick", 2020, mixed): Average document-level topic entropy was measured on the test sets across three runs.
- `arxiv:1811.08790v1#c03` (arxiv:1811.08790v1 — "Learning Quadratic Games on Networks", 2020, mixed): AUC is used to evaluate learning performance because positive cases are rare across all three graph types.
- `arxiv:2006.12620v1#c05` (arxiv:2006.12620v1 — "A Maximum-Entropy Approach to Off-Policy Evaluation in Average-Reward MDPs", 2020, positive): For off-policy evaluation, the method uses a least-squares estimate.
- `arxiv:1901.01365v1#c04` (arxiv:1901.01365v1 — "Hierarchical Reinforcement Learning via Advantage-Weighted Information Maximization", 2019, mixed): The experiments were run five times with different seeds and reported averaged test return.
- `arxiv:1910.13895v1#c03` (arxiv:1910.13895v1 — "Learning Deterministic Weighted Automata with Queries and Counterexamples", 2019, positive): The extracted models are evaluated against their target RNNs using WER and NDCG, with NDCG being the SPiCe scoring function.
- `arxiv:1909.05557v1#c03` (arxiv:1909.05557v1 — "Modular Meta-Learning with Shrinkage", 2020, mixed): The algorithms are compared by generalization loss after adaptation to a new task with multiple gradient descent steps, using negative log-likelihood up to a constant as the loss.
- `arxiv:1909.05557v1#c05` (arxiv:1909.05557v1 — "Modular Meta-Learning with Shrinkage", 2020, mixed): Results are reported as mean generalization loss with 68% credible intervals over 1000 test tasks versus test-time adaptation steps.
- `arxiv:1810.07155v1#c01` (arxiv:1810.07155v1 — "Hunting for Discriminatory Proxies in Linear Regression Models", 2018, mixed): Proxies for race were somewhat stronger than proxies for gender, but neither had significant influence beyond small levels.
- `arxiv:1810.07155v1#c02` (arxiv:1810.07155v1 — "Hunting for Discriminatory Proxies in Linear Regression Models", 2018, negative): The estimate is generally about 3--4 times larger than the actual influence.
- `arxiv:2006.11827v1#c01` (arxiv:2006.11827v1 — "Refined bounds for algorithm configuration: The knife-edge of dual class approximability", 2020, positive): The proposed bounds are significantly stronger than the best-known worst-case guarantees in integer programming algorithm configuration, yielding a large sample complexity improvement.
- `arxiv:2006.11827v1#c03` (arxiv:2006.11827v1 — "Refined bounds for algorithm configuration: The knife-edge of dual class approximability", 2020, mixed): A worst-case generalization bound holds with high probability over N samples for all parameters r in [0,1].
- `arxiv:2006.11827v1#c04` (arxiv:2006.11827v1 — "Refined bounds for algorithm configuration: The knife-edge of dual class approximability", 2020, mixed): For integer programs, the empirical-expected performance gap is upper-bounded by the minimum of the worst-case bound and another bound, under stated high-probability conditions.
- `arxiv:2006.16540v1#c02` (arxiv:2006.16540v1 — "Associative Memory in Iterated Overparameterized Sigmoid Autoencoders", 2020, mixed): There should be n-1 eigenvalues with norm around 1.
- `arxiv:2006.16540v1#c04` (arxiv:2006.16540v1 — "Associative Memory in Iterated Overparameterized Sigmoid Autoencoders", 2020, mixed): A symmetric matrix is characterized by eigenvalue 1 with multiplicity m-1, eigenvalue 0 with multiplicity k-m, and another eigenvalue between 0 and 1.
- `arxiv:1901.11143v1#c03` (arxiv:1901.11143v1 — "Natural Analysts in Adaptive Data Analysis", 2019, positive): For arbitrarily large t, h_t is differentially private with the stated parameterization.
- `arxiv:1806.06086v1#c05` (arxiv:1806.06086v1 — "Minibatch Gibbs Sampling on Large Graphical Models", 2018, negative): The evaluation uses average ℓ2-distance error in estimated marginals relative to the fully-mixed state.
- `arxiv:2007.00717v1#c05` (arxiv:2007.00717v1 — "Adaptive Discretization for Model-Based Reinforcement Learning", 2020, mixed): A Wasserstein-distance bound holds uniformly over all h, k, and B with high probability.
- `arxiv:2007.01488v1#c01` (arxiv:2007.01488v1 — "On the Relation between Quality-Diversity Evaluation and Distribution-Fitting Goal in Text Generation", 2020, negative): The commonly used BLEU and Self-BLEU metric pair fails to reflect the distribution-fitting goal.
- `arxiv:2007.01488v1#c02` (arxiv:2007.01488v1 — "On the Relation between Quality-Diversity Evaluation and Distribution-Fitting Goal in Text Generation", 2020, positive): Ground truth text data are clearly outperformed on both BLEU and NSBLEU by a manually constructed model.
- `arxiv:2007.01488v1#c03` (arxiv:2007.01488v1 — "On the Relation between Quality-Diversity Evaluation and Distribution-Fitting Goal in Text Generation", 2020, negative): The BLEU-NSBLEU metric pair is divergence-incompatible when the reference set size is usually far more than 1.
- `arxiv:2007.01488v1#c04` (arxiv:2007.01488v1 — "On the Relation between Quality-Diversity Evaluation and Distribution-Fitting Goal in Text Generation", 2020, positive): The time complexity of the CR-NRR algorithm is O(m+n), much lower than BLEU-NSBLEU's O(m(m+n)).
- `arxiv:1812.05551v1#c04` (arxiv:1812.05551v1 — "Exploration Conscious Reinforcement Learning Revisited", 2019, mixed): Test error was evaluated using a fixed value iteration procedure with high precision.
- `arxiv:1807.09387v1#c05` (arxiv:1807.09387v1 — "Learning from Delayed Outcomes via Proxies with Applications to Recommender Systems", 2019, mixed): The experiments compare average cumulative error of three forecasters over 200 independent trials under both adversarial schedules.
- `arxiv:1910.04817v1#c03` (arxiv:1910.04817v1 — "Estimation of Bounds on Potential Outcomes For Decision Making", 2020, mixed): The table reports achieved FCR and mean IW for the proposed model and baselines averaged over 20 simulations.
- `arxiv:2004.00198v1#c04` (arxiv:2004.00198v1 — "Extreme Multi-label Classification from Aggregated Labels", 2020, mixed): An ablation study compares Precision@1 across no label assignment, no label learning, and label learning settings.
- `arxiv:2012.01866v1#c04` (arxiv:2012.01866v1 — "Make One-Shot Video Object Segmentation Efficient Again", 2020, negative): On the test-dev set, all methods perform substantially worse across all metrics than on the validation set.
- `arxiv:1806.06931v1#c03` (arxiv:1806.06931v1 — "Reinforcement Learning with Function-Valued Action Spaces for Partial Differential Equation Control", 2018, positive): Upper bounds hold for the logarithm of the covering number under the stated constants.
- `arxiv:1905.03826v1#c05` (arxiv:1905.03826v1 — "Random Function Priors for Correlation Modeling", 2019, mixed): The experiments report corpus train/test/vocabulary/token counts for New York Times, 20Newsgroups, and NeurIPS.
- `arxiv:1811.10121v1#c05` (arxiv:1811.10121v1 — "Foreground Clustering for Joint Segmentation and Localization in Videos and Images", 2018, mixed): Segmentation accuracy is measured using Intersection over Union (IoU), also known as the Jaccard index.
- `arxiv:1811.01129v1#c03` (arxiv:1811.01129v1 — "Efficient Projection onto the Perfect Phylogeny Model", 2018, mixed): Runtime experiments were conducted on trees with 1000 nodes.

### h170 — One metric rewards confidence calibration or uncertainty coverage, while the other rewards sharp point predictions or classification accuracy, producing opposite conclusions for the same probabilistic outputs.

**Mechanism.** Coverage/ECE-type metrics evaluate whether predicted probabilities or intervals match empirical frequencies, while accuracy or point error mainly reward decisiveness and correctness at a threshold; overconfident or underconfident models can score well on one and poorly on the other using identical outputs.

**Predictions:**
- Temperature scaling improves calibration metric more than accuracy
- Wider intervals raise coverage while worsening sharpness/point error

**Minimal test.** Obtain probabilistic outputs from the same model on a shared eval set: class probabilities for classification or predictive means/intervals for uncertainty tasks. Cross-score the exact same predictions with a calibration metric (ECE or interval coverage residual) and a correctness metric (accuracy, NLL, or point error). Then generate paired variants from the same outputs by temperature scaling probabilities or widening/narrowing predictive intervals, and rescore with both metrics to test whether calibration improves while accuracy/sharpness does not.

**Scope.** method=evaluation-method, task=evaluation

**Evidence gap.** Current claims indicate calibration and coverage concerns, but they do not report side-by-side calibration-versus-accuracy outcomes on a matched prediction set.

**Graph bridge.** Expected Calibration Error → accuracy

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.84 | 0.89 | 1.00 | 0.53 | 0.00 | 0.94 |

### Anomaly a514 — community_disconnect

**Central question:** Could natural language understanding and text modeling be connected by a shared mechanism?

**Shared entities:** community_from=natural language understanding, community_to=text modeling, shared_concepts=evaluation protocol, text

**Evidence claims:**
- `arxiv:1804.07461v1#c01` (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, positive): Using attention improves multi-task BiLSTM performance over a vanilla BiLSTM on sentence-pair tasks.
- `arxiv:1804.07461v1#c02` (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, negative): Existing models do not substantially outperform training a separate model for each task on the main GLUE benchmark.
- `arxiv:1804.07461v1#c03` (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, mixed): Naive multi-task learning with standard models performs no better overall than training separate models per task.
- `arxiv:1804.07461v1#c04` (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, positive): Multi-task learning improves over a single-task model for tasks with less training data.
- `arxiv:2010.12055v1#c01` (arxiv:2010.12055v1 — "A Discrete Variational Recurrent Topic Model without the Reparametrization Trick", 2020, positive): VRTM has higher SwitchP than LDA VB, averaged across three runs.
- `arxiv:2010.12055v1#c02` (arxiv:2010.12055v1 — "A Discrete Variational Recurrent Topic Model without the Reparametrization Trick", 2020, positive): SwitchP had high positive correlation with human judgments.
- `arxiv:2010.12055v1#c03` (arxiv:2010.12055v1 — "A Discrete Variational Recurrent Topic Model without the Reparametrization Trick", 2020, mixed): SwitchP was compared for the proposed algorithm against a variational Bayes implementation of LDA.
- `arxiv:2010.12055v1#c04` (arxiv:2010.12055v1 — "A Discrete Variational Recurrent Topic Model without the Reparametrization Trick", 2020, mixed): Average document-level topic entropy was measured on the test sets across three runs.
- `arxiv:2010.12055v1#c05` (arxiv:2010.12055v1 — "A Discrete Variational Recurrent Topic Model without the Reparametrization Trick", 2020, mixed): Test perplexity was reported for different RNN cells and embedding sizes on APNEWS.

### h066 — Cross-pollination would likely be most productive if NLU contributed low-resource multi-task transfer protocols to text modeling, while text modeling contributed human-correlated intrinsic metrics to NLU.

**Mechanism.** The claims suggest NLU has a nuanced prior that multi-task learning helps especially in limited-data settings, whereas text modeling has stronger machinery for judging text quality beyond raw likelihood; each tool addresses the other's blind spot.

**Predictions:**
- Multi-task training across related text corpora improves topic-model quality most on small datasets
- Human-correlated intrinsic metrics separate NLU models that have similar benchmark averages

**Minimal test.** Run a small replication where a recurrent topic model is trained jointly on a high-resource and low-resource corpus with NLU-style shared-encoder multi-tasking, and separately evaluate sentence-pair models with SwitchP-like human-correlation judgments on output text explanations or token transitions.

**Scope.** method=shared-encoder multi-task learning and human-correlated intrinsic evaluation, task=low-resource text modeling and sentence-pair natural language understanding

**Evidence gap.** What is missing is a direct benchmark where both imported techniques are tested under matched data regimes and compared against each field's default baseline.

**Graph bridge.** evaluation protocol → natural language understanding

**Utility breakdown**

| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.00 | 1.00 | 0.90 | 0.86 | 1.00 | 0.70 | 0.00 | 0.95 |

## Evidence claims

- **arxiv:1802.03654v1#c01** (arxiv:1802.03654v1 — "Beyond the One-Step Greedy Approach in Reinforcement Learning", 2018, positive): The h-greedy policy yields component-wise value improvement, with equality only for an optimal policy.
- **arxiv:1802.03654v1#c02** (arxiv:1802.03654v1 — "Beyond the One-Step Greedy Approach in Reinforcement Learning", 2018, positive): The h-PI algorithm converges within a stated finite iteration bound.
- **arxiv:1802.03654v1#c03** (arxiv:1802.03654v1 — "Beyond the One-Step Greedy Approach in Reinforcement Learning", 2018, positive): The λ-PI algorithm converges within a stated finite iteration bound.
- **arxiv:1802.03796v1#c04** (arxiv:1802.03796v1 — "Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks", 2018, mixed): Evaluation used the CIFAR-100 and STL-10 datasets.
- **arxiv:1802.04335v1#c02** (arxiv:1802.04335v1 — "Neural Program Search: Solving Programming Tasks from Description and Examples", 2018, positive): The final Seq2Tree + Search model achieves the best result of 85.8%.
- **arxiv:1802.04335v1#c03** (arxiv:1802.04335v1 — "Neural Program Search: Solving Programming Tasks from Description and Examples", 2018, positive): Seq2Tree + Search attains 85.8% test accuracy, higher than Attentional Seq2Seq at 54.1% and Attentional Seq2Seq + Search at 72.3%.
- **arxiv:1804.07461v1#c01** (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, positive): Using attention improves multi-task BiLSTM performance over a vanilla BiLSTM on sentence-pair tasks.
- **arxiv:1804.07461v1#c02** (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, negative): Existing models do not substantially outperform training a separate model for each task on the main GLUE benchmark.
- **arxiv:1804.07461v1#c03** (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, mixed): Naive multi-task learning with standard models performs no better overall than training separate models per task.
- **arxiv:1804.07461v1#c04** (arxiv:1804.07461v1 — "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", 2019, positive): Multi-task learning improves over a single-task model for tasks with less training data.
- **arxiv:1804.08838v1#c06** (arxiv:1804.08838v1 — "Measuring the Intrinsic Dimension of Objective Landscapes", 2018, mixed): The threshold used differs by dataset: 90% for MNIST and 45% for CIFAR-10.
- **arxiv:1805.10915v1#c03** (arxiv:1805.10915v1 — "Dirichlet-based Gaussian Processes for Large-scale Calibrated Classification", 2018, mixed): The experiments use an 80%-20% split for training and Platt scaling calibration.
- **arxiv:1805.10915v1#c04** (arxiv:1805.10915v1 — "Dirichlet-based Gaussian Processes for Large-scale Calibrated Classification", 2018, mixed): Calibration is assessed using Expected Calibration Error.
- **arxiv:1809.02840v1#c05** (arxiv:1809.02840v1 — "Neural Guided Constraint Logic Programming for Program Synthesis", 2018, positive): RobustFill Attention-C with a large beam size solved one more problem than RNN+Constraints on flattened tree manipulation problems.
- **arxiv:1901.09749v1#c06** (arxiv:1901.09749v1 — "Fairwashing: the risk of rationalization", 2019, mixed): Test sets are used to evaluate the accuracy and unfairness of the black-box models.
- **arxiv:1902.06349v1#c05** (arxiv:1902.06349v1 — "Learning to Infer Program Sketches", 2019, mixed): With beam size 10 on the full dataset, the Generator only RNN baseline far exceeds previously reported state of the art and achieves near-perfect accuracy, while the synthesis-only model fails to achieve high performance.
- **arxiv:1906.00588v1#c02** (arxiv:1906.00588v1 — "Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel", 2020, negative): Achieving accurate 68% CI coverage is harder than achieving accurate 95% or 90% CI coverage.
- **arxiv:1906.00588v1#c04** (arxiv:1906.00588v1 — "Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel", 2020, negative): Residuals between observed coverage percentages and target coverage levels remain biased.
- **arxiv:1906.00588v1#c05** (arxiv:1906.00588v1 — "Quantifying Point-Prediction Uncertainty in Neural Networks via Residual Estimation with an I/O Kernel", 2020, positive): Z-scores are used to reduce bias in coverage evaluation under a Gaussian outcome assumption.
- **arxiv:1906.05253v1#c01** (arxiv:1906.05253v1 — "Search on the Replay Buffer: Bridging Planning and Reinforcement Learning", 2019, positive): SoRB generalizes to new houses and reaches almost 80% of goals 10 steps away, outperforming goal-conditioned RL.
- **arxiv:1909.01504v1#c01** (arxiv:1909.01504v1 — "Censored Semi-Bandits: A Framework for Resource Allocation with Censored Feedback", 2019, mixed): CSB-DT is empirically compared against CSB-DT-UCB in experiments.
- **arxiv:1909.01504v1#c04** (arxiv:1909.01504v1 — "Censored Semi-Bandits: A Framework for Resource Allocation with Censored Feedback", 2019, negative): Problem instances with δ = 0 are hopeless because optimal allocation requires full-precision threshold estimates.
- **arxiv:2002.10365v1#c01** (arxiv:2002.10365v1 — "The Early Phase of Neural Network Training", 2020, positive): Evaluation accuracy rises rapidly early in training, reaching 55% after the first epoch and exceeding half of the final 91.5% accuracy.
- **arxiv:2002.10648v1#c01** (arxiv:2002.10648v1 — "I Am Going MAD: Maximum Discrepancy Competition for Comparing Classifiers Adaptively", 2020, positive): VGG16BN outperforms ResNet34 and ResNet101 under similar computation budgets.
- **arxiv:2002.10648v1#c03** (arxiv:2002.10648v1 — "I Am Going MAD: Maximum Discrepancy Competition for Comparing Classifiers Adaptively", 2020, negative): PNASNet-5-Large's rank drops substantially under MAD.
- **arxiv:2006.15085v1#c01** (arxiv:2006.15085v1 — "What can I do here? A Theory of Affordances in Reinforcement Learning", 2020, positive): Using affordances significantly reduces planning time compared to using the full model, especially in larger environments.
- **arxiv:2006.15085v1#c02** (arxiv:2006.15085v1 — "What can I do here? A Theory of Affordances in Reinforcement Learning", 2020, mixed): In the small data regime, minimum planning loss is achieved at intermediate values, corresponding to an intermediate affordance set size.
- **arxiv:2006.15820v1#c01** (arxiv:2006.15820v1 — "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search", 2020, positive): Retro* solves 31% more test molecules than DFPN-E under a budget of 500 one-step calls.
- **arxiv:2006.15820v1#c02** (arxiv:2006.15820v1 — "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search", 2020, mixed): Retro* achieves higher success rate and better route quality than several baselines in the performance summary.
- **arxiv:2006.15820v1#c04** (arxiv:2006.15820v1 — "Retro*: Learning Retrosynthetic Planning with Neural Guided A* Search", 2020, positive): Using a value function helps MCTS because it provides a lower-variance value estimate than rollout.
- **arxiv:2007.05655v1#c01** (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): EGP improves navigation metrics over the best prior models on the test set.
- **arxiv:2007.05655v1#c02** (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): Using the EGP module without synthetic data augmentation improves performance over the baseline agent on Val Unseen.
- **arxiv:2007.05655v1#c03** (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, positive): The EGP agent with synthetic data augmentation outperforms prior art on all metrics on validation-unseen and test sets.
- **arxiv:2007.05655v1#c04** (arxiv:2007.05655v1 — "Evolving Graphical Planner: Contextual Global Planning for Vision-and-Language Navigation", 2020, negative): Reducing top-K makes paths shorter but consistently lowers model accuracy.
- **arxiv:2010.12055v1#c01** (arxiv:2010.12055v1 — "A Discrete Variational Recurrent Topic Model without the Reparametrization Trick", 2020, positive): VRTM has higher SwitchP than LDA VB, averaged across three runs.
- **arxiv:2010.12055v1#c02** (arxiv:2010.12055v1 — "A Discrete Variational Recurrent Topic Model without the Reparametrization Trick", 2020, positive): SwitchP had high positive correlation with human judgments.
