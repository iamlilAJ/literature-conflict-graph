# LLM_audit_sample — Q1 judgment spot-check

> 10 LLM judgments from Q1, re-read by cc with a fresh prompt.
> Format: each row shows the input, the gpt-5.4-mini judgment, and cc's
> agree/disagree verdict with one-line reason.

## Methodology

cc (this agent) re-reads each of 10 Q1 judgments. The re-read prompt is:

> "For the given `(method, task)` pair, decide whether this should be in
> a canonical ML method registry (V_M). Choose one: `novel_method`,
> `alias_not_in_Atlas`, `too_generic`, `garbage`. Justify in one sentence."

cc's verdict notes any disagreement with the gpt-5.4-mini judge and the
reason. Read with skepticism — cc is the same family of model as the
judge, so this is a same-family audit, not an independent audit.

## 10 judgments

### 1. Unmatched: method = 'GPT-4 as evaluator', task = 'step-level reasoning evaluation'
  - LLM label: **garbage**
  - LLM rationale: '"GPT-4 as evaluator" refers to using a model as an evaluation tool rather than a distinct method name, so it is not a canonical ML method.'
  - cc verdict: **AGREE.** The string is "<model> as <use case>", a usage pattern, not a method. Should not be V_M.

### 2. Unmatched: method = 'Qwen2.5-VL-72B-Instruct', task = 'visual reasoning'
  - LLM label: **garbage**
  - LLM rationale: 'Qwen2.5-VL-72B-Instruct is a specific model name, not a method name or algorithm.'
  - cc verdict: **AGREE on label, DISAGREE on framing.** Calling a model name "garbage" is harsh — it's a clean instance of "aigraph's claim extractor is putting a *model* into the *method* field". The extractor bug is the finding, not that the string is noise.

### 3. Unmatched: method = 'SpatialRGPT', task = 'spatial reasoning'
  - LLM label: **novel_method**
  - LLM rationale: 'SpatialRGPT appears to be a specific named method tailored to spatial reasoning rather than a generic technique or a base model name.'
  - cc verdict: **AGREE.** Capitalized portmanteau, task-specific, looks like a real method name not in V_M.

### 4. Unmatched: method = 'Citation-Enhanced Generation (CEG)', task = 'hallucination detection'
  - LLM label: **novel_method**
  - LLM rationale: 'Citation-Enhanced Generation (CEG) appears to be a specific named technique for hallucination-related generation, not a generic phrase or model name, and it is plausibly a distinct method rather than a standard canonical alias.'
  - cc verdict: **AGREE.** Title-case noun phrase + parenthesized acronym is the canonical pattern for a paper introducing a method.

### 5. Unmatched: method = 'MedRag with RRF-2 retriever fusion', task = 'medical question answering'
  - LLM label: **alias_not_in_Atlas**
  - LLM rationale: '"MedRag" appears to be a known retrieval-augmented generation method adapted for medical QA, and "RRF-2 retriever fusion" is a configuration detail rather than a distinct canonical method name.'
  - cc verdict: **DISAGREE.** This is a *compound* description with at least 2 method candidates — MedRag and RRF-2 retriever fusion. Should be either `novel_method` (multi-method composite) or split into 2 claims. The judge's label is defensible but masks an aigraph extractor problem.

### 6. Unmatched: method = 'Retrieval Augmented Generation (RAG) agent', task = 'summarization and question answering over EIC resources'
  - LLM label: **alias_not_in_Atlas**
  - LLM rationale: '"Retrieval Augmented Generation (RAG) agent" refers to the known RAG paradigm with an agentic implementation, so it is a non-canonical alias rather than a distinct method name.'
  - cc verdict: **AGREE.** "RAG agent" = RAG + agentic wrapper; the base method IS in V_M as "Retrieval-Augmented Generation". This is a clear alias case.

### 7. Unmatched: method = 'Top-1 accuracy', task = 'visual reasoning evaluation'
  - LLM label: **too_generic**
  - LLM rationale: 'Top-1 accuracy is an evaluation metric, not a specific machine learning method name.'
  - cc verdict: **DISAGREE on label.** Top-1 accuracy is a *metric*, not a "too generic method". The correct label is closer to `garbage` (extractor put a metric into the method field). Reveals the same aigraph-bug pattern as #2 (model in method slot).

### 8. Unmatched: method = 'drawing to reason in space paradigm without training', task = 'spatial reasoning'
  - LLM label: **too_generic**
  - LLM rationale: 'The phrase describes a broad reasoning approach in spatial tasks rather than a specific named ML method.'
  - cc verdict: **AGREE.** Descriptive phrase, not a name. Could be referring to a real method in the paper's text but the extracted form is paraphrased.

### 9. Matched via token_set: method = 'goal-baseline regularization for CLIP reward models'
  - V_M match: 'CLIP' (score=1.0)
  - cc verdict: **DISAGREE with the match.** This is a multi-word phrase whose token-set contains "CLIP". The match is technically true (CLIP IS a token) but the *method* being claimed is "goal-baseline regularization", not CLIP. token_set scorer with threshold 0.85 over-matches when V_M contains short distinctive tokens (CLIP, RAG, MAS).

### 10. Matched via token_set: method = 'Transformer architecture shape'
  - V_M match: 'Transformer' (score=1.0)
  - cc verdict: **MIXED.** The claim *is* about Transformer architecture (so V_M=Transformer is the relevant entry), but the phrase is descriptive ("architecture shape"). Counting this as a clean match inflates Q1's number. A stricter token-overlap rule (require non-stopword head word match) would correctly classify this.

## Aggregate cc audit verdict

- AGREEMENTS: 6/10
- DISAGREEMENTS with the judge's label: 2/10 (#5, #7)
- DISAGREEMENTS with the *match* (judge wasn't called, but token_set was overzealous): 2/10 (#9, #10)

**Implication for Q1 headline number.** The 31% match rate in Q1 is
likely *inflated* — at least 2 of the 50 token_set matches I sampled
(#9, #10) are loose matches that wouldn't survive a stricter
"head-word + at least one informative token" rule. Estimating from
those 2 of 2 sampled: maybe 50% of token_set matches are loose.
That would put the *real* clean match rate at:

```
exact (11) + ratio (1) + token_set × 0.5 (25)  =  37 / 200  =  18.5%
```

So the headline reported in Q1 (31%) is the upper bound; the more
honest number is ~18-20%. Either way, J1 is dead.

## Confounders cc spot-checked

- **Same-family audit.** The judge is gpt-5.4-mini; cc reading is
  gpt-5.4-tier. A genuinely independent audit would use Anthropic
  Sonnet or Gemini. cc's disagreements on #5/#7/#9/#10 are
  defensible without cross-family validation but should not be taken
  as definitive.
- **Sample is not random across labels.** I picked 2 of each label
  class + 2 matched. Real disagreement rates per label require larger
  N per class. With 2 in each, I cannot estimate within-label rates.
- **A 4-way label space conflates two different failure modes.**
  "garbage" lumps together (a) noise from extraction errors and
  (b) model-name-in-method-slot. These are different aigraph bugs.
  Future audit should split these.
