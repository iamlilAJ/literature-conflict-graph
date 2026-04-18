"""Synthetic sample papers around RAG / LLM evaluation for the MVP demo."""

from __future__ import annotations

from .models import Paper


def build_sample_papers() -> list[Paper]:
    papers: list[dict] = [
        {
            "paper_id": "p001",
            "title": "Retrieval-Augmented Generation for Factual Question Answering",
            "year": 2021,
            "venue": "NeurIPS",
            "abstract": (
                "We show that retrieval-augmented generation (RAG) improves factual "
                "question answering on NaturalQuestions by +8.2 EM over a closed-book "
                "LLM baseline when using DPR retrieval with top-k=5."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "performance_improvement",
                    "claim_text": "RAG improves factual QA on NaturalQuestions by +8.2 EM over a closed-book LLM.",
                    "method": "RAG",
                    "model": "GPT-3.5",
                    "task": "factual QA",
                    "dataset": "NaturalQuestions",
                    "metric": "Exact Match",
                    "baseline": "closed-book LLM",
                    "result": "+8.2",
                    "direction": "positive",
                    "setting": {"retriever": "DPR", "top_k": "5", "context_length": "4k", "task_type": "factual"},
                    "evidence_span": "RAG improves NaturalQuestions EM by +8.2 over the closed-book baseline.",
                }
            ],
        },
        {
            "paper_id": "p002",
            "title": "When Retrieval Hurts: Noise in Retrieval-Augmented QA",
            "year": 2022,
            "venue": "ACL",
            "abstract": (
                "RAG improves recall on NaturalQuestions but introduces irrelevant "
                "evidence at high top-k. On HotpotQA, increasing top-k to 20 degrades "
                "multi-hop EM by 3 points relative to a closed-book baseline."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "limitation",
                    "claim_text": "RAG improves factual recall on NaturalQuestions but introduces irrelevant evidence.",
                    "method": "RAG",
                    "model": "GPT-3.5",
                    "task": "factual QA",
                    "dataset": "NaturalQuestions",
                    "metric": "Exact Match",
                    "baseline": "closed-book LLM",
                    "result": "+2.1",
                    "direction": "mixed",
                    "setting": {"retriever": "DPR", "top_k": "10", "context_length": "4k", "task_type": "factual"},
                    "evidence_span": "RAG improves recall but introduces noisy passages.",
                },
                {
                    "claim_type": "limitation",
                    "claim_text": "RAG hurts HotpotQA multi-hop reasoning when top-k is high.",
                    "method": "RAG",
                    "model": "GPT-3.5",
                    "task": "multi-hop QA",
                    "dataset": "HotpotQA",
                    "metric": "Exact Match",
                    "baseline": "closed-book LLM",
                    "result": "-3.0",
                    "direction": "negative",
                    "setting": {"retriever": "DPR", "top_k": "20", "context_length": "4k", "task_type": "multi-hop"},
                    "evidence_span": "At top-k=20 RAG loses 3 EM on HotpotQA.",
                },
            ],
        },
        {
            "paper_id": "p003",
            "title": "Entailment-Filtered Retrieval for Multi-hop Reasoning",
            "year": 2023,
            "venue": "EMNLP",
            "abstract": (
                "Filtering retrieved passages with an entailment model improves "
                "HotpotQA multi-hop reasoning by +5.4 EM over standard DPR retrieval."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "performance_improvement",
                    "claim_text": "Entailment-filtered retrieval improves multi-hop QA on HotpotQA.",
                    "method": "entailment-filtered retrieval",
                    "model": "GPT-4",
                    "task": "multi-hop QA",
                    "dataset": "HotpotQA",
                    "metric": "Exact Match",
                    "baseline": "RAG",
                    "result": "+5.4",
                    "direction": "positive",
                    "setting": {"retriever": "DPR+NLI", "top_k": "5", "context_length": "8k", "task_type": "multi-hop"},
                    "evidence_span": "Entailment filtering yields +5.4 EM on HotpotQA.",
                }
            ],
        },
        {
            "paper_id": "p004",
            "title": "Long-Context LLMs Dissolve the RAG Advantage",
            "year": 2024,
            "venue": "ICLR",
            "abstract": (
                "On NaturalQuestions, long-context models with 128k windows shrink the "
                "marginal benefit of RAG to under 1 EM, compared to +8 EM at 4k."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "setting_effect",
                    "claim_text": "Long-context models reduce the marginal benefit of RAG on factual QA.",
                    "method": "RAG",
                    "model": "GPT-4-128k",
                    "task": "factual QA",
                    "dataset": "NaturalQuestions",
                    "metric": "Exact Match",
                    "baseline": "long-context LLM",
                    "result": "+0.6",
                    "direction": "negative",
                    "setting": {"retriever": "DPR", "top_k": "5", "context_length": "128k", "task_type": "long-context"},
                    "evidence_span": "At 128k context the RAG gain drops to +0.6 EM.",
                }
            ],
        },
        {
            "paper_id": "p005",
            "title": "Generator-Aware Reranking Beats Larger Top-K",
            "year": 2024,
            "venue": "NAACL",
            "abstract": (
                "A generator-aware reranker improves HotpotQA multi-hop EM by +6.1 "
                "over doubling top-k, at fixed retriever cost."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "mechanism",
                    "claim_text": "Generator-aware reranking improves multi-hop QA more than increasing top-k.",
                    "method": "generator-aware reranking",
                    "model": "GPT-4",
                    "task": "multi-hop QA",
                    "dataset": "HotpotQA",
                    "metric": "Exact Match",
                    "baseline": "RAG top-k=20",
                    "result": "+6.1",
                    "direction": "positive",
                    "setting": {"retriever": "DPR", "top_k": "5", "context_length": "8k", "task_type": "multi-hop"},
                    "evidence_span": "Reranking yields +6.1 EM over top-k=20 RAG.",
                }
            ],
        },
        {
            "paper_id": "p006",
            "title": "Are We Measuring the Right Thing? A Critique of QA Metrics",
            "year": 2023,
            "venue": "TACL",
            "abstract": (
                "Exact Match and F1 over-reward short answer overlap and ignore "
                "whether retrieved evidence actually supports the predicted answer."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "limitation",
                    "claim_text": "EM and F1 over-reward short-answer overlap and ignore evidence-chain validity.",
                    "method": "EM/F1 metric",
                    "model": "metric-agnostic",
                    "task": "QA evaluation",
                    "dataset": "NaturalQuestions",
                    "metric": "Exact Match",
                    "baseline": "evidence-chain scoring",
                    "result": "n/a",
                    "direction": "mixed",
                    "setting": {"retriever": "n/a", "top_k": "n/a", "context_length": "n/a", "task_type": "factual"},
                    "evidence_span": "EM inflates gains that are not supported by evidence.",
                }
            ],
        },
        {
            "paper_id": "p007",
            "title": "Calibrating RAG Against Stronger Baselines",
            "year": 2024,
            "venue": "arXiv",
            "abstract": (
                "When compared against a strong long-context LLM baseline rather than "
                "a vanilla closed-book model, reported RAG gains on NaturalQuestions "
                "drop from +8 EM to +1.2 EM."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "comparison",
                    "claim_text": "Stronger baselines shrink reported RAG gains on factual QA.",
                    "method": "RAG",
                    "model": "GPT-4",
                    "task": "factual QA",
                    "dataset": "NaturalQuestions",
                    "metric": "Exact Match",
                    "baseline": "strong long-context LLM",
                    "result": "+1.2",
                    "direction": "negative",
                    "setting": {"retriever": "DPR", "top_k": "5", "context_length": "32k", "task_type": "factual"},
                    "evidence_span": "Against strong baselines the RAG gain falls to +1.2 EM.",
                }
            ],
        },
        {
            "paper_id": "p008",
            "title": "Tight RAG: Low Top-K Boosts HotpotQA",
            "year": 2023,
            "venue": "EMNLP",
            "abstract": (
                "With top-k=3 and a DPR retriever, RAG recovers a +4.0 EM gain on "
                "HotpotQA multi-hop reasoning."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "setting_effect",
                    "claim_text": "RAG with low top-k improves HotpotQA multi-hop QA.",
                    "method": "RAG",
                    "model": "GPT-4",
                    "task": "multi-hop QA",
                    "dataset": "HotpotQA",
                    "metric": "Exact Match",
                    "baseline": "closed-book LLM",
                    "result": "+4.0",
                    "direction": "positive",
                    "setting": {"retriever": "DPR", "top_k": "3", "context_length": "8k", "task_type": "multi-hop"},
                    "evidence_span": "At top-k=3 RAG gains +4.0 EM on HotpotQA.",
                }
            ],
        },
        {
            "paper_id": "p009",
            "title": "DPR Reranking on NaturalQuestions",
            "year": 2022,
            "venue": "ACL",
            "abstract": (
                "Reranking DPR candidates with a cross-encoder gives +2.3 EM on "
                "NaturalQuestions factual QA."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "performance_improvement",
                    "claim_text": "Cross-encoder reranking improves RAG factual QA on NaturalQuestions.",
                    "method": "RAG",
                    "model": "GPT-3.5",
                    "task": "factual QA",
                    "dataset": "NaturalQuestions",
                    "metric": "Exact Match",
                    "baseline": "closed-book LLM",
                    "result": "+2.3",
                    "direction": "positive",
                    "setting": {"retriever": "DPR+CE", "top_k": "5", "context_length": "4k", "task_type": "factual"},
                    "evidence_span": "Cross-encoder reranking adds +2.3 EM on NQ.",
                }
            ],
        },
        {
            "paper_id": "p010",
            "title": "BM25 vs DPR for Factual QA",
            "year": 2022,
            "venue": "SIGIR",
            "abstract": (
                "BM25 retrieval produces competitive factual QA on NaturalQuestions "
                "compared to DPR, within 1 EM, suggesting retriever choice matters "
                "less than generator capacity."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "comparison",
                    "claim_text": "BM25 retrieval yields comparable factual QA performance to DPR.",
                    "method": "RAG",
                    "model": "GPT-3.5",
                    "task": "factual QA",
                    "dataset": "NaturalQuestions",
                    "metric": "Exact Match",
                    "baseline": "RAG with DPR",
                    "result": "-0.8",
                    "direction": "mixed",
                    "setting": {"retriever": "BM25", "top_k": "5", "context_length": "4k", "task_type": "factual"},
                    "evidence_span": "BM25 is within 1 EM of DPR on NQ.",
                }
            ],
        },
        {
            "paper_id": "p011",
            "title": "Chain-of-Thought without Retrieval",
            "year": 2023,
            "venue": "ICLR",
            "abstract": (
                "Strong chain-of-thought prompting lifts closed-book LLMs on HotpotQA "
                "by +3.5 EM, closing part of the gap to RAG."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "performance_improvement",
                    "claim_text": "Chain-of-thought prompting improves closed-book multi-hop QA.",
                    "method": "chain-of-thought",
                    "model": "GPT-4",
                    "task": "multi-hop QA",
                    "dataset": "HotpotQA",
                    "metric": "Exact Match",
                    "baseline": "closed-book LLM",
                    "result": "+3.5",
                    "direction": "positive",
                    "setting": {"retriever": "none", "top_k": "0", "context_length": "8k", "task_type": "multi-hop"},
                    "evidence_span": "CoT adds +3.5 EM on HotpotQA without retrieval.",
                }
            ],
        },
        {
            "paper_id": "p012",
            "title": "Hybrid Retrieval Mixtures for Agentic QA",
            "year": 2024,
            "venue": "arXiv",
            "abstract": (
                "A mixture-of-retrievers approach gives mixed results on agentic "
                "long-context QA: +2.0 EM on some tasks, -1.5 EM on others."
            ),
            "text": "",
            "structured_hint": [
                {
                    "claim_type": "setting_effect",
                    "claim_text": "Mixture-of-retrievers gives mixed results on agentic long-context QA.",
                    "method": "mixture-of-retrievers",
                    "model": "GPT-4-128k",
                    "task": "long-context QA",
                    "dataset": "LongBench",
                    "metric": "Exact Match",
                    "baseline": "RAG",
                    "result": "mixed",
                    "direction": "mixed",
                    "setting": {"retriever": "BM25+DPR", "top_k": "8", "context_length": "128k", "task_type": "agentic"},
                    "evidence_span": "Hybrid retrieval is uneven on agentic long-context QA.",
                }
            ],
        },
    ]
    return [Paper.model_validate(p) for p in papers]
