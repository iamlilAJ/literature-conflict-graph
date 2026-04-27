from __future__ import annotations

import gzip
import html as html_lib
import io
import json
import math
import os
import re
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .claim_schema import sentence_spans
from .fetch_openalex import fetch_openalex_papers
from .io import read_jsonl, write_json, write_jsonl
from .models import Paper, PaperArtifactStatus, PaperSection, PaperSentence
from .paper_select import score_paper


DEFAULT_CORPUS_ROOT = Path("data/corpus/arxiv_reasoning")
PARSER_VERSION = "corpus-v1"
TEXT_FILE = "text.json"
SECTIONS_FILE = "sections.json"
SENTENCES_FILE = "sentences.json"
METADATA_FILE = "metadata.json"

_SOURCE_TIMEOUT = 45.0
_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}")
_ABSTRACT_RE = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)
_APPENDIX_RE = re.compile(r"\\appendix\b")
_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}")
_END_DOC_RE = re.compile(r"\\end\{document\}")
_COMMAND_WITH_ARG_RE = re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}")
_COMMAND_RE = re.compile(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?")
_MULTISPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKLINE_RE = re.compile(r"\n{3,}")
_PDF_LITERAL_RE = re.compile(r"\(([^()]*(?:\\.[^()]*)*)\)\s*Tj")
_PDF_ARRAY_RE = re.compile(r"\[(.*?)\]\s*TJ", re.S)
_PDF_ARRAY_LITERAL_RE = re.compile(r"\(([^()]*(?:\\.[^()]*)*)\)")
_PDF_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*)\s+([A-Z][A-Za-z0-9 ,:/_-]{2,})$")
_APPENDIX_HEADING_RE = re.compile(r"^Appendix(?:\s+[A-Z0-9]+)?[:\s-]*(.*)$", re.IGNORECASE)
_SECTION_TITLE_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_PDF_PARSE_MAX_CHARS = 1_500_000
_REASONING_RULES: tuple[tuple[str, str], ...] = (
    ("reasoning-language-models", 'all:reasoning AND all:"language model"'),
    ("llm-reasoning", 'all:reasoning AND (all:"large language model" OR all:llm)'),
    ("chain-of-thought", 'all:"large language model" AND all:"chain of thought"'),
    ("tree-of-thought", 'all:"large language model" AND all:"tree of thought"'),
    ("reasoning-steps", 'all:"large language model" AND all:"reasoning step"'),
    ("multi-step-reasoning", 'all:"large language model" AND all:"multi-step reasoning"'),
    ("logical-reasoning", 'all:"large language model" AND all:"logical reasoning"'),
    ("mathematical-reasoning", 'all:"large language model" AND all:"mathematical reasoning"'),
    ("commonsense-reasoning", 'all:"large language model" AND all:"commonsense reasoning"'),
    ("reasoning-trace", 'all:"large language model" AND all:"reasoning trace"'),
    ("reasoning-models", 'all:"reasoning model"'),
    ("self-consistency", 'all:"large language model" AND all:"self-consistency"'),
    ("verifier", 'all:"large language model" AND all:verifier'),
    ("process-supervision", 'all:"large language model" AND all:"process supervision"'),
    ("outcome-supervision", 'all:"large language model" AND all:"outcome supervision"'),
    ("search", 'all:"large language model" AND all:search AND all:reasoning'),
    ("planning", 'all:"large language model" AND all:planning AND all:reasoning'),
    ("test-time-scaling", 'all:"large language model" AND all:"test-time scaling"'),
    ("inference-time-scaling", 'all:"large language model" AND all:"inference-time scaling"'),
    ("test-time-compute", 'all:"large language model" AND all:"test-time compute"'),
    ("reasoning-via-tool-use", 'all:"large language model" AND all:reasoning AND all:"tool use"'),
    ("program-aided-reasoning", 'all:"large language model" AND all:reasoning AND all:program'),
    ("agent-llm", 'all:"large language model" AND all:agent'),
    ("agent-planning", 'all:"language model agent" AND all:planning'),
    ("autonomous-agent", 'all:"autonomous agent" AND all:"language model"'),
    ("agent-reasoning", 'all:agent AND all:reasoning AND all:"language model"'),
    ("multi-agent-llm", 'all:"multi-agent" AND all:"large language model"'),
    ("rlhf", 'all:"large language model" AND all:"reinforcement learning from human feedback"'),
    ("rl-language-models", 'all:"reinforcement learning" AND all:"large language model"'),
    ("ppo-llm", 'all:"large language model" AND all:"proximal policy optimization"'),
    ("dpo", 'all:"direct preference optimization"'),
    ("grpo", 'all:"group relative policy optimization"'),
    ("rl-reasoning", 'all:"reinforcement learning" AND all:reasoning AND all:"language model"'),
    ("tool-use", 'all:"large language model" AND all:"tool use"'),
    ("function-calling", 'all:"large language model" AND all:"function calling"'),
    ("tool-learning", 'all:"tool learning" AND all:"language model"'),
    ("multimodal-reasoning", 'all:multimodal AND all:reasoning AND all:"language model"'),
    ("vision-language-reasoning", 'all:"vision language" AND all:reasoning'),
    ("visual-reasoning-llm", 'all:"visual reasoning" AND all:"language model"'),
    ("reasoning-benchmark", 'all:"large language model" AND all:reasoning AND all:benchmark'),
    ("llm-evaluation", 'all:"large language model" AND all:evaluation AND all:reasoning'),
    ("math-benchmark", 'all:"large language model" AND all:MATH AND all:benchmark'),
    ("retrieval-augmented", 'all:"retrieval-augmented generation"'),
    ("rag-llm", 'all:RAG AND all:"large language model"'),
    ("knowledge-retrieval", 'all:"large language model" AND all:"knowledge retrieval"'),
    ("retrieval-reasoning", 'all:retrieval AND all:reasoning AND all:"language model"'),
    ("self-correction", 'all:"large language model" AND all:"self-correction"'),
    ("self-refine", 'all:"large language model" AND all:"self-refine"'),
    ("code-reasoning", 'all:"large language model" AND all:code AND all:reasoning'),
)
_SECTION_CANONICAL_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("abstract", ("abstract",)),
    (
        "introduction",
        (
            "introduction",
            "overview",
            "motivation",
            "background",
            "background and motivation",
            "problem setup",
        ),
    ),
    (
        "method",
        (
            "method",
            "methods",
            "methodology",
            "approach",
            "approaches",
            "framework",
            "framework overview",
            "model",
            "models",
            "proposed method",
            "proposed approach",
        ),
    ),
    (
        "results",
        (
            "results",
            "result",
            "main results",
            "evaluation",
            "evaluation results",
            "experimental results",
            "results and analysis",
            "analysis and results",
            "experiments",
            "experiment",
            "empirical evaluation",
            "benchmark results",
            "experimental setup",
            "ablation",
            "ablations",
        ),
    ),
    (
        "discussion",
        (
            "discussion",
            "analysis",
            "further analysis",
            "error analysis",
            "results analysis",
            "discussion and analysis",
            "analysis and discussion",
        ),
    ),
    (
        "limitations",
        (
            "limitations",
            "limitation",
            "limitations and future work",
            "limitations and broader impact",
            "risks and limitations",
            "failure analysis",
            "failure analyses",
        ),
    ),
    (
        "conclusion",
        (
            "conclusion",
            "conclusions",
            "concluding remarks",
            "conclusion and future work",
            "future work",
        ),
    ),
)
_ROLE_PRIORITY_WEIGHTS = {
    "survey": 1.0,
    "benchmark": 0.95,
    "dataset": 0.85,
    "failure": 0.8,
    "industry": 0.7,
    "method": 0.6,
    "other": 0.4,
}


@dataclass
class ParsedArtifact:
    text: str
    sections: list[PaperSection]
    source: str
    warnings: list[str]


def configured_corpus_root(root: str | Path | None = None) -> Path:
    raw = root or os.environ.get("AIGRAPH_CORPUS_ROOT") or DEFAULT_CORPUS_ROOT
    return Path(raw)


def artifact_dir(root: str | Path, paper_id: str) -> Path:
    safe_id = paper_id.replace(":", "__").replace("/", "_")
    return Path(root) / "artifacts" / safe_id


def seed_reasoning_corpus(
    root: str | Path,
    *,
    from_year: int = 2022,
    to_year: int = 2026,
    per_query_limit: int = 400,
    fetcher: Any | None = None,
) -> list[Paper]:
    root_path = configured_corpus_root(root)
    root_path.mkdir(parents=True, exist_ok=True)
    manifest = root_path / "papers.jsonl"
    existing: dict[str, Paper] = {}
    if manifest.exists():
        for paper in read_jsonl(manifest, Paper):
            if not _paper_has_valid_arxiv_id(paper):
                continue
            existing[_manifest_key(paper)] = paper

    source = os.environ.get("AIGRAPH_SEED_SOURCE", "openalex").strip().lower()
    if fetcher is not None:
        fetch = fetcher
    elif source == "arxiv":
        from .fetch_arxiv import fetch_arxiv_papers as _arxiv_fetch

        def fetch(query, from_year, to_year, limit, strategy=None, **_kwargs):  # type: ignore[no-redef]
            return _arxiv_fetch(
                query=query,
                from_year=from_year,
                to_year=to_year,
                limit=limit,
                strategy="balanced",
            )
    else:
        fetch = fetch_openalex_papers
    now = _now_iso()
    tag_filter_raw = os.environ.get("AIGRAPH_CORPUS_REASONING_TAGS", "").strip()
    allowed_tags = {t.strip() for t in tag_filter_raw.split(",") if t.strip()} if tag_filter_raw else None
    rules = [(t, q) for t, q in _REASONING_RULES if allowed_tags is None or t in allowed_tags]
    for tag, query in rules:
        papers = fetch(
            query=query,
            from_year=from_year,
            to_year=to_year,
            limit=per_query_limit,
            strategy="high-impact",
            candidate_multiplier=6,
            min_relevance=0.12,
            require_core_match=False,
        )
        for paper in papers:
            manifest_paper = _manifest_paper_from_candidate(
                paper,
                seed_tag=tag,
                query=query,
                now=now,
                existing=existing.get(_manifest_key(paper)),
            )
            if manifest_paper is None:
                continue
            existing[_manifest_key(manifest_paper)] = manifest_paper

    ordered = _order_manifest(existing.values())
    write_jsonl(manifest, ordered)
    return ordered


def sync_arxiv_corpus(
    root: str | Path,
    *,
    refresh: bool = False,
    limit: int | None = None,
    client: Any | None = None,
) -> list[PaperArtifactStatus]:
    root_path = configured_corpus_root(root)
    papers_path = root_path / "papers.jsonl"
    if not papers_path.exists():
        return []
    papers = read_jsonl(papers_path, Paper)
    selected = _select_sync_batch(papers, batch_size=limit, refresh=refresh)

    owns_client = False
    if client is None:
        import httpx

        client = httpx.Client(timeout=_SOURCE_TIMEOUT, follow_redirects=True)
        owns_client = True

    statuses: list[PaperArtifactStatus] = []
    try:
        for paper in selected:
            statuses.append(_sync_one_paper(root_path, paper, client=client, refresh=refresh))
    finally:
        if owns_client:
            client.close()
    _update_manifest_after_sync(root_path, papers, selected, statuses, refresh=refresh)
    return statuses


def validate_corpus(root: str | Path) -> dict[str, Any]:
    root_path = configured_corpus_root(root)
    paper_dirs = list((root_path / "artifacts").glob("*"))
    summaries: list[PaperArtifactStatus] = []
    for paper_dir in paper_dirs:
        metadata_path = paper_dir / METADATA_FILE
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            status_payload = payload.get("artifact_status") or {}
            summaries.append(PaperArtifactStatus.model_validate(status_payload))
            continue
        text_payload = _read_optional_json(paper_dir / TEXT_FILE) or {}
        sections_payload = _read_optional_json(paper_dir / SECTIONS_FILE) or []
        sentences_payload = _read_optional_json(paper_dir / SENTENCES_FILE) or []
        paper_id = str(text_payload.get("paper_id") or paper_dir.name.replace("__", ":", 1))
        summaries.append(
            PaperArtifactStatus(
                paper_id=paper_id,
                canonical_source=text_payload.get("canonical_source"),
                parse_status="complete" if text_payload.get("text") else "missing",
                text_length=len(str(text_payload.get("text") or "")),
                section_count=len(sections_payload) if isinstance(sections_payload, list) else 0,
                sentence_count=len(sentences_payload) if isinstance(sentences_payload, list) else 0,
            )
        )

    total = len(summaries)
    with_source = sum(1 for item in summaries if item.source_fetched)
    with_sections = sum(1 for item in summaries if item.section_count > 0)
    pdf_fallback = sum(1 for item in summaries if item.canonical_source == "pdf")
    weak_sections = [item.paper_id for item in summaries if item.section_count <= 1]
    summary = {
        "total_papers": total,
        "pct_with_source": round(with_source / total, 4) if total else 0.0,
        "pct_with_sections": round(with_sections / total, 4) if total else 0.0,
        "pct_pdf_fallback": round(pdf_fallback / total, 4) if total else 0.0,
        "average_sections_per_paper": round(sum(item.section_count for item in summaries) / total, 2) if total else 0.0,
        "average_sentences_per_paper": round(sum(item.sentence_count for item in summaries) / total, 2) if total else 0.0,
        "weak_section_papers": weak_sections[:50],
    }
    write_json(root_path / "summary.json", summary)
    return summary


def enrich_citations_from_semantic_scholar(
    root: str | Path,
    *,
    batch_size: int = 500,
    timeout: float = 30.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Hydrate manifest papers with Semantic Scholar citation counts and rerank."""
    import time as _time
    import httpx

    root_path = configured_corpus_root(root)
    manifest = root_path / "papers.jsonl"
    if not manifest.exists():
        return {"updated": 0, "missing": 0, "total": 0}

    papers = list(read_jsonl(manifest, Paper))
    id_to_indices: dict[str, list[int]] = {}
    for idx, paper in enumerate(papers):
        base = (
            paper.arxiv_id_base
            or _arxiv_base_from_full(paper.arxiv_id_full)
            or _arxiv_base_from_paper_id(paper.paper_id)
        )
        if base:
            id_to_indices.setdefault(f"arXiv:{base}", []).append(idx)

    unique_ids = list(id_to_indices.keys())
    headers = {"x-api-key": api_key} if api_key else {}
    updated = 0
    missing = 0

    with httpx.Client(timeout=timeout, headers=headers) as client:
        for start in range(0, len(unique_ids), batch_size):
            chunk = unique_ids[start : start + batch_size]
            attempt = 0
            while True:
                response = client.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": "citationCount,influentialCitationCount,referenceCount"},
                    json={"ids": chunk},
                )
                if response.status_code == 429 and attempt < 3:
                    _time.sleep(15 * (attempt + 1))
                    attempt += 1
                    continue
                response.raise_for_status()
                break
            payload = response.json()
            for s2_id, entry in zip(chunk, payload):
                if not entry:
                    missing += len(id_to_indices.get(s2_id, []))
                    continue
                cc = int(entry.get("citationCount") or 0)
                for idx in id_to_indices[s2_id]:
                    papers[idx] = papers[idx].model_copy(update={"cited_by_count": cc})
                    updated += 1
            _time.sleep(1.0)

    for idx, paper in enumerate(papers):
        impact = _academic_impact_score(paper.cited_by_count)
        role_weight = _role_priority_weight(paper.paper_role)
        priority = _priority_score(
            academic_impact=impact,
            recency_score=float(paper.recency_score or 0.0),
            reasoning_relevance=float(paper.reasoning_relevance or 0.0),
            role_weight=role_weight,
        )
        papers[idx] = paper.model_copy(
            update={
                "academic_impact": round(impact, 4),
                "role_weight": round(role_weight, 4),
                "priority_score": round(priority, 4),
            }
        )

    ordered = _order_manifest(papers)
    write_jsonl(manifest, ordered)
    return {"updated": updated, "missing": missing, "total": len(papers), "unique_arxiv_ids": len(unique_ids)}


def export_corpus_paper(root: str | Path, paper_id: str) -> dict[str, Any]:
    root_path = configured_corpus_root(root)
    paper_root = artifact_dir(root_path, paper_id)
    metadata = _read_optional_json(paper_root / METADATA_FILE) or {}
    text = _read_optional_json(paper_root / TEXT_FILE) or {}
    sections = _read_optional_json(paper_root / SECTIONS_FILE) or []
    sentences = _read_optional_json(paper_root / SENTENCES_FILE) or []
    return {
        "metadata": metadata,
        "text": text,
        "sections": sections,
        "sentences": sentences,
    }


def hydrate_paper_from_corpus(paper: Paper, root: str | Path | None = None) -> Paper:
    corpus_root = configured_corpus_root(root)
    text_payload = _read_optional_json(artifact_dir(corpus_root, paper.paper_id) / TEXT_FILE)
    text = str((text_payload or {}).get("text") or "").strip()
    if not text:
        return paper
    return paper.model_copy(update={"text": text})


def hydrate_papers_from_corpus(papers: list[Paper], root: str | Path | None = None) -> list[Paper]:
    return [hydrate_paper_from_corpus(paper, root=root) for paper in papers]


def load_corpus_sections(paper: Paper, root: str | Path | None = None) -> list[PaperSection]:
    corpus_root = configured_corpus_root(root)
    payload = _read_optional_json(artifact_dir(corpus_root, paper.paper_id) / SECTIONS_FILE)
    if not isinstance(payload, list):
        return []
    return [PaperSection.model_validate(item) for item in payload]


def load_corpus_sentences(paper: Paper, root: str | Path | None = None) -> list[PaperSentence]:
    corpus_root = configured_corpus_root(root)
    payload = _read_optional_json(artifact_dir(corpus_root, paper.paper_id) / SENTENCES_FILE)
    if not isinstance(payload, list):
        return []
    return [PaperSentence.model_validate(item) for item in payload]


def _sync_one_paper(root: Path, paper: Paper, *, client: Any, refresh: bool) -> PaperArtifactStatus:
    paper_root = artifact_dir(root, paper.paper_id)
    paper_root.mkdir(parents=True, exist_ok=True)
    metadata_path = paper_root / METADATA_FILE
    existing_status = _load_existing_status(metadata_path)
    if existing_status and existing_status.parse_status == "complete" and not refresh:
        return existing_status

    status = PaperArtifactStatus(
        paper_id=paper.paper_id,
        source_url=_source_url_for_paper(paper),
        html_url=_html_url_for_paper(paper),
        pdf_url=_pdf_url_for_paper(paper),
    )

    parsed: ParsedArtifact | None = None
    source_blob = _download_optional(client, status.source_url)
    if source_blob is not None:
        status.source_fetched = True
        _write_bytes(paper_root / "source" / "artifact.bin", source_blob)
        parsed = _parse_source_blob(source_blob)
        if parsed.text:
            status.canonical_source = "tex"

    if (parsed is None or not parsed.text) and status.html_url:
        html_blob = _download_optional(client, status.html_url)
        if html_blob is not None:
            status.html_fetched = True
            (paper_root / "html").mkdir(parents=True, exist_ok=True)
            (paper_root / "html" / "paper.html").write_text(html_blob.decode("utf-8", errors="ignore"), encoding="utf-8")
            parsed = _parse_html_blob(html_blob)
            if parsed.text:
                status.canonical_source = "html"

    if (parsed is None or not parsed.text) and status.pdf_url:
        pdf_blob = _download_optional(client, status.pdf_url)
        if pdf_blob is not None:
            status.pdf_fetched = True
            _write_bytes(paper_root / "pdf" / "paper.pdf", pdf_blob)
            parsed = _parse_pdf_blob(pdf_blob)
            if parsed.text:
                status.canonical_source = "pdf"

    if parsed is None:
        parsed = ParsedArtifact(text="", sections=[], source="pdf", warnings=["No source, HTML, or PDF artifact could be parsed."])

    sentences = _build_sentences(parsed.text, parsed.sections)
    status.text_length = len(parsed.text)
    status.section_count = len(parsed.sections)
    status.sentence_count = len(sentences)
    status.warnings.extend(parsed.warnings)
    status.parse_status = _derive_parse_status(parsed.text, parsed.sections)
    if status.canonical_source is None and parsed.text:
        status.canonical_source = parsed.source  # type: ignore[assignment]
    if not parsed.text:
        status.errors.append("No canonical text extracted.")

    write_json(
        paper_root / TEXT_FILE,
        {
            "paper_id": paper.paper_id,
            "text": parsed.text,
            "canonical_source": status.canonical_source,
            "parser_version": PARSER_VERSION,
        },
    )
    write_json(paper_root / SECTIONS_FILE, [item.model_dump() for item in parsed.sections])
    write_json(paper_root / SENTENCES_FILE, [item.model_dump() for item in sentences])
    write_json(
        metadata_path,
        {
            "paper": paper.model_dump(),
            "artifact_status": status.model_dump(),
        },
    )
    return status


def _load_existing_status(metadata_path: Path) -> PaperArtifactStatus | None:
    payload = _read_optional_json(metadata_path)
    if not isinstance(payload, dict):
        return None
    status_payload = payload.get("artifact_status")
    if not isinstance(status_payload, dict):
        return None
    try:
        return PaperArtifactStatus.model_validate(status_payload)
    except Exception:
        return None


def _merge_seed_reason(existing: Paper | None, tag: str) -> str:
    reasons = [item for item in [existing.seed_reason if existing else None, tag] if item]
    return "; ".join(sorted(dict.fromkeys(reasons)))


def _manifest_key(paper: Paper) -> str:
    return paper.arxiv_id_base or _arxiv_base_from_full(paper.arxiv_id_full) or _arxiv_base_from_paper_id(paper.paper_id) or paper.paper_id


def _paper_has_valid_arxiv_id(paper: Paper) -> bool:
    return bool(_valid_arxiv_identifier(paper.arxiv_id_full) or _valid_arxiv_identifier(_arxiv_full_from_paper_id(paper.paper_id)))


def _manifest_paper_from_candidate(
    paper: Paper,
    *,
    seed_tag: str,
    query: str,
    now: str,
    existing: Paper | None,
) -> Paper | None:
    arxiv_full = paper.arxiv_id_full or _arxiv_full_from_paper_id(paper.paper_id)
    arxiv_base = paper.arxiv_id_base or _arxiv_base_from_full(arxiv_full) or _arxiv_base_from_paper_id(paper.paper_id)
    if not arxiv_full or not arxiv_base:
        return None

    candidate = paper.model_copy(
        update={
            "paper_id": f"arxiv:{arxiv_full}",
            "url": f"https://arxiv.org/abs/{arxiv_full}",
            "arxiv_id_full": arxiv_full,
            "arxiv_id_base": arxiv_base,
            "corpus_tag": "reasoning-core",
            "seed_reason": _merge_seed_reason(existing, seed_tag),
            "first_seen_at": existing.first_seen_at if existing else now,
            "last_seen_at": now,
        }
    )
    metrics = score_paper(candidate, query=query, strategy="high-impact")
    academic_impact = _academic_impact_score(candidate.cited_by_count)
    role_weight = _role_priority_weight(candidate.paper_role)
    priority_score = _priority_score(
        academic_impact=academic_impact,
        recency_score=float(metrics["recency_score"]),
        reasoning_relevance=float(metrics["title_relevance_score"]),
        role_weight=role_weight,
    )
    candidate = candidate.model_copy(
        update={
            "academic_impact": round(academic_impact, 4),
            "recency_score": round(float(metrics["recency_score"]), 4),
            "reasoning_relevance": round(float(metrics["title_relevance_score"]), 4),
            "role_weight": round(role_weight, 4),
            "priority_score": round(priority_score, 4),
            "selection_score": float(metrics["selection_score"]),
            "selection_reason": str(metrics["selection_reason"]),
        }
    )
    if existing is None:
        return candidate.model_copy(update={"sync_status": "queued"})

    if _prefer_existing_manifest(existing, candidate):
        return existing.model_copy(
            update={
                "seed_reason": _merge_seed_reason(existing, seed_tag),
                "last_seen_at": now,
            }
        )

    replacement = {
        "sync_status": existing.sync_status if existing.arxiv_id_full == arxiv_full else "queued",
        "sync_attempt_count": existing.sync_attempt_count if existing.arxiv_id_full == arxiv_full else 0,
        "last_attempted_at": existing.last_attempted_at if existing.arxiv_id_full == arxiv_full else None,
        "completed_at": existing.completed_at if existing.arxiv_id_full == arxiv_full else None,
    }
    return candidate.model_copy(update=replacement)


def _prefer_existing_manifest(existing: Paper, incoming: Paper) -> bool:
    existing_version = _arxiv_version(existing.arxiv_id_full)
    incoming_version = _arxiv_version(incoming.arxiv_id_full)
    if existing_version > incoming_version:
        return True
    if incoming_version > existing_version:
        return False
    return float(existing.priority_score or 0.0) >= float(incoming.priority_score or 0.0)


def _order_manifest(papers: Any) -> list[Paper]:
    return sorted(
        list(papers),
        key=lambda item: (
            0 if item.sync_status != "complete" else 1,
            -float(item.priority_score or 0.0),
            -int(item.year or 0),
            item.paper_id,
        ),
    )


def _select_sync_batch(papers: list[Paper], *, batch_size: int | None, refresh: bool) -> list[Paper]:
    candidates = [paper for paper in papers if refresh or paper.sync_status != "complete"]
    ordered = sorted(candidates, key=lambda paper: (-_effective_sync_priority(paper), -int(paper.year or 0), paper.paper_id))
    if batch_size is None:
        return ordered
    return ordered[: max(0, int(batch_size))]


def _update_manifest_after_sync(
    root: Path,
    manifest_papers: list[Paper],
    selected: list[Paper],
    statuses: list[PaperArtifactStatus],
    *,
    refresh: bool,
) -> None:
    if not selected:
        return
    status_by_id = {status.paper_id: status for status in statuses}
    selected_ids = {paper.paper_id for paper in selected}
    now = _now_iso()
    updated: list[Paper] = []
    for paper in manifest_papers:
        if paper.paper_id not in selected_ids:
            updated.append(paper)
            continue
        status = status_by_id.get(paper.paper_id)
        if status is None:
            updated.append(paper)
            continue
        sync_status = "complete" if status.parse_status == "complete" else "failed"
        updated.append(
            paper.model_copy(
                update={
                    "sync_status": sync_status,
                    "sync_attempt_count": int(paper.sync_attempt_count or 0) + 1,
                    "last_attempted_at": now,
                    "completed_at": now if sync_status == "complete" else paper.completed_at,
                }
            )
        )
    write_jsonl(root / "papers.jsonl", _order_manifest(updated))


def _academic_impact_score(cited_by_count: int) -> float:
    return min(1.0, math.log1p(max(0, int(cited_by_count))) / 8.0)


def _role_priority_weight(role: str | None) -> float:
    return float(_ROLE_PRIORITY_WEIGHTS.get(str(role or "other"), _ROLE_PRIORITY_WEIGHTS["other"]))


def _priority_score(*, academic_impact: float, recency_score: float, reasoning_relevance: float, role_weight: float) -> float:
    return (
        0.45 * academic_impact
        + 0.25 * role_weight
        + 0.15 * recency_score
        + 0.15 * reasoning_relevance
    )


def _effective_sync_priority(paper: Paper) -> float:
    starvation_bonus = 0.08 if not paper.last_attempted_at else min(0.20, 0.02 * _days_since(paper.last_attempted_at))
    retry_penalty = min(0.15, 0.03 * max(0, int(paper.sync_attempt_count or 0)))
    return float(paper.priority_score or 0.0) + starvation_bonus - retry_penalty


def _days_since(timestamp: str | None) -> float:
    if not timestamp:
        return 0.0
    try:
        then = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    now = datetime.now(timezone.utc)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0.0, (now - then.astimezone(timezone.utc)).total_seconds() / 86400.0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _download_optional(client: Any, url: str | None) -> bytes | None:
    if not url:
        return None
    try:
        response = client.get(url)
    except Exception:
        return None
    if hasattr(response, "raise_for_status"):
        try:
            response.raise_for_status()
        except Exception:
            return None
    status_code = getattr(response, "status_code", 200)
    if status_code and int(status_code) >= 400:
        return None
    content = getattr(response, "content", None)
    if content is None and hasattr(response, "text"):
        return str(response.text).encode("utf-8", errors="ignore")
    return bytes(content or b"")


def _source_url_for_paper(paper: Paper) -> str | None:
    arxiv_id = _arxiv_id(paper)
    return f"https://export.arxiv.org/e-print/{arxiv_id}" if arxiv_id else None


def _html_url_for_paper(paper: Paper) -> str | None:
    arxiv_id = _arxiv_id(paper)
    return f"https://arxiv.org/html/{arxiv_id}" if arxiv_id else None


def _pdf_url_for_paper(paper: Paper) -> str | None:
    arxiv_id = _arxiv_id(paper)
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None


def _arxiv_id(paper: Paper) -> str:
    if paper.arxiv_id_full:
        return paper.arxiv_id_full
    if paper.paper_id.startswith("arxiv:"):
        return paper.paper_id.split(":", 1)[1]
    if paper.url and "/abs/" in paper.url:
        return paper.url.rsplit("/", 1)[-1]
    return ""


def _arxiv_full_from_paper_id(paper_id: str | None) -> str | None:
    if not paper_id:
        return None
    if paper_id.startswith("arxiv:"):
        return paper_id.split(":", 1)[1]
    return None


def _arxiv_base_from_paper_id(paper_id: str | None) -> str | None:
    return _arxiv_base_from_full(_arxiv_full_from_paper_id(paper_id))


def _arxiv_base_from_full(arxiv_id: str | None) -> str | None:
    if not _valid_arxiv_identifier(arxiv_id):
        return None
    return re.sub(r"v\d+$", "", arxiv_id)


def _valid_arxiv_identifier(arxiv_id: str | None) -> bool:
    if not arxiv_id:
        return False
    return bool(re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", arxiv_id, flags=re.IGNORECASE))


def _arxiv_version(arxiv_id: str | None) -> int:
    if not arxiv_id:
        return 0
    match = re.search(r"v(\d+)$", arxiv_id)
    if not match:
        return 1 if arxiv_id else 0
    return int(match.group(1))


def _parse_source_blob(blob: bytes) -> ParsedArtifact:
    tex_files = _extract_tex_files(blob)
    warnings: list[str] = []
    if not tex_files:
        text = _decode_best_effort(_maybe_gunzip(blob))
        sections, canonical = _sections_from_tex(text, source="tex")
        return ParsedArtifact(text=canonical, sections=sections, source="tex", warnings=["Source artifact contained no .tex files."])

    main_name = _select_main_tex_file(tex_files)
    flattened = _flatten_tex(main_name, tex_files, warnings)
    sections, canonical = _sections_from_tex(flattened, source="tex")
    return ParsedArtifact(text=canonical, sections=sections, source="tex", warnings=warnings)


def _extract_tex_files(blob: bytes) -> dict[str, str]:
    files: dict[str, str] = {}
    raw = _maybe_gunzip(blob)
    if _is_tar_bytes(raw):
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".tex"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                files[_normalize_tex_path(member.name)] = _decode_best_effort(extracted.read())
        return files
    decoded = _decode_best_effort(raw)
    if decoded.strip():
        files["main.tex"] = decoded
    return files


def _select_main_tex_file(files: dict[str, str]) -> str:
    best_name = ""
    best_score = float("-inf")
    for name, text in files.items():
        score = 0.0
        lower = name.lower()
        if "\\documentclass" in text:
            score += 5
        if "\\begin{document}" in text:
            score += 3
        score += min(3, text.count("\\section"))
        if any(token in lower for token in ("main", "paper", "root")):
            score += 1
        score += min(len(text) / 5000.0, 2.0)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name or sorted(files)[0]


def _flatten_tex(name: str, files: dict[str, str], warnings: list[str], seen: set[str] | None = None) -> str:
    seen = seen or set()
    normalized_name = _normalize_tex_path(name)
    if normalized_name in seen:
        warnings.append(f"Skipped recursive include for {normalized_name}.")
        return ""
    seen.add(normalized_name)
    text = _strip_tex_comments(files.get(normalized_name, ""))

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        resolved = _resolve_tex_include(normalized_name, target, files)
        if resolved is None:
            warnings.append(f"Missing include target {target!r} from {normalized_name}.")
            return ""
        return _flatten_tex(resolved, files, warnings, seen)

    return _INPUT_RE.sub(replace, text)


def _resolve_tex_include(current_name: str, target: str, files: dict[str, str]) -> str | None:
    base = Path(current_name).parent
    candidates = [target, f"{target}.tex"]
    for candidate in candidates:
        normalized = _normalize_tex_path(str(base / candidate))
        if normalized in files:
            return normalized
    target_name = Path(target).name
    for candidate in candidates:
        needle = Path(candidate).name
        for existing in files:
            if Path(existing).name == needle or Path(existing).stem == Path(needle).stem == target_name:
                return existing
    return None


def _sections_from_tex(text: str, *, source: str) -> tuple[list[PaperSection], str]:
    body = text
    begin = _BEGIN_DOC_RE.search(body)
    end = _END_DOC_RE.search(body)
    if begin:
        body = body[begin.end() :]
    if end:
        body = body[: end.start()]

    abstract_text = ""
    abstract_match = _ABSTRACT_RE.search(body)
    if abstract_match:
        abstract_text = abstract_match.group(1)
        body = body[: abstract_match.start()] + body[abstract_match.end() :]

    appendix_pos = None
    appendix_match = _APPENDIX_RE.search(body)
    if appendix_match:
        appendix_pos = appendix_match.start()
        body = body[: appendix_match.start()] + body[appendix_match.end() :]

    raw_sections: list[dict[str, Any]] = []
    if abstract_text.strip():
        raw_sections.append({"title": "Abstract", "kind": "abstract", "level": 0, "text": _clean_latex_text(abstract_text)})

    matches = list(_SECTION_RE.finditer(body))
    if not matches:
        fallback_text = _clean_latex_text(body)
        if fallback_text:
            raw_sections.append({"title": "Body", "kind": "other", "level": 0, "text": fallback_text})
        return _finalize_sections(raw_sections, source=source)

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        segment = _clean_latex_text(body[start:end])
        title = _clean_latex_text(match.group(2))
        level = {"section": 1, "subsection": 2, "subsubsection": 3}.get(match.group(1), 1)
        kind = "appendix" if appendix_pos is not None and match.start() > appendix_pos else ("section" if level == 1 else "subsection")
        raw_sections.append({"title": title or "Untitled", "kind": kind, "level": level, "text": segment})
    return _finalize_sections(raw_sections, source=source)


class _HeadingHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections: list[dict[str, Any]] = []
        self._capture_heading_level: int | None = None
        self._heading_parts: list[str] = []
        self._text_parts: list[str] = []
        self._ignore_depth = 0
        self._current: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._flush_text()
            self._capture_heading_level = int(tag[1])
            self._heading_parts = []
        elif tag == "br":
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"} and self._ignore_depth:
            self._ignore_depth -= 1
            return
        if self._ignore_depth:
            return
        if self._capture_heading_level and tag == f"h{self._capture_heading_level}":
            title = " ".join("".join(self._heading_parts).split())
            self._current = {
                "title": title or "Untitled",
                "kind": "section" if self._capture_heading_level == 1 else "subsection",
                "level": self._capture_heading_level,
                "text": "",
            }
            self.sections.append(self._current)
            self._capture_heading_level = None
            self._heading_parts = []
        elif tag in {"p", "div", "li", "section", "article"}:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        if self._capture_heading_level:
            self._heading_parts.append(data)
            return
        self._text_parts.append(data)

    def close(self) -> None:
        self._flush_text()
        super().close()

    def _flush_text(self) -> None:
        text = _clean_html_text("".join(self._text_parts))
        self._text_parts = []
        if not text:
            return
        if self._current is None:
            self._current = {"title": "Body", "kind": "other", "level": 0, "text": text}
            self.sections.append(self._current)
            return
        if self._current.get("text"):
            self._current["text"] += "\n\n" + text
        else:
            self._current["text"] = text


def _parse_html_blob(blob: bytes) -> ParsedArtifact:
    parser = _HeadingHTMLParser()
    parser.feed(blob.decode("utf-8", errors="ignore"))
    parser.close()
    sections, text = _finalize_sections(parser.sections, source="html")
    warnings = []
    if not sections and text:
        warnings.append("HTML parser fell back to body-only extraction.")
    return ParsedArtifact(text=text, sections=sections, source="html", warnings=warnings)


def _parse_pdf_blob(blob: bytes) -> ParsedArtifact:
    text = _extract_pdf_text(blob)
    sections, canonical = _sections_from_plain_text(text, source="pdf")
    warnings = []
    if len(blob) > _PDF_PARSE_MAX_CHARS:
        warnings.append("PDF parsing used a truncated scan window for performance.")
    if not sections and canonical:
        warnings.append("PDF parser produced body-only text without strong heading structure.")
    return ParsedArtifact(text=canonical, sections=sections, source="pdf", warnings=warnings)


def _extract_pdf_text(blob: bytes) -> str:
    raw = blob.decode("latin-1", errors="ignore")
    if len(raw) > _PDF_PARSE_MAX_CHARS:
        raw = raw[:_PDF_PARSE_MAX_CHARS]
    chunks: list[str] = []
    for match in _PDF_LITERAL_RE.finditer(raw):
        chunks.append(_unescape_pdf_literal(match.group(1)))
    for match in _PDF_ARRAY_RE.finditer(raw):
        for literal in _PDF_ARRAY_LITERAL_RE.findall(match.group(1)):
            chunks.append(_unescape_pdf_literal(literal))
    text = "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    if text.strip():
        return _clean_pdf_text(text)

    fallback = re.findall(r"[A-Za-z0-9][A-Za-z0-9 ,.;:()/%+\\-]{20,}", raw)
    return _clean_pdf_text("\n".join(fallback))


def _sections_from_plain_text(text: str, *, source: str) -> tuple[list[PaperSection], str]:
    lines = [line.strip() for line in text.splitlines()]
    raw_sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        current["text"] = _clean_plain_text("\n".join(current.get("lines", [])))
        current.pop("lines", None)
        raw_sections.append(current)
        current = None

    for line in lines:
        if not line:
            continue
        if line.lower() == "abstract":
            flush_current()
            current = {"title": "Abstract", "kind": "abstract", "level": 0, "lines": []}
            continue
        appendix = _APPENDIX_HEADING_RE.match(line)
        if appendix:
            flush_current()
            title = appendix.group(1).strip() or "Appendix"
            current = {"title": title, "kind": "appendix", "level": 1, "lines": []}
            continue
        heading = _PDF_HEADING_RE.match(line)
        if heading:
            flush_current()
            title = heading.group(1).strip()
            level = line.split(" ", 1)[0].count(".") + 1
            current = {"title": title, "kind": "section" if level == 1 else "subsection", "level": level, "lines": []}
            continue
        if current is None:
            current = {"title": "Body", "kind": "other", "level": 0, "lines": []}
        current.setdefault("lines", []).append(line)
    flush_current()
    return _finalize_sections(raw_sections, source=source)


def _build_sentences(text: str, sections: list[PaperSection]) -> list[PaperSentence]:
    by_section: dict[str, int] = {}
    sentences: list[PaperSentence] = []
    for idx, span in enumerate(sentence_spans(text)):
        section_id = None
        for section in sections:
            if section.char_start <= int(span["start"]) < section.char_end or (
                section.char_start == section.char_end and section.char_start == int(span["start"])
            ):
                section_id = section.section_id
                break
        section_index = by_section.get(section_id or "", 0)
        by_section[section_id or ""] = section_index + 1
        sentences.append(
            PaperSentence(
                sentence_id=f"s{idx + 1:04d}",
                section_id=section_id,
                text=str(span["sentence"]),
                char_start=int(span["start"]),
                char_end=int(span["end"]),
                sentence_index=idx,
                section_sentence_index=section_index,
            )
        )
    return sentences


def _finalize_sections(raw_sections: list[dict[str, Any]], *, source: str) -> tuple[list[PaperSection], str]:
    sections: list[PaperSection] = []
    parts: list[str] = []
    level_stack: dict[int, str] = {}
    cursor = 0
    for idx, raw in enumerate(raw_sections, start=1):
        text = _clean_plain_text(str(raw.get("text") or ""))
        title = _clean_plain_text(str(raw.get("title") or f"Section {idx}"))
        kind = str(raw.get("kind") or "other")
        level = int(raw.get("level") or 0)
        parent_id = None
        for probe in range(level - 1, -1, -1):
            if probe in level_stack:
                parent_id = level_stack[probe]
                break
        section_id = f"sec{idx:03d}"
        normalized_kind = kind if kind in {"abstract", "section", "subsection", "appendix", "other"} else "other"
        canonical_type, canonical_confidence, canonical_matched_by = _canonicalize_section_title(
            title or f"Section {idx}",
            normalized_kind,
        )
        level_stack[level] = section_id
        level_stack = {k: v for k, v in level_stack.items() if k <= level}
        start = cursor
        if text:
            parts.append(text)
            cursor += len(text)
            parts.append("\n\n")
            cursor += 2
        sections.append(
            PaperSection(
                section_id=section_id,
                title=title or f"Section {idx}",
                kind=normalized_kind,
                level=level,
                parent_id=parent_id,
                char_start=start,
                char_end=cursor - 2 if text else start,
                source=source,  # type: ignore[arg-type]
                canonical_type=canonical_type,  # type: ignore[arg-type]
                canonical_confidence=canonical_confidence,
                canonical_matched_by=canonical_matched_by,
            )
        )
    canonical = "".join(parts).strip()
    if canonical:
        for section in sections:
            if section.char_end > len(canonical):
                section.char_end = len(canonical)
    return sections, canonical


def _derive_parse_status(text: str, sections: list[PaperSection]) -> str:
    if text and sections:
        return "complete"
    if text:
        return "partial"
    return "failed"


def _canonicalize_section_title(title: str, kind: str) -> tuple[str, float, str]:
    normalized_title = _normalize_section_title(title)
    if kind == "abstract" or normalized_title == "abstract":
        return "abstract", 1.0, "kind_rule"

    for canonical_type, aliases in _SECTION_CANONICAL_ALIASES:
        for alias in aliases:
            normalized_alias = _normalize_section_title(alias)
            if normalized_title == normalized_alias:
                return canonical_type, 0.98, "title_exact"

    for canonical_type, aliases in _SECTION_CANONICAL_ALIASES:
        for alias in aliases:
            normalized_alias = _normalize_section_title(alias)
            if not normalized_alias:
                continue
            if normalized_alias in normalized_title or normalized_title in normalized_alias:
                return canonical_type, 0.9, "title_substring"

    title_tokens = set(_section_title_tokens(normalized_title))
    for canonical_type, aliases in _SECTION_CANONICAL_ALIASES:
        for alias in aliases:
            alias_tokens = set(_section_title_tokens(alias))
            if len(alias_tokens) >= 2 and alias_tokens.issubset(title_tokens):
                return canonical_type, 0.82, "title_token_rule"

    return "other", 0.0, "fallback_other"


def _normalize_section_title(title: str) -> str:
    lowered = title.lower().replace("&", " and ").replace("-", " ")
    lowered = _SECTION_TITLE_TOKEN_RE.sub(" ", lowered)
    return " ".join(lowered.split())


def _section_title_tokens(title: str) -> list[str]:
    normalized = _normalize_section_title(title)
    return [token for token in normalized.split() if token]


def _maybe_gunzip(blob: bytes) -> bytes:
    if blob[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(blob)
        except Exception:
            return blob
    return blob


def _is_tar_bytes(blob: bytes) -> bool:
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*"):
            return True
    except Exception:
        return False


def _normalize_tex_path(path: str) -> str:
    return str(Path(path).as_posix()).lstrip("./")


def _strip_tex_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        escaped = False
        chars: list[str] = []
        for char in line:
            if char == "%" and not escaped:
                break
            chars.append(char)
            escaped = char == "\\"
        lines.append("".join(chars))
    return "\n".join(lines)


def _clean_latex_text(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", cleaned)
    cleaned = re.sub(r"\\label\{[^}]*\}|\\ref\{[^}]*\}|\\cite\{[^}]*\}|\\url\{[^}]*\}", " ", cleaned)
    cleaned = _COMMAND_WITH_ARG_RE.sub(lambda m: f" {m.group(1)} ", cleaned)
    cleaned = _COMMAND_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = cleaned.replace("~", " ")
    cleaned = cleaned.replace("\\\\", "\n")
    return _clean_plain_text(cleaned)


def _clean_html_text(text: str) -> str:
    return _clean_plain_text(html_lib.unescape(text))


def _clean_pdf_text(text: str) -> str:
    return _clean_plain_text(text.replace("\\n", "\n"))


def _clean_plain_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _MULTISPACE_RE.sub(" ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = _BLANKLINE_RE.sub("\n\n", text)
    return text.strip()


def _decode_best_effort(blob: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return blob.decode(encoding)
        except Exception:
            continue
    return blob.decode("utf-8", errors="ignore")


def _unescape_pdf_literal(value: str) -> str:
    return (
        value.replace(r"\(", "(")
        .replace(r"\)", ")")
        .replace(r"\\", "\\")
        .replace(r"\n", "\n")
        .replace(r"\r", "")
    )


def _write_bytes(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


def _read_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
