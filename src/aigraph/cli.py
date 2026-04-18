"""`aigraph` CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .anomalies import detect_anomalies
from .automation import build_fix_bundle, critique_runs, harvest_topics, render_crontab, run_fix_session, run_preflight_checks, run_topic_batch
from .extract import RuleBasedExtractor, extract_claims
from .graph import build_graph, load_graph, save_graph
from .hypotheses import generate_hypotheses
from .io import read_jsonl, write_jsonl
from .models import Anomaly, Claim, Hypothesis, Insight, Paper
from .report import render_report
from .sample_data import build_sample_papers
from .scoring import score_all, select_mmr
from .visualize import render_visualization


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


_load_dotenv_if_available()

app = typer.Typer(add_completion=False, help="Graph-based literature conflict explorer for AI paper claims.")
console = Console()


DEFAULT_PAPERS = Path("data/sample_papers.jsonl")
DEFAULT_CLAIMS = Path("outputs/claims.jsonl")
DEFAULT_GRAPH = Path("outputs/graph.json")
DEFAULT_ANOMALIES = Path("outputs/anomalies.jsonl")
DEFAULT_HYPOTHESES = Path("outputs/hypotheses.jsonl")
DEFAULT_REPORT = Path("outputs/selected_hypotheses.md")
DEFAULT_AUTOMATION = Path("automation")


@app.command("init-sample")
def init_sample(output: Path = typer.Option(DEFAULT_PAPERS, "--output")) -> None:
    """Write the synthetic sample papers to disk."""
    papers = build_sample_papers()
    write_jsonl(output, papers)
    console.print(f"[green]Wrote {len(papers)} sample papers to[/] {output}")


@app.command("extract")
def extract_cmd(
    input: Path = typer.Option(DEFAULT_PAPERS, "--input"),
    output: Path = typer.Option(DEFAULT_CLAIMS, "--output"),
    extractor: str = typer.Option("rule", "--extractor", help="rule|llm"),
    model: Optional[str] = typer.Option(None, "--model", help="LLM model id (overrides AIGRAPH_MODEL)"),
    resume: bool = typer.Option(False, "--resume/--no-resume", help="Append only missing papers when output exists"),
) -> None:
    """Extract typed claims from papers."""
    papers = read_jsonl(input, Paper)
    extractor_impl = _build_extractor(extractor, model)
    if (extractor or "rule").lower() == "llm":
        claims = _extract_claims_incremental(papers, extractor_impl, output, resume=resume)
    else:
        claims = extract_claims(papers, extractor=extractor_impl)
        write_jsonl(output, claims)
    console.print(
        f"[green]Extracted {len(claims)} claims to[/] {output} "
        f"[dim](extractor={extractor})[/]"
    )


def _build_extractor(kind: str, model: Optional[str]):
    kind = (kind or "rule").lower()
    if kind == "rule":
        return RuleBasedExtractor()
    if kind == "llm":
        from .llm_extract import LLMClaimExtractor

        return LLMClaimExtractor(model=model)
    raise typer.BadParameter(f"Unknown extractor '{kind}'. Use 'rule' or 'llm'.")


def _extract_claims_incremental(papers: list[Paper], extractor_impl, output: Path, resume: bool = False) -> list[Claim]:
    output.parent.mkdir(parents=True, exist_ok=True)
    claims: list[Claim] = []
    completed_papers: set[str] = set()
    mode = "w"

    if resume and output.exists():
        claims = read_jsonl(output, Claim)
        completed_papers = {c.paper_id for c in claims}
        mode = "a"
        console.print(
            f"[cyan]Resuming extraction:[/] {len(completed_papers)} paper(s), "
            f"{len(claims)} claim(s) already in {output}"
        )

    with output.open(mode, encoding="utf-8") as f:
        for i, paper in enumerate(papers, start=1):
            if paper.paper_id in completed_papers:
                console.print(f"[dim]Skipping {i}/{len(papers)} {paper.paper_id} already extracted[/]")
                continue

            title = paper.title[:80] + ("..." if len(paper.title) > 80 else "")
            console.print(f"[cyan]Extracting {i}/{len(papers)}[/] {paper.paper_id} — {title}")
            try:
                new_claims = extractor_impl.extract(paper, start_index=len(claims))
            except Exception as e:  # pragma: no cover - defensive; LLM extractor normally catches network errors
                console.print(f"[yellow]  warning:[/] extraction failed for {paper.paper_id}: {e}")
                new_claims = []

            for claim in new_claims:
                f.write(claim.model_dump_json(by_alias=True))
                f.write("\n")
            f.flush()
            claims.extend(new_claims)
            console.print(f"[green]  +{len(new_claims)} claim(s)[/] total={len(claims)}")

    return claims


@app.command("build-graph")
def build_graph_cmd(
    claims: Path = typer.Option(DEFAULT_CLAIMS, "--claims"),
    papers: Optional[Path] = typer.Option(None, "--papers", help="Optional papers.jsonl for citation metadata"),
    output: Path = typer.Option(DEFAULT_GRAPH, "--output"),
) -> None:
    """Build the typed claim graph and save as node-link JSON."""
    claim_records = read_jsonl(claims, Claim)
    paper_records = read_jsonl(papers, Paper) if papers is not None and papers.exists() else None
    g = build_graph(claim_records, papers=paper_records)
    save_graph(g, output)
    console.print(
        f"[green]Built graph with {g.number_of_nodes()} nodes, "
        f"{g.number_of_edges()} edges -> [/]{output}"
    )


@app.command("detect-anomalies")
def detect_anomalies_cmd(
    graph: Path = typer.Option(DEFAULT_GRAPH, "--graph"),
    claims: Path = typer.Option(DEFAULT_CLAIMS, "--claims"),
    output: Path = typer.Option(DEFAULT_ANOMALIES, "--output"),
) -> None:
    """Detect anomalies in the graph."""
    g = load_graph(graph)
    claim_records = read_jsonl(claims, Claim)
    anomalies = detect_anomalies(g, claim_records)
    write_jsonl(output, anomalies)
    by_type: dict[str, int] = {}
    for a in anomalies:
        by_type[a.type] = by_type.get(a.type, 0) + 1
    console.print(f"[green]Found {len(anomalies)} anomalies[/] {by_type} -> {output}")


@app.command("generate-hypotheses")
def generate_hypotheses_cmd(
    anomalies: Path = typer.Option(DEFAULT_ANOMALIES, "--anomalies"),
    claims: Path = typer.Option(DEFAULT_CLAIMS, "--claims"),
    output: Path = typer.Option(DEFAULT_HYPOTHESES, "--output"),
    generator: str = typer.Option("template", "--generator", help="template|llm"),
    model: Optional[str] = typer.Option(None, "--model", help="LLM model id when --generator llm"),
) -> None:
    """Generate candidate explanations per anomaly."""
    anom_records = read_jsonl(anomalies, Anomaly)
    claim_records = read_jsonl(claims, Claim)
    generator_impl = _build_hypothesis_generator(generator, model)
    hyps = generate_hypotheses(anom_records, claim_records, generator=generator_impl)
    write_jsonl(output, hyps)
    console.print(f"[green]Generated {len(hyps)} hypotheses to[/] {output} [dim](generator={generator})[/]")


@app.command("generate-insights")
def generate_insights_cmd(
    graph: Path = typer.Option(DEFAULT_GRAPH, "--graph"),
    claims: Path = typer.Option(DEFAULT_CLAIMS, "--claims"),
    papers: Path = typer.Option(DEFAULT_PAPERS, "--papers"),
    anomalies: Path = typer.Option(DEFAULT_ANOMALIES, "--anomalies"),
    output: Path = typer.Option(Path("outputs/insights.jsonl"), "--output"),
    generator: str = typer.Option("template", "--generator", help="template|llm"),
    model: Optional[str] = typer.Option(None, "--model", help="LLM model id when --generator llm"),
) -> None:
    """Generate community-level topology/citation insights."""
    from .insights import generate_insights

    g = load_graph(graph)
    claim_records = read_jsonl(claims, Claim)
    paper_records = read_jsonl(papers, Paper) if papers.exists() else []
    anom_records = read_jsonl(anomalies, Anomaly) if anomalies.exists() else []
    generator_impl = _build_insight_generator(generator, model)
    insights = generate_insights(g, claim_records, paper_records, anom_records, generator=generator_impl)
    write_jsonl(output, insights)
    console.print(f"[green]Generated {len(insights)} insights to[/] {output} [dim](generator={generator})[/]")


def _build_insight_generator(kind: str, model: Optional[str]):
    kind = (kind or "template").lower()
    if kind == "template":
        from .insights import TemplateInsightGenerator

        return TemplateInsightGenerator()
    if kind == "llm":
        from .insights import LLMInsightGenerator

        return LLMInsightGenerator(model=model)
    raise typer.BadParameter(f"Unknown generator '{kind}'. Use 'template' or 'llm'.")


def _build_hypothesis_generator(kind: str, model: Optional[str]):
    kind = (kind or "template").lower()
    if kind == "template":
        from .hypotheses import TemplateGenerator

        return TemplateGenerator()
    if kind == "llm":
        from .llm_hypotheses import LLMHypothesisGenerator

        return LLMHypothesisGenerator(model=model)
    raise typer.BadParameter(f"Unknown generator '{kind}'. Use 'template' or 'llm'.")


@app.command("select")
def select_cmd(
    hypotheses: Path = typer.Option(DEFAULT_HYPOTHESES, "--hypotheses"),
    claims: Path = typer.Option(DEFAULT_CLAIMS, "--claims"),
    anomalies: Path = typer.Option(DEFAULT_ANOMALIES, "--anomalies"),
    k: int = typer.Option(4, "--k"),
    lambda_: float = typer.Option(0.7, "--lambda"),
    min_anomalies: int = typer.Option(2, "--min-anomalies"),
    papers: Optional[Path] = typer.Option(None, "--papers", help="Optional papers.jsonl for title/year in report"),
    insights: Optional[Path] = typer.Option(None, "--insights", help="Optional insights.jsonl for community insight report"),
    output: Path = typer.Option(DEFAULT_REPORT, "--output"),
) -> None:
    """Score and select a diverse set of hypotheses; render Markdown report."""
    hyp_records = read_jsonl(hypotheses, Hypothesis)
    claim_records = read_jsonl(claims, Claim)
    anom_records = read_jsonl(anomalies, Anomaly)
    paper_lookup = None
    if papers is not None and papers.exists():
        paper_lookup = {p.paper_id: p for p in read_jsonl(papers, Paper)}
    insight_records = read_jsonl(insights, Insight) if insights is not None and insights.exists() else []

    scores = score_all(hyp_records, anom_records, claim_records)
    selected = select_mmr(hyp_records, scores, k=k, lambda_=lambda_, min_anomalies=min_anomalies)
    md = render_report(
        selected,
        anom_records,
        claim_records,
        scores,
        paper_lookup=paper_lookup,
        insights=insight_records,
        paper_count=len(paper_lookup or {}),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    console.print(f"[green]Selected {len(selected)} hypotheses -> [/]{output}")


@app.command("fetch-openalex")
def fetch_openalex_cmd(
    query: str = typer.Option(..., "--query"),
    from_year: int = typer.Option(2020, "--from-year"),
    to_year: int = typer.Option(2026, "--to-year"),
    limit: int = typer.Option(50, "--limit"),
    strategy: str = typer.Option("balanced", "--strategy", help="balanced|high-impact|recent"),
    citation_weight: Optional[float] = typer.Option(
        None,
        "--citation-weight",
        help="Optional 0..0.85 impact weight for OpenAlex reranking",
    ),
    min_relevance: Optional[float] = typer.Option(
        None,
        "--min-relevance",
        help="Optional 0..1 relevance gate before citation reranking",
    ),
    output: Path = typer.Option(Path("data/openalex_papers.jsonl"), "--output"),
    mailto: Optional[str] = typer.Option(None, "--mailto", help="Contact email for OpenAlex polite pool"),
) -> None:
    """Fetch abstract-level AI papers from OpenAlex and save as papers.jsonl."""
    from .fetch_openalex import fetch_openalex_papers

    console.print(
        f"[cyan]Fetching OpenAlex works for:[/] {query}  "
        f"({from_year}-{to_year}, limit={limit}, strategy={strategy}, "
        f"citation_weight={citation_weight}, min_relevance={min_relevance})"
    )
    papers = fetch_openalex_papers(
        query=query,
        from_year=from_year,
        to_year=to_year,
        limit=limit,
        mailto=mailto,
        strategy=strategy,
        citation_weight=citation_weight,
        min_relevance=min_relevance,
    )
    write_jsonl(output, papers)
    console.print(f"[green]Saved {len(papers)} papers to[/] {output}")


@app.command("fetch-arxiv")
def fetch_arxiv_cmd(
    query: str = typer.Option(..., "--query"),
    from_year: int = typer.Option(2020, "--from-year"),
    to_year: int = typer.Option(2026, "--to-year"),
    limit: int = typer.Option(50, "--limit"),
    strategy: str = typer.Option("balanced", "--strategy", help="balanced|recent"),
    output: Path = typer.Option(Path("data/arxiv_papers.jsonl"), "--output"),
) -> None:
    """Fetch abstract-level papers from arXiv and save as papers.jsonl."""
    from .fetch_arxiv import fetch_arxiv_papers

    console.print(
        f"[cyan]Fetching arXiv papers for:[/] {query}  "
        f"({from_year}-{to_year}, limit={limit}, strategy={strategy})"
    )
    papers = fetch_arxiv_papers(
        query=query,
        from_year=from_year,
        to_year=to_year,
        limit=limit,
        strategy=strategy,
    )
    write_jsonl(output, papers)
    console.print(f"[green]Saved {len(papers)} papers to[/] {output}")


@app.command("visualize")
def visualize_cmd(
    input_dir: Path = typer.Option(Path("outputs/demo_kimi_full"), "--input-dir"),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    """Render a static HTML graph explorer for an aigraph output directory."""
    output_path = output or (input_dir / "index.html")
    render_visualization(input_dir, output_path)
    console.print(f"[green]Wrote visualization to[/] {output_path}")


@app.command("run-real-demo")
def run_real_demo(
    query: str = typer.Option("retrieval augmented generation large language models", "--query"),
    from_year: int = typer.Option(2020, "--from-year"),
    to_year: int = typer.Option(2026, "--to-year"),
    limit: int = typer.Option(50, "--limit"),
    strategy: str = typer.Option("balanced", "--strategy", help="balanced|high-impact|recent"),
    citation_weight: Optional[float] = typer.Option(None, "--citation-weight"),
    min_relevance: Optional[float] = typer.Option(None, "--min-relevance"),
    model: Optional[str] = typer.Option(None, "--model"),
    mailto: Optional[str] = typer.Option(None, "--mailto"),
    k: int = typer.Option(4, "--k"),
    output_dir: Path = typer.Option(Path("outputs/openalex_rag"), "--output-dir"),
) -> None:
    """Fetch OpenAlex -> LLM-extract -> graph -> anomalies -> hypotheses -> report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    papers_path = output_dir / "papers.jsonl"
    claims_path = output_dir / "claims.jsonl"
    graph_path = output_dir / "graph.json"
    anomalies_path = output_dir / "anomalies.jsonl"
    hyps_path = output_dir / "hypotheses.jsonl"
    insights_path = output_dir / "insights.jsonl"
    report_path = output_dir / "selected_hypotheses.md"

    console.rule("[bold]aigraph real demo[/bold]")
    fetch_openalex_cmd(
        query=query, from_year=from_year, to_year=to_year, limit=limit,
        strategy=strategy, citation_weight=citation_weight, min_relevance=min_relevance,
        output=papers_path, mailto=mailto,
    )
    extract_cmd(input=papers_path, output=claims_path, extractor="llm", model=model, resume=False)
    build_graph_cmd(claims=claims_path, papers=papers_path, output=graph_path)
    detect_anomalies_cmd(graph=graph_path, claims=claims_path, output=anomalies_path)
    generate_hypotheses_cmd(
        anomalies=anomalies_path,
        claims=claims_path,
        output=hyps_path,
        generator="template",
        model=None,
    )
    generate_insights_cmd(
        graph=graph_path,
        claims=claims_path,
        papers=papers_path,
        anomalies=anomalies_path,
        output=insights_path,
        generator="template",
        model=None,
    )
    select_cmd(
        hypotheses=hyps_path,
        claims=claims_path,
        anomalies=anomalies_path,
        k=k,
        lambda_=0.7,
        min_anomalies=2,
        papers=papers_path,
        insights=insights_path,
        output=report_path,
    )
    console.rule(f"[bold green]Done[/] -> {report_path}")


@app.command("run-arxiv-demo")
def run_arxiv_demo(
    query: str = typer.Option('all:"large language models" AND (all:finance OR all:"time series" OR all:forecasting)', "--query"),
    from_year: int = typer.Option(2020, "--from-year"),
    to_year: int = typer.Option(2026, "--to-year"),
    limit: int = typer.Option(30, "--limit"),
    strategy: str = typer.Option("balanced", "--strategy", help="balanced|recent"),
    model: Optional[str] = typer.Option(None, "--model"),
    k: int = typer.Option(8, "--k"),
    output_dir: Path = typer.Option(Path("outputs/arxiv_finance_timeseries"), "--output-dir"),
    insight_generator: str = typer.Option("template", "--insight-generator", help="template|llm"),
) -> None:
    """Fetch arXiv -> LLM-extract -> graph -> anomalies -> insights -> report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    papers_path = output_dir / "papers.jsonl"
    claims_path = output_dir / "claims.jsonl"
    graph_path = output_dir / "graph.json"
    anomalies_path = output_dir / "anomalies.jsonl"
    hyps_path = output_dir / "hypotheses.jsonl"
    insights_path = output_dir / "insights.jsonl"
    report_path = output_dir / "selected_hypotheses.md"
    html_path = output_dir / "index.html"

    console.rule("[bold]aigraph arXiv demo[/bold]")
    fetch_arxiv_cmd(
        query=query,
        from_year=from_year,
        to_year=to_year,
        limit=limit,
        strategy=strategy,
        output=papers_path,
    )
    extract_cmd(input=papers_path, output=claims_path, extractor="llm", model=model, resume=False)
    build_graph_cmd(claims=claims_path, papers=papers_path, output=graph_path)
    detect_anomalies_cmd(graph=graph_path, claims=claims_path, output=anomalies_path)
    generate_hypotheses_cmd(
        anomalies=anomalies_path,
        claims=claims_path,
        output=hyps_path,
        generator="template",
        model=None,
    )
    generate_insights_cmd(
        graph=graph_path,
        claims=claims_path,
        papers=papers_path,
        anomalies=anomalies_path,
        output=insights_path,
        generator=insight_generator,
        model=model,
    )
    select_cmd(
        hypotheses=hyps_path,
        claims=claims_path,
        anomalies=anomalies_path,
        k=k,
        lambda_=0.7,
        min_anomalies=2,
        papers=papers_path,
        insights=insights_path,
        output=report_path,
    )
    visualize_cmd(input_dir=output_dir, output=html_path)
    console.rule(f"[bold green]Done[/] -> {html_path}")


@app.command("run-demo")
def run_demo() -> None:
    """Run the full pipeline end-to-end on the sample data."""
    console.rule("[bold]aigraph demo[/bold]")
    init_sample(output=DEFAULT_PAPERS)
    extract_cmd(input=DEFAULT_PAPERS, output=DEFAULT_CLAIMS, extractor="rule", model=None, resume=False)
    build_graph_cmd(claims=DEFAULT_CLAIMS, papers=DEFAULT_PAPERS, output=DEFAULT_GRAPH)
    detect_anomalies_cmd(graph=DEFAULT_GRAPH, claims=DEFAULT_CLAIMS, output=DEFAULT_ANOMALIES)
    generate_hypotheses_cmd(
        anomalies=DEFAULT_ANOMALIES,
        claims=DEFAULT_CLAIMS,
        output=DEFAULT_HYPOTHESES,
        generator="template",
        model=None,
    )
    select_cmd(
        hypotheses=DEFAULT_HYPOTHESES,
        claims=DEFAULT_CLAIMS,
        anomalies=DEFAULT_ANOMALIES,
        k=4,
        lambda_=0.7,
        min_anomalies=2,
        papers=DEFAULT_PAPERS,
        insights=None,
        output=DEFAULT_REPORT,
    )
    console.rule(f"[bold green]Done[/] -> {DEFAULT_REPORT}")


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs-dir"),
) -> None:
    """Run the local Baidu-style literature search web server."""
    from .server import serve

    serve(host=host, port=port, runs_dir=runs_dir)


@app.command("rebuild-community")
def rebuild_community_cmd(
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs-dir"),
) -> None:
    """Rebuild the living community graph from completed runs."""
    from .community import rebuild_community

    status = rebuild_community(runs_dir)
    console.print(
        "[green]Rebuilt living graph:[/] "
        f"{status.get('runs', 0)} runs · {status.get('papers', 0)} papers · "
        f"{status.get('claims', 0)} claims -> {runs_dir / '_community' / 'index.html'}"
    )


@app.command("automation-harvest")
def automation_harvest_cmd(
    automation_dir: Path = typer.Option(DEFAULT_AUTOMATION, "--automation-dir"),
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs-dir"),
    limit: int = typer.Option(12, "--limit"),
    cooldown_hours: int = typer.Option(12, "--cooldown-hours"),
) -> None:
    """Harvest seed topics and recent searches into a pending automation queue."""
    topics = harvest_topics(automation_dir, runs_dir, limit=limit, cooldown_hours=cooldown_hours)
    console.print(f"[green]Queued {len(topics)} topic(s) for automation[/] -> {automation_dir / 'topics' / 'generated_topics.jsonl'}")


@app.command("automation-run-batch")
def automation_run_batch_cmd(
    automation_dir: Path = typer.Option(DEFAULT_AUTOMATION, "--automation-dir"),
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs-dir"),
    batch_size: int = typer.Option(3, "--batch-size"),
) -> None:
    """Run a small batch of pending automation topics through the existing pipeline."""
    results = run_topic_batch(automation_dir, runs_dir, batch_size=batch_size)
    console.print(f"[green]Processed {len(results)} automated run(s)[/] -> {automation_dir / 'runs' / 'runs_index.jsonl'}")


@app.command("automation-critic")
def automation_critic_cmd(
    automation_dir: Path = typer.Option(DEFAULT_AUTOMATION, "--automation-dir"),
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs-dir"),
    limit: int = typer.Option(8, "--limit"),
) -> None:
    """Critique completed runs and emit normalized product/code issues."""
    issues = critique_runs(automation_dir, runs_dir, limit=limit)
    console.print(f"[green]Generated {len(issues)} issue(s)[/] -> {automation_dir / 'issues' / 'issues.jsonl'}")


@app.command("automation-fix-bundle")
def automation_fix_bundle_cmd(
    automation_dir: Path = typer.Option(DEFAULT_AUTOMATION, "--automation-dir"),
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs-dir"),
    max_issues: int = typer.Option(3, "--max-issues"),
) -> None:
    """Build a Codex-ready bundle of top issues for repo fixing."""
    bundle = build_fix_bundle(automation_dir, runs_dir, max_issues=max_issues)
    console.print(
        f"[green]Built fix bundle with {bundle.get('issue_count', 0)} issue(s)[/] "
        f"-> {automation_dir / 'issues' / 'fix_bundle.json'}"
    )


@app.command("automation-fix-run")
def automation_fix_run_cmd(
    automation_dir: Path = typer.Option(DEFAULT_AUTOMATION, "--automation-dir"),
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs-dir"),
    repo_dir: Path = typer.Option(Path("."), "--repo-dir"),
    max_issues: int = typer.Option(3, "--max-issues"),
    codex_command: Optional[str] = typer.Option(
        None,
        "--codex-command",
        help="Shell command template used to invoke Codex. Supports {repo_dir} {bundle_path} {prompt_path} {pr_body_path} {branch} {test_command}.",
    ),
    branch_prefix: str = typer.Option("codex/automation-fix", "--branch-prefix"),
    test_command: str = typer.Option("./.venv/bin/pytest -q", "--test-command"),
    push: bool = typer.Option(False, "--push/--no-push"),
    open_pr: bool = typer.Option(False, "--open-pr/--no-open-pr"),
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run"),
) -> None:
    """Run the nightly repo-fixer flow and optionally push a draft PR."""
    result = run_fix_session(
        automation_dir,
        runs_dir,
        repo_dir=repo_dir,
        max_issues=max_issues,
        codex_command=codex_command,
        branch_prefix=branch_prefix,
        test_command=test_command,
        push=push,
        open_pr=open_pr,
        dry_run=dry_run,
    )
    console.print(f"[green]Fix session status:[/] {result.get('status', 'unknown')}")
    console.print(f"[cyan]Bundle:[/] {result.get('bundle_path')}")
    console.print(f"[cyan]Prompt:[/] {result.get('prompt_path')}")
    if result.get("branch"):
        console.print(f"[cyan]Branch:[/] {result.get('branch')}")
    if result.get("commit"):
        console.print(f"[cyan]Commit:[/] {result.get('commit')}")
    if result.get("pr_url"):
        console.print(f"[cyan]Draft PR:[/] {result.get('pr_url')}")
    if result.get("error"):
        console.print(f"[yellow]Note:[/] {result.get('error')}")


@app.command("automation-crontab")
def automation_crontab_cmd(
    repo_dir: Path = typer.Option(Path("."), "--repo-dir"),
    python_bin: str = typer.Option("./.venv/bin/python", "--python-bin"),
    automation_dir: Path = typer.Option(DEFAULT_AUTOMATION, "--automation-dir"),
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs-dir"),
    batch_size: int = typer.Option(3, "--batch-size"),
    critic_limit: int = typer.Option(8, "--critic-limit"),
    max_fix_issues: int = typer.Option(3, "--max-fix-issues"),
    output: Optional[Path] = typer.Option(None, "--output"),
) -> None:
    """Render a ready-to-install crontab for the aigraph automation loop."""
    payload = render_crontab(
        repo_dir=repo_dir,
        python_bin=python_bin,
        automation_dir=automation_dir,
        runs_dir=runs_dir,
        batch_size=batch_size,
        critic_limit=critic_limit,
        max_fix_issues=max_fix_issues,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        console.print(f"[green]Wrote crontab template to[/] {output}")
        return
    console.print(payload, soft_wrap=True)


@app.command("automation-preflight")
def automation_preflight_cmd(
    repo_dir: Path = typer.Option(Path("."), "--repo-dir"),
    python_bin: str = typer.Option("./.venv/bin/python", "--python-bin"),
) -> None:
    """Check whether nightly automated draft PRs are ready to run."""
    report = run_preflight_checks(repo_dir=repo_dir, python_bin=python_bin)
    console.print(f"[green]Automation ready:[/] {report.get('ready')}")
    for item in report.get("checks", []):
        marker = "[green]OK[/]" if item.get("ok") else "[yellow]WARN[/]"
        console.print(f"{marker} {item.get('name')}: {item.get('detail')}")


if __name__ == "__main__":
    app()
