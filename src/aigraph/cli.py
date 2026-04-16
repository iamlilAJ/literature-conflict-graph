"""`aigraph` CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .anomalies import detect_anomalies
from .extract import RuleBasedExtractor, extract_claims
from .graph import build_graph, load_graph, save_graph
from .hypotheses import generate_hypotheses
from .io import read_jsonl, write_jsonl
from .models import Anomaly, Claim, Hypothesis, Paper
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
    output: Path = typer.Option(DEFAULT_GRAPH, "--output"),
) -> None:
    """Build the typed claim graph and save as node-link JSON."""
    claim_records = read_jsonl(claims, Claim)
    g = build_graph(claim_records)
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
) -> None:
    """Generate candidate hypotheses per anomaly using deterministic templates."""
    anom_records = read_jsonl(anomalies, Anomaly)
    claim_records = read_jsonl(claims, Claim)
    hyps = generate_hypotheses(anom_records, claim_records)
    write_jsonl(output, hyps)
    console.print(f"[green]Generated {len(hyps)} hypotheses to[/] {output}")


@app.command("select")
def select_cmd(
    hypotheses: Path = typer.Option(DEFAULT_HYPOTHESES, "--hypotheses"),
    claims: Path = typer.Option(DEFAULT_CLAIMS, "--claims"),
    anomalies: Path = typer.Option(DEFAULT_ANOMALIES, "--anomalies"),
    k: int = typer.Option(4, "--k"),
    lambda_: float = typer.Option(0.7, "--lambda"),
    min_anomalies: int = typer.Option(2, "--min-anomalies"),
    papers: Optional[Path] = typer.Option(None, "--papers", help="Optional papers.jsonl for title/year in report"),
    output: Path = typer.Option(DEFAULT_REPORT, "--output"),
) -> None:
    """Score and select a diverse set of hypotheses; render Markdown report."""
    hyp_records = read_jsonl(hypotheses, Hypothesis)
    claim_records = read_jsonl(claims, Claim)
    anom_records = read_jsonl(anomalies, Anomaly)
    paper_lookup = None
    if papers is not None and papers.exists():
        paper_lookup = {p.paper_id: p for p in read_jsonl(papers, Paper)}

    scores = score_all(hyp_records, anom_records, claim_records)
    selected = select_mmr(hyp_records, scores, k=k, lambda_=lambda_, min_anomalies=min_anomalies)
    md = render_report(selected, anom_records, claim_records, scores, paper_lookup=paper_lookup)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    console.print(f"[green]Selected {len(selected)} hypotheses -> [/]{output}")


@app.command("fetch-openalex")
def fetch_openalex_cmd(
    query: str = typer.Option(..., "--query"),
    from_year: int = typer.Option(2020, "--from-year"),
    to_year: int = typer.Option(2026, "--to-year"),
    limit: int = typer.Option(50, "--limit"),
    output: Path = typer.Option(Path("data/openalex_papers.jsonl"), "--output"),
    mailto: Optional[str] = typer.Option(None, "--mailto", help="Contact email for OpenAlex polite pool"),
) -> None:
    """Fetch abstract-level AI papers from OpenAlex and save as papers.jsonl."""
    from .fetch_openalex import fetch_openalex_papers

    console.print(f"[cyan]Fetching OpenAlex works for:[/] {query}  ({from_year}-{to_year}, limit={limit})")
    papers = fetch_openalex_papers(query=query, from_year=from_year, to_year=to_year, limit=limit, mailto=mailto)
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
    report_path = output_dir / "selected_hypotheses.md"

    console.rule("[bold]aigraph real demo[/bold]")
    fetch_openalex_cmd(
        query=query, from_year=from_year, to_year=to_year, limit=limit,
        output=papers_path, mailto=mailto,
    )
    extract_cmd(input=papers_path, output=claims_path, extractor="llm", model=model, resume=False)
    build_graph_cmd(claims=claims_path, output=graph_path)
    detect_anomalies_cmd(graph=graph_path, claims=claims_path, output=anomalies_path)
    generate_hypotheses_cmd(anomalies=anomalies_path, claims=claims_path, output=hyps_path)
    select_cmd(
        hypotheses=hyps_path,
        claims=claims_path,
        anomalies=anomalies_path,
        k=k,
        lambda_=0.7,
        min_anomalies=2,
        papers=papers_path,
        output=report_path,
    )
    console.rule(f"[bold green]Done[/] -> {report_path}")


@app.command("run-demo")
def run_demo() -> None:
    """Run the full pipeline end-to-end on the sample data."""
    console.rule("[bold]aigraph demo[/bold]")
    init_sample(output=DEFAULT_PAPERS)
    extract_cmd(input=DEFAULT_PAPERS, output=DEFAULT_CLAIMS, extractor="rule", model=None, resume=False)
    build_graph_cmd(claims=DEFAULT_CLAIMS, output=DEFAULT_GRAPH)
    detect_anomalies_cmd(graph=DEFAULT_GRAPH, claims=DEFAULT_CLAIMS, output=DEFAULT_ANOMALIES)
    generate_hypotheses_cmd(anomalies=DEFAULT_ANOMALIES, claims=DEFAULT_CLAIMS, output=DEFAULT_HYPOTHESES)
    select_cmd(
        hypotheses=DEFAULT_HYPOTHESES,
        claims=DEFAULT_CLAIMS,
        anomalies=DEFAULT_ANOMALIES,
        k=4,
        lambda_=0.7,
        min_anomalies=2,
        papers=DEFAULT_PAPERS,
        output=DEFAULT_REPORT,
    )
    console.rule(f"[bold green]Done[/] -> {DEFAULT_REPORT}")


if __name__ == "__main__":
    app()
