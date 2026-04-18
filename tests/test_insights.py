import json

from aigraph.graph import build_graph
from aigraph.insights import LLMInsightGenerator, TemplateInsightGenerator, prune_insights
from aigraph.models import Claim, Paper


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **kwargs):
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


def _claim(cid: str, paper_id: str, domain: str) -> Claim:
    return Claim(
        claim_id=cid,
        paper_id=paper_id,
        claim_text=f"{domain} claim {cid}",
        method="LLM",
        task="forecasting",
        canonical_task="other",
        direction="positive",
        domain=domain,
        data_modality="text + time series",
        mechanism="event grounding",
        failure_mode="temporal leakage",
        temporal_property="non-stationarity",
    )


def _papers(cites: bool = False) -> list[Paper]:
    return [
        Paper(
            paper_id="openalex:W1",
            title="Finance LLMs",
            year=2024,
            venue="ACL",
            cited_by_count=10,
            referenced_works=["openalex:W3"] if cites else [],
        ),
        Paper(paper_id="openalex:W2", title="Finance Forecasting", year=2024, venue="ACL"),
        Paper(paper_id="openalex:W3", title="Time Series LLMs", year=2024, venue="NeurIPS"),
        Paper(paper_id="openalex:W4", title="Temporal Foundation Models", year=2024, venue="NeurIPS"),
    ]


def _claims() -> list[Claim]:
    return [
        _claim("c001", "openalex:W1", "finance"),
        _claim("c002", "openalex:W2", "finance"),
        _claim("c003", "openalex:W3", "time series"),
        _claim("c004", "openalex:W4", "time series"),
    ]


def test_template_insight_generates_unifying_theory_for_disconnected_communities():
    claims = _claims()
    papers = _papers(cites=False)
    g = build_graph(claims, papers=papers)
    insights = TemplateInsightGenerator().generate(g, claims, papers, [])
    assert insights
    insight = insights[0]
    assert insight.type == "unifying_theory"
    assert insight.communities == ["finance", "time series"]
    assert "non-stationarity" in insight.shared_concepts
    assert "No internal citation path" in insight.citation_gap


def test_template_insight_skips_directly_connected_communities():
    claims = _claims()
    papers = _papers(cites=True)
    g = build_graph(claims, papers=papers)
    insights = TemplateInsightGenerator().generate(g, claims, papers, [])
    assert insights == []


def test_llm_insight_rewrites_template_output_with_fake_client():
    claims = _claims()
    papers = _papers(cites=False)
    g = build_graph(claims, papers=papers)
    payload = {
        "title": "Finance and time-series LLMs share temporal reasoning problems",
        "insight": "Both communities study text-conditioned forecasting under non-stationarity.",
        "unifying_frame": "Language-conditioned temporal forecasting under regime shift.",
        "citation_gap": "The communities are conceptually close but citation-disconnected.",
        "transfer_suggestions": ["Transfer backtesting protocols into general LLM time-series benchmarks."],
    }
    generator = LLMInsightGenerator(model="stub", client=_FakeClient(json.dumps(payload)), api_key="test-key")
    insights = generator.generate(g, claims, papers, [])
    assert insights[0].title == payload["title"]
    assert insights[0].unifying_frame == payload["unifying_frame"]


def test_prune_insights_drops_generic_community_labels():
    claims = _claims()
    insights = [
        TemplateInsightGenerator().generate(build_graph(claims, papers=_papers(cites=False)), claims, _papers(cites=False), [])[0],
    ]
    generic = insights[0].model_copy(
        update={
            "communities": ["scientific literature", "scientific research"],
            "title": "Scientific literature and scientific research may share a unifying mechanism",
            "shared_concepts": ["evaluation protocol", "modality alignment"],
            "confidence_score": 0.7,
            "topology_score": 0.7,
        }
    )
    pruned = prune_insights([generic], claims)
    assert pruned
    assert pruned[0].communities != ["scientific literature", "scientific research"]
    assert pruned[0].quality_score > 0


def test_prune_insights_filters_weak_keyword_only_bridge():
    claims = _claims()
    weak = TemplateInsightGenerator().generate(build_graph(claims, papers=_papers(cites=False)), claims, _papers(cites=False), [])[0]
    weak = weak.model_copy(
        update={
            "shared_concepts": ["forecasting"],
            "communities": ["research", "literature"],
            "confidence_score": 0.12,
            "topology_score": 0.18,
        }
    )
    assert prune_insights([weak], claims) == []
