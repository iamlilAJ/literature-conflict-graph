"""Tests for #49 extraction robustness: corpus-level direction-bias audit +
the per-paper retry / coverage report in extract_claims_with_status."""
from pathlib import Path

import aigraph.server as server
from aigraph.extraction_audit import audit_direction_bias
from aigraph.models import Claim, Paper


def _claim(cid: str, direction: str) -> Claim:
    return Claim(claim_id=cid, paper_id="p1", claim_text="t", direction=direction,
                 claim_type="performance_improvement")


# ---- direction-bias audit ------------------------------------------------

def test_monopositive_corpus_flags_bias():
    claims = [_claim(f"c{i}", "positive") for i in range(8)]
    a = audit_direction_bias(claims)
    assert a["positive_bias_suspected"] is True
    assert a["non_positive_count"] == 0
    assert a["positive_fraction"] == 1.0


def test_mixed_corpus_not_flagged():
    claims = [_claim(f"c{i}", "positive") for i in range(7)] + [_claim("cn", "negative")]
    a = audit_direction_bias(claims)
    assert a["positive_bias_suspected"] is False
    assert a["non_positive_count"] == 1


def test_small_corpus_not_flagged_even_if_all_positive():
    # below the threshold an all-positive run is plausibly just uncontested
    claims = [_claim(f"c{i}", "positive") for i in range(4)]
    assert audit_direction_bias(claims)["positive_bias_suspected"] is False


def test_audit_empty():
    a = audit_direction_bias([])
    assert a["claims_total"] == 0 and a["positive_bias_suspected"] is False


# ---- retry + coverage report in extract_claims_with_status ---------------

class _FakeRead:
    def __init__(self):
        self.candidates = []
        self.mode_used = "abstract"
        self.latency_sec = 0.0
        self.fallback_used = False
        self.prefilter_count = 0


def _paper(pid: str) -> Paper:
    return Paper(paper_id=pid, title=pid, year=2024, venue="arXiv", abstract="a")


def test_retry_recovers_empty_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "read_paper_candidates", lambda *a, **k: _FakeRead())
    monkeypatch.setenv("AIGRAPH_EXTRACT_MAX_ATTEMPTS", "3")

    calls = {"p_ok": 0, "p_empty": 0}

    class _FakeExtractor:
        def extract(self, paper, start_index=0, candidates=None):
            if paper.paper_id == "p_ok":
                calls["p_ok"] += 1
                # empty on first attempt, then a claim -> retry must recover it
                if calls["p_ok"] == 1:
                    return []
                return [Claim(claim_id="x", paper_id="p_ok", claim_text="t",
                              direction="positive", claim_type="performance_improvement")]
            calls["p_empty"] += 1
            return []  # genuinely empty every attempt

    monkeypatch.setattr(server, "LLMClaimExtractor", lambda *a, **k: _FakeExtractor())

    claims, report = server.extract_claims_with_status(
        [_paper("p_ok"), _paper("p_empty")], tmp_path / "claims.jsonl",
        status=lambda *a, **k: None, base_status={"run_id": "r"})

    assert len(claims) == 1                          # retry recovered p_ok's claim
    assert calls["p_ok"] == 2                        # retried once
    assert calls["p_empty"] == 3                     # exhausted attempts on empty
    assert report["papers_total"] == 2
    assert report["papers_with_claims"] == 1
    assert report["coverage"] == 0.5
    assert report["extraction_quality"] == "ok"      # 0.5 is not < floor(0.5)
    assert report["retried_papers"] == 2             # both needed >1 attempt
    # per-paper status artifact written
    rows = (tmp_path / "extraction_status.jsonl").read_text().strip().splitlines()
    assert len(rows) == 2


def test_empty_corpus_quality_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "read_paper_candidates", lambda *a, **k: _FakeRead())

    class _Empty:
        def extract(self, *a, **k):
            return []

    monkeypatch.setattr(server, "LLMClaimExtractor", lambda *a, **k: _Empty())
    claims, report = server.extract_claims_with_status(
        [_paper("p1")], tmp_path / "c.jsonl", status=lambda *a, **k: None,
        base_status={"run_id": "r"})
    assert claims == [] and report["extraction_quality"] == "empty"
