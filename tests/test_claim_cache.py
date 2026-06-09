"""Tests for the cross-run claim-extraction cache (src/aigraph/claim_cache.py)."""
import json

from aigraph import claim_cache
from aigraph.models import Claim, Paper


def _paper(pid="2401.00001", *, abstract="an abstract", text="", arxiv_base=None) -> Paper:
    return Paper(
        paper_id=pid,
        title=f"title-{pid}",
        year=2024,
        venue="arXiv",
        abstract=abstract,
        text=text,
        arxiv_id_base=arxiv_base,
    )


def _claims(pid="2401.00001", n=2) -> list[Claim]:
    return [
        Claim(claim_id=f"{pid}#c{i:02d}", paper_id=pid, claim_text=f"claim {i}", method="RAG", task="QA")
        for i in range(1, n + 1)
    ]


# --- key stability / sensitivity ---------------------------------------------

def test_key_is_stable_for_same_inputs():
    p = _paper()
    k1 = claim_cache.compute_key(p, model="M", reader_mode="fast", reader_max_candidates=5)
    k2 = claim_cache.compute_key(p, model="M", reader_mode="fast", reader_max_candidates=5)
    assert k1 == k2


def test_key_changes_with_model():
    p = _paper()
    a = claim_cache.compute_key(p, model="DeepSeek-V4", reader_mode="fast")
    b = claim_cache.compute_key(p, model="Kimi", reader_mode="fast")
    assert a != b


def test_key_changes_with_paper_content():
    a = claim_cache.compute_key(_paper(abstract="old"), model="M")
    b = claim_cache.compute_key(_paper(abstract="new"), model="M")
    assert a != b


def test_key_changes_with_reader_config():
    p = _paper()
    a = claim_cache.compute_key(p, model="M", reader_mode="fast", reader_max_candidates=5)
    b = claim_cache.compute_key(p, model="M", reader_mode="fast", reader_max_candidates=8)
    c = claim_cache.compute_key(p, model="M", reader_mode="deep", reader_max_candidates=5)
    assert len({a, b, c}) == 3


def test_key_identity_prefers_arxiv_base():
    # same arxiv_id_base + identical content -> same key even if paper_id differs
    common = dict(title="same title", year=2024, venue="arXiv", abstract="same abs", arxiv_id_base="2401.55555")
    p1 = Paper(paper_id="local-1", **common)
    p2 = Paper(paper_id="local-2", **common)
    assert claim_cache.compute_key(p1, model="M") == claim_cache.compute_key(p2, model="M")


# --- store / load round-trip --------------------------------------------------

def test_store_then_load_round_trips_claims(tmp_path):
    p = _paper()
    key = claim_cache.compute_key(p, model="M")
    original = _claims(n=3)
    assert claim_cache.store(tmp_path, key, original, paper=p, model="M") is True

    loaded = claim_cache.load(tmp_path, key)
    assert loaded is not None
    assert [c.model_dump() for c in loaded] == [c.model_dump() for c in original]


def test_load_miss_returns_none(tmp_path):
    assert claim_cache.load(tmp_path, "deadbeef") is None


def test_load_corrupt_file_is_fail_open(tmp_path):
    key = "ab" + "c" * 38
    path = claim_cache._path_for(tmp_path, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert claim_cache.load(tmp_path, key) is None


# --- store policy: non-empty + write-once ------------------------------------

def test_store_refuses_empty_claims(tmp_path):
    p = _paper()
    key = claim_cache.compute_key(p, model="M")
    assert claim_cache.store(tmp_path, key, [], paper=p, model="M") is False
    assert claim_cache.load(tmp_path, key) is None


def test_store_is_write_once(tmp_path):
    p = _paper()
    key = claim_cache.compute_key(p, model="M")
    first = _claims(n=1)
    assert claim_cache.store(tmp_path, key, first, paper=p, model="M") is True
    # a second store under the same key does not overwrite
    assert claim_cache.store(tmp_path, key, _claims(n=5), paper=p, model="M") is False
    loaded = claim_cache.load(tmp_path, key)
    assert loaded is not None and len(loaded) == 1


# --- enable / disable / dir resolution ---------------------------------------

def test_cache_disabled_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AIGRAPH_CLAIM_CACHE", "0")
    assert claim_cache.cache_enabled() is False
    assert claim_cache.cache_dir_for(tmp_path / "run" / "claims.jsonl") is None


def test_cache_dir_defaults_to_runs_root_sibling(monkeypatch, tmp_path):
    monkeypatch.delenv("AIGRAPH_CLAIM_CACHE", raising=False)
    monkeypatch.delenv("AIGRAPH_CLAIM_CACHE_DIR", raising=False)
    claims_output = tmp_path / "runs" / "run-1" / "claims.jsonl"
    # <runs_root>/_claim_cache  (sibling of the run dir, dashboard skips it)
    assert claim_cache.cache_dir_for(claims_output) == tmp_path / "runs" / "_claim_cache"


def test_cache_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv("AIGRAPH_CLAIM_CACHE", raising=False)
    monkeypatch.setenv("AIGRAPH_CLAIM_CACHE_DIR", str(tmp_path / "shared"))
    assert claim_cache.cache_dir_for(tmp_path / "x" / "claims.jsonl") == tmp_path / "shared"


def test_none_cache_dir_is_safe():
    assert claim_cache.load(None, "k") is None
    assert claim_cache.store(None, "k", _claims(), paper=_paper(), model="M") is False


# --- reuse simulation (mirrors server.run_one) -------------------------------

def test_second_run_reuses_cache_without_re_extracting(tmp_path):
    """First pass populates the cache; a second pass over the same papers under
    the same config hits the cache and never calls the (fake) extractor."""
    papers = [_paper(pid=f"2401.{i:05d}") for i in range(4)]
    calls = {"extract": 0}

    def fake_extract(paper):
        calls["extract"] += 1
        return _claims(pid=paper.paper_id, n=2)

    def pass_over():
        produced = 0
        for p in papers:
            key = claim_cache.compute_key(p, model="M", reader_mode="fast")
            cached = claim_cache.load(tmp_path, key)
            if cached is None:
                cs = fake_extract(p)
                claim_cache.store(tmp_path, key, cs, paper=p, model="M", reader_mode="fast")
                cached = cs
            produced += len(cached)
        return produced

    first = pass_over()
    assert calls["extract"] == 4           # cold: every paper extracted
    second = pass_over()
    assert calls["extract"] == 4           # warm: no new extractions
    assert first == second == 8


def test_stats_counts_entries(tmp_path):
    for i in range(3):
        p = _paper(pid=f"2401.{i:05d}")
        key = claim_cache.compute_key(p, model="M")
        claim_cache.store(tmp_path, key, _claims(pid=p.paper_id), paper=p, model="M")
    s = claim_cache.stats(tmp_path)
    assert s["entries"] == 3 and s["enabled"] is True


def test_stored_payload_has_provenance_fields(tmp_path):
    p = _paper(pid="2401.99999", arxiv_base="2401.99999")
    key = claim_cache.compute_key(p, model="DeepSeek-V4", reader_mode="fast")
    claim_cache.store(tmp_path, key, _claims(pid=p.paper_id), paper=p, model="DeepSeek-V4", reader_mode="fast")
    payload = json.loads(claim_cache._path_for(tmp_path, key).read_text(encoding="utf-8"))
    assert payload["schema"] == claim_cache.CACHE_SCHEMA_VERSION
    assert payload["paper_id"] == "2401.99999"
    assert payload["model"] == "DeepSeek-V4"
    assert payload["claim_count"] == 2
