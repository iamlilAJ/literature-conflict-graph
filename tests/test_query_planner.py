"""Unit tests for the LLM query planner (src/aigraph/query_planner.py).

The planner is non-frozen and sits in front of the frozen lexical query layer
as the mandatory normalize step. These tests pin its fail-open contract (no key
/ disabled / no client / LLM error / unparseable reply all pass through with the
raw topic), its defensive JSON parsing, and the candidate-query ordering that
guarantees the raw topic is always a never-worse-than-baseline fallback — all
without a real LLM (the gateway call is injected).
"""
import aigraph.query_planner as qp


def _enable(monkeypatch):
    """Enable the planner with a fake key so plan_query reaches the LLM path."""
    monkeypatch.setenv("AIGRAPH_QUERY_PLANNER", "1")
    monkeypatch.setattr(qp, "configured_api_key", lambda *_a, **_k: "sk-test")
    monkeypatch.setattr(qp, "configured_model", lambda *_a, **_k: "m")
    monkeypatch.setattr(qp, "build_openai_client", lambda *_a, **_k: object())


def _reply(monkeypatch, text):
    monkeypatch.setattr(qp, "call_llm_text", lambda *_a, **_k: text)


# --------------------------------------------------------------------------- #
# fail-open contract
# --------------------------------------------------------------------------- #

def test_disabled_env_is_passthrough(monkeypatch):
    monkeypatch.setenv("AIGRAPH_QUERY_PLANNER", "0")
    monkeypatch.setattr(qp, "configured_api_key", lambda *_a, **_k: "sk-test")
    p = qp.plan_query("graph of thoguhts")
    assert p["applied"] is False
    assert p["reason"] == "disabled"
    assert p["query"] == "graph of thoguhts"  # whitespace-collapsed raw


def test_no_key_is_passthrough(monkeypatch):
    monkeypatch.setenv("AIGRAPH_QUERY_PLANNER", "1")
    monkeypatch.setattr(qp, "configured_api_key", lambda *_a, **_k: None)
    p = qp.plan_query("tree of thoughts")
    assert p["applied"] is False
    assert p["query"] == "tree of thoughts"


def test_empty_topic_is_passthrough(monkeypatch):
    _enable(monkeypatch)
    p = qp.plan_query("   ")
    assert p["applied"] is False
    assert p["reason"] == "empty"
    assert p["query"] == ""


def test_client_build_error_is_passthrough(monkeypatch):
    monkeypatch.setenv("AIGRAPH_QUERY_PLANNER", "1")
    monkeypatch.setattr(qp, "configured_api_key", lambda *_a, **_k: "sk-test")

    def _boom(*_a, **_k):
        raise RuntimeError("no network")

    monkeypatch.setattr(qp, "build_openai_client", _boom)
    p = qp.plan_query("retrieval augmented generation")
    assert p["applied"] is False
    assert p["reason"] == "no_client"


def test_llm_error_is_passthrough(monkeypatch):
    _enable(monkeypatch)

    def _boom(*_a, **_k):
        raise RuntimeError("gateway 500")

    monkeypatch.setattr(qp, "call_llm_text", _boom)
    p = qp.plan_query("mixture of experts routing")
    assert p["applied"] is False
    assert p["reason"] == "llm_error"
    assert p["query"] == "mixture of experts routing"


def test_unparseable_reply_is_passthrough(monkeypatch):
    _enable(monkeypatch)
    _reply(monkeypatch, "I cannot help with that.")
    p = qp.plan_query("chain of thought")
    assert p["applied"] is False
    assert p["reason"] == "unparseable"


def test_reply_missing_query_is_passthrough(monkeypatch):
    _enable(monkeypatch)
    _reply(monkeypatch, '{"alternates": ["x"], "is_niche": true}')
    p = qp.plan_query("chain of thought")
    assert p["applied"] is False  # no usable query -> pass through


# --------------------------------------------------------------------------- #
# happy path + defensive parsing
# --------------------------------------------------------------------------- #

def test_clean_json_is_applied(monkeypatch):
    _enable(monkeypatch)
    _reply(monkeypatch, '{"query": "graph-of-thoughts prompting LLM reasoning", '
                        '"canonical_terms": ["graph-of-thoughts"], '
                        '"domain_anchors": ["prompting", "LLM", "reasoning"], '
                        '"alternates": ["tree-of-thoughts reasoning"], "is_niche": false}')
    p = qp.plan_query("graph of thoguhts")
    assert p["applied"] is True
    assert p["query"] == "graph-of-thoughts prompting LLM reasoning"
    assert p["canonical_terms"] == ["graph-of-thoughts"]
    assert p["domain_anchors"] == ["prompting", "LLM", "reasoning"]
    assert p["alternates"] == ["tree-of-thoughts reasoning"]
    assert p["is_niche"] is False
    assert p["raw"] == "graph of thoguhts"


def test_json_wrapped_in_prose_and_fences(monkeypatch):
    _enable(monkeypatch)
    _reply(monkeypatch, 'Sure!```json\n{"query": "MCTS arithmetic reasoning"}\n```trailing')
    p = qp.plan_query("tree search for math")
    assert p["applied"] is True
    assert p["query"] == "MCTS arithmetic reasoning"


def test_alternates_capped_at_two(monkeypatch):
    _enable(monkeypatch)
    _reply(monkeypatch, '{"query": "q", "alternates": ["a", "b", "c", "d"]}')
    p = qp.plan_query("topic")
    assert p["alternates"] == ["a", "b"]


def test_str_list_dedups_case_insensitively(monkeypatch):
    _enable(monkeypatch)
    _reply(monkeypatch, '{"query": "q", "domain_anchors": ["LLM", "llm", "reasoning"]}')
    p = qp.plan_query("topic")
    assert p["domain_anchors"] == ["LLM", "reasoning"]


# --------------------------------------------------------------------------- #
# candidate ordering — raw is always the never-worse-than-baseline backstop
# --------------------------------------------------------------------------- #

def test_candidate_queries_order_and_backstop():
    plan = {"query": "graph-of-thoughts prompting LLM reasoning",
            "alternates": ["tree-of-thoughts reasoning"],
            "raw": "graph of thoguhts"}
    cands = qp.candidate_queries(plan)
    assert cands == ["graph-of-thoughts prompting LLM reasoning",
                     "tree-of-thoughts reasoning",
                     "graph of thoguhts"]


def test_candidate_queries_dedup_when_passthrough():
    # passthrough plan: query == raw, no alternates -> single candidate
    plan = {"query": "graph of thoughts", "alternates": [], "raw": "graph of thoughts"}
    assert qp.candidate_queries(plan) == ["graph of thoughts"]


def test_candidate_queries_case_insensitive_dedup():
    plan = {"query": "Graph Reasoning", "alternates": ["graph reasoning"],
            "raw": "GRAPH REASONING"}
    assert qp.candidate_queries(plan) == ["Graph Reasoning"]
