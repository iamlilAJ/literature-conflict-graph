"""Tests for #50 run-local controlled taxonomy post-pass."""
import aigraph.canonicalize_local as cl
from aigraph.models import Claim


def _claim(cid, method, canon_method, task="t", canon_task="other"):
    return Claim(claim_id=cid, paper_id="p", claim_text="x", method=method,
                 canonical_method=canon_method, task=task, canonical_task=canon_task)


# ---- apply (the conservative re-map) -------------------------------------

def test_only_remaps_other_never_overwrites_good():
    claims = [
        _claim("c1", "on-policy knowledge distillation", "other"),
        _claim("c2", "chain of thought prompting", "chain-of-thought"),  # already good
    ]
    method_remap = {"on-policy knowledge distillation": "on-policy-distillation",
                    "chain of thought prompting": "SHOULD-NOT-APPLY"}
    out = cl.apply_local_canonicalization(claims, method_remap, {})
    assert out[0].canonical_method == "on-policy-distillation"   # 'other' -> bucket
    assert out[1].canonical_method == "chain-of-thought"         # good value untouched


def test_remap_matches_normalized_raw():
    claims = [_claim("c1", "  On-Policy   Distillation ", "other")]
    out = cl.apply_local_canonicalization(claims, {"on-policy distillation": "opd"}, {})
    assert out[0].canonical_method == "opd"


def test_no_raw_method_stays_other():
    # canonical=other because raw method was null -> nothing to match, stays other
    claims = [_claim("c1", None, "other")]
    out = cl.apply_local_canonicalization(claims, {"x": "y"}, {})
    assert cl._is_other(out[0].canonical_method)


def test_empty_remaps_is_noop_identity():
    claims = [_claim("c1", "m", "other")]
    out = cl.apply_local_canonicalization(claims, {}, {})
    assert out is claims  # identity short-circuit


def test_task_remap_independent():
    claims = [_claim("c1", "m", "fine-tuning", task="open-ended generation", canon_task="other")]
    out = cl.apply_local_canonicalization(claims, {}, {"open-ended generation": "generation"})
    assert out[0].canonical_method == "fine-tuning"   # untouched
    assert out[0].canonical_task == "generation"


# ---- derive (gating + parsing) -------------------------------------------

def test_derive_disabled_without_key(monkeypatch):
    monkeypatch.setattr(cl, "configured_api_key", lambda *a, **k: None)
    claims = [_claim(f"c{i}", "m", "other") for i in range(5)]
    assert cl.derive_local_taxonomy(claims) == ({}, {})


def test_derive_skips_when_few_other(monkeypatch):
    monkeypatch.setattr(cl, "configured_api_key", lambda *a, **k: "k")
    # only 2 'other' (< _MIN_OTHER=3) -> skip, no LLM
    claims = [_claim("c1", "m1", "other"), _claim("c2", "m2", "other"),
              _claim("c3", "m3", "fine-tuning")]
    assert cl.derive_local_taxonomy(claims) == ({}, {})


def test_derive_parses_llm_remap(monkeypatch):
    monkeypatch.setattr(cl, "configured_api_key", lambda *a, **k: "k")
    monkeypatch.setattr(cl, "build_openai_client", lambda *a, **k: object())
    monkeypatch.setattr(cl, "configured_model", lambda *a, **k: "m")
    monkeypatch.setattr(cl, "call_llm_text", lambda *a, **k:
                        '{"method_remap": {"OPD": "on-policy-distillation", "x": "other"}, "task_remap": {}}')
    claims = [_claim(f"c{i}", "OPD", "other") for i in range(3)]
    mr, tr = cl.derive_local_taxonomy(claims)
    assert mr == {"opd": "on-policy-distillation"}  # 'other' bucket dropped, keys normalized
    assert tr == {}


def test_derive_failopen_on_llm_error(monkeypatch):
    monkeypatch.setattr(cl, "configured_api_key", lambda *a, **k: "k")
    monkeypatch.setattr(cl, "build_openai_client", lambda *a, **k: object())
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(cl, "call_llm_text", boom)
    claims = [_claim(f"c{i}", "m", "other") for i in range(4)]
    assert cl.derive_local_taxonomy(claims) == ({}, {})


def test_canonicalize_claims_failopen(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(cl, "derive_local_taxonomy", boom)
    claims = [_claim("c1", "m", "other")]
    assert cl.canonicalize_claims(claims) is claims
