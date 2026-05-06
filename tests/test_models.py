"""Tests for pydantic model behavior.

Notably exercises Hypothesis.extra='allow' which preserves the
``novelty_check`` extras key written by the check-novelty CLI and
read by predict-influence (Bug #1, see PR #14 caveats).
"""

from aigraph.models import Hypothesis


def test_hypothesis_preserves_extras_through_roundtrip():
    """Verify Hypothesis with extras serializes + deserializes cleanly.

    Mirrors the production flow: check-novelty CLI emits a JSONL line
    with novelty_check baked into the dict; predict-influence loads via
    model_validate_json. Extras must survive that round-trip.
    """
    payload = {
        "hypothesis_id": "h001",
        "anomaly_id": "a001",
        "hypothesis": "test",
        "novelty_check": {
            "is_novel": True,
            "similar_papers": [{"arxiv_id": "2403.12345", "title": "test"}],
            "rationale": "no close match found",
        },
    }

    h = Hypothesis.model_validate(payload)
    json_str = h.model_dump_json()

    assert "novelty_check" in json_str
    assert "is_novel" in json_str

    h2 = Hypothesis.model_validate_json(json_str)

    assert getattr(h2, "novelty_check", None) is not None
    assert h2.novelty_check["is_novel"] is True


def test_hypothesis_allows_unknown_field_at_init():
    """Constructing Hypothesis with an unknown field should preserve it,
    not silently drop it (extra='allow' behavior)."""
    h = Hypothesis(
        hypothesis_id="h001",
        anomaly_id="a001",
        hypothesis="test",
        novelty_check={"is_novel": False},
    )
    assert getattr(h, "novelty_check", None) == {"is_novel": False}
