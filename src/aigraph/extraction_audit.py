"""Non-frozen post-parse audit of claim-direction bias (#49).

The frozen extractor (`llm_extract.py::_normalize_claim_dict`) defaults a
null/missing ``direction`` to ``"positive"`` *before* the :class:`Claim` is
built, so the fact that a value was *defaulted* (vs. genuinely positive) is not
recoverable from the finished claim. This module therefore reports a
**corpus-level** bias signal — the direction distribution and a
``positive_bias_suspected`` flag — rather than per-claim defaulted detection.

A corpus with many claims but *zero* negative/mixed directions is the symptom
the conflict detector cares about: it has nothing to put on the other side of a
disagreement, so anomalies collapse. The per-claim, in-place fix (preserve the
raw direction / stop defaulting to positive) requires a thaw of
``llm_extract.py`` and is intentionally out of scope here.
"""

from __future__ import annotations

from typing import Any

from .models import Claim

# A corpus this size with no opposition is a meaningful bias signal (below it,
# an all-positive run is just as likely to be a genuinely-uncontested topic).
_MIN_CLAIMS_FOR_BIAS = 8


def audit_direction_bias(claims: list[Claim]) -> dict[str, Any]:
    """Return a corpus-level direction-bias audit record.

    ``positive_bias_suspected`` is True when there are enough claims to judge and
    *none* of them carry a non-positive direction — the pattern that starves the
    conflict detector and is the tell-tale of the frozen positive-default.
    """
    n = len(claims)
    distribution: dict[str, int] = {}
    for c in claims:
        d = str(getattr(c, "direction", None) or "unknown").strip().lower() or "unknown"
        distribution[d] = distribution.get(d, 0) + 1
    positive = distribution.get("positive", 0)
    non_positive = n - positive
    suspected = n >= _MIN_CLAIMS_FOR_BIAS and non_positive == 0
    return {
        "event": "direction_audit",
        "claims_total": n,
        "direction_distribution": distribution,
        "positive_fraction": round(positive / n, 3) if n else 0.0,
        "non_positive_count": non_positive,
        "positive_bias_suspected": suspected,
        "note": (
            "corpus-level only; per-claim defaulted-direction detection requires "
            "an llm_extract.py thaw"
        ),
    }
