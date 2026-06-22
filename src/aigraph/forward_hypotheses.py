"""Back-compat shim.

The forward-design generation contract was folded into the (now §7-thawed)
frozen generator `llm_hypotheses.LLMHypothesisGenerator` — see §7 Thaw #4 in
docs/v0.7-pipeline-freeze.md. This module previously held a separate
`ForwardDesignGenerator` used as the treatment arm of the blind A/B
(scripts/hyp_ab_generators.py). Post-thaw the frozen generator IS the forward
generator, so `ForwardDesignGenerator` is just an alias and
`forward_generator_enabled()` is retained for any callers that imported it.

To reproduce the pre-thaw *control* generator for a fresh A/B, check out the
`v0.7-frozen` git tag (its `llm_hypotheses.py` carries the retrospective
contract).
"""
from __future__ import annotations

import os

from .llm_hypotheses import LLMHypothesisGenerator as ForwardDesignGenerator

__all__ = ["ForwardDesignGenerator", "forward_generator_enabled"]


def forward_generator_enabled() -> bool:
    """Retained for back-compat. The forward contract is now the default
    generator (§7 Thaw #4), so this reflects an explicit opt-out rather than an
    opt-in: it returns True unless AIGRAPH_FORWARD_GENERATOR is set to a falsey
    value."""
    val = os.environ.get("AIGRAPH_FORWARD_GENERATOR", "1").strip().lower()
    return val not in {"0", "false", "no", "off"}
