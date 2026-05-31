"""Topic→hypothesis relevance helpers.

The default per-query filter in ``aigraph_query.py`` uses plain
bag-of-words token overlap: count how many query tokens appear in the
hypothesis text. That misses tangential matches ("tree search" vs
"MCTS"; "uncertainty quantification" vs "calibration") and over-weights
common words.

This module adds a TF-IDF cosine variant: ``tfidf_topic_scores`` fits a
small vectorizer over (hypothesis texts + topic) — so the topic's terms
share the vocabulary — and returns cosine similarity per hypothesis.
IDF down-weights common terms; 1–2 grams catch phrases. It is the
*cheap* middle step between bag-of-words and true embeddings:
scikit-learn is already a dependency, no model download, milliseconds
per query. Swap for sentence-transformers when warranted.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def hypothesis_haystack(hyp, anomaly=None, claims=None) -> str:
    """Concatenate every textual field of a hypothesis (plus parent
    anomaly + cited claims) into one bag-of-text. Mirrors what
    ``_topic_relevance`` walks in ``aigraph_query.py`` so both the
    bag-of-words and TF-IDF paths see the same haystack — keeps the
    two relevance signals comparable.
    """
    parts: list[str] = [
        hyp.hypothesis or "",
        hyp.mechanism or "",
        " ".join(hyp.predictions or []),
        hyp.evidence_gap or "",
    ]
    if anomaly is not None:
        parts.append(anomaly.central_question or "")
        parts.extend(str(v) for v in (anomaly.shared_entities or {}).values())
    if claims:
        for c in claims:
            parts.append(c.claim_text or "")
            for field in ("method", "task", "dataset", "metric"):
                v = getattr(c, field, None)
                if v:
                    parts.append(str(v))
    return " ".join(parts)


def tfidf_topic_scores(hyp_texts: list[str], topic: str) -> list[float]:
    """Cosine(topic_tfidf, hyp_tfidf) for each ``hyp_texts[i]``, in input order.

    Returns a list of zeros (length matching ``hyp_texts``) when either
    side is empty or when the vocabulary collapses to nothing (e.g. an
    all-stopword topic).

    One ``TfidfVectorizer.fit_transform`` per call over the combined
    (corpus + topic) text — ensures the topic's tokens are in the
    vocabulary so the cosine isn't trivially zero for a one-shot query.
    """
    if not hyp_texts or not (topic or "").strip():
        return [0.0] * len(hyp_texts)
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )
    try:
        matrix = vec.fit_transform(hyp_texts + [topic])
    except ValueError:
        return [0.0] * len(hyp_texts)
    sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
    return [float(s) for s in sims]
