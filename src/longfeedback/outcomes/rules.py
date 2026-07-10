"""High-precision rule labelers for future-user feedback (stage 1 of §6.5).

These deterministic rules are the first stage of the design doc's label
pipeline; LLM-assisted labeling and human validation come later. Labels are
behavioral proxies, not ground-truth satisfaction, and rule outputs are
frozen before any train/test modeling. Rules read future user messages; model
inputs must never include them.
"""

from __future__ import annotations

import re

LABELER_VERSION = "rules-v2"

_NEGATIVE_PATTERN = re.compile(
    r"(?ix)"
    r"\b(?:"
    r"(?:that(?:'s|\s+is)\s+)?(?:wrong|incorrect)"
    r"|not\s+(?:right|correct|true|what\s+i\s+(?:asked|wanted|meant))"
    r"|(?:doesn|didn|don)'?t\s+work"
    r"|not\s+working"
    r"|(?:still|same)\s+(?:doesn'?t|not|the\s+same)"
    r"|you\s+(?:didn'?t|failed|misunderstood|are\s+wrong|got\s+it\s+wrong)"
    r"|i\s+(?:said|asked\s+for|already\s+(?:said|told))"
    r"|try\s+again"
    r"|(?:it|this|that)\s+fail(?:ed|s)"
    r"|(?:i\s+get|throws?|gives?\s+me)\s+(?:an?\s+)?error"
    r")"
    r"|^\s*no[,.!]"
)

_POSITIVE_PATTERN = re.compile(
    r"(?ix)"
    r"\b(?:"
    r"thanks?|thank\s+you"
    r"|perfect|excellent|awesome|brilliant"
    r"|(?<!how\s)(?<!why\s)(?:that|it)\s+work(?:s|ed)"
    r"|works\s+(?:now|great|perfectly)"
    r"|exactly\s+what\s+i"
    r"|great\s+(?:answer|job|work)"
    r"|very\s+helpful"
    r")"
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def negative_signal(text: str) -> bool:
    """True when the message contains an explicit correction/failure signal."""

    return _NEGATIVE_PATTERN.search(text) is not None


def positive_signal(text: str) -> bool:
    """True when the message contains an explicit satisfaction signal."""

    return _POSITIVE_PATTERN.search(text) is not None


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(text.lower()))


def is_repetition(
    current: str,
    reference: str,
    *,
    threshold: float = 0.6,
    min_tokens: int = 5,
) -> bool:
    """True when a later user message semantically repeats an earlier request.

    Token-set Jaccard similarity is a deliberately conservative stand-in for
    embedding similarity; short messages are never counted as repetitions.
    """

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    current_tokens = _tokens(current)
    reference_tokens = _tokens(reference)
    if len(current_tokens) < min_tokens or len(reference_tokens) < min_tokens:
        return False
    union = current_tokens | reference_tokens
    if not union:
        return False
    return len(current_tokens & reference_tokens) / len(union) >= threshold
