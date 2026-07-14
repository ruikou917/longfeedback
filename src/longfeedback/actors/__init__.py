"""Candidate-scoring actor policies for E11/E12.

This package stays torch-free at import time; the trainable and LLM policies
live in :mod:`longfeedback.actors.trainable` and
:mod:`longfeedback.actors.llm_candidates` and are imported directly by the
research-extra experiment runners.
"""

from longfeedback.actors.base import (
    CandidatePolicy,
    PolicyDecision,
    PolicyScores,
    RenderedPrompt,
    canonical_candidates,
    render_prompt,
)
from longfeedback.actors.mock import MockCandidatePolicy

__all__ = [
    "CandidatePolicy",
    "MockCandidatePolicy",
    "PolicyDecision",
    "PolicyScores",
    "RenderedPrompt",
    "canonical_candidates",
    "render_prompt",
]
