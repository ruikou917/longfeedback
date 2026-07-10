"""Tests for deterministic run manifests."""

from __future__ import annotations

from longfeedback.experiments.manifest import canonical_json, sha256_json


def test_canonical_json_and_hash_ignore_mapping_order() -> None:
    left = {"b": 2, "a": {"d": 4, "c": 3}}
    right = {"a": {"c": 3, "d": 4}, "b": 2}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_json(left) == sha256_json(right)
