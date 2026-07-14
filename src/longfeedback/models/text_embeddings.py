"""Frozen text-embedding providers for semantic states and candidate actions.

The critic's text encoder is deliberately separate from the E12 actor, so
critic inputs cannot drift when the actor's LoRA weights change. The hashed
embedder is the deterministic CPU default; a pinned neural embedding model can
implement the same protocol for full runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from longfeedback.environments.base import normalize_text

FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class TextEmbeddingProvider(Protocol):
    @property
    def embedding_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> FloatArray: ...


@dataclass
class HashedTextEmbedder:
    """Deterministic signed feature-hashing embedder (words + char trigrams).

    Semantically identical canonical texts map to identical vectors on every
    platform because bucketing uses SHA-256, not Python's randomized ``hash``.
    Embeddings are cached by normalized content.
    """

    dim: int = 48
    _cache: dict[str, FloatArray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")

    @property
    def embedding_id(self) -> str:
        return f"hashed-ngram-v1:dim={self.dim}"

    @property
    def dimension(self) -> int:
        return self.dim

    def embed(self, text: str) -> FloatArray:
        normalized = normalize_text(text)
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        vector = np.zeros(self.dim, dtype=np.float64)
        features = normalized.split()
        padded = f"##{normalized}##"
        features.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        vector.setflags(write=False)
        self._cache[normalized] = vector
        return vector

    def embed_batch(self, texts: list[str]) -> FloatArray:
        return np.stack([self.embed(text) for text in texts], axis=0)
