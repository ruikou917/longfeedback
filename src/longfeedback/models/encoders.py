"""Small CPU-friendly causal sequence encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class EncoderArchitecture:
    """Capacity-defining settings shared by every capacity-matched variant."""

    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    feedforward_multiple: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("d_model, n_layers, and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CausalTransformerEncoder(nn.Module):
    """A causal Transformer over pre-built token features.

    Position ``t`` of the output depends only on tokens ``0..t``; the unit
    tests assert this causality property directly.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        max_length: int,
        architecture: EncoderArchitecture,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or max_length <= 0:
            raise ValueError("input_dim and max_length must be positive")
        self.max_length = max_length
        self.input_projection = nn.Linear(input_dim, architecture.d_model)
        self.position_embedding = nn.Embedding(max_length, architecture.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=architecture.d_model,
            nhead=architecture.n_heads,
            dim_feedforward=architecture.d_model * architecture.feedforward_multiple,
            dropout=architecture.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=architecture.n_layers,
            enable_nested_tensor=False,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [batch, length, features], got {tuple(tokens.shape)}")
        length = tokens.shape[1]
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length {self.max_length}")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.input_projection(tokens) + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length, device=tokens.device)
        encoded: Tensor = self.transformer(hidden, mask=causal_mask, is_causal=True)
        return encoded
