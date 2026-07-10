"""Sequence models for delayed-outcome prediction and credit (requires torch).

Importing this package requires the ``research`` extra; install it with
``uv sync --extra research``.
"""

from longfeedback.models.docm import (
    DelayedOutcomeCreditModel,
    DocmLossWeights,
    SequenceDataset,
    TrainingSettings,
)
from longfeedback.models.encoders import CausalTransformerEncoder, EncoderArchitecture
from longfeedback.models.outcome import (
    VARIANT_LOSS_WEIGHTS,
    redistributed_rewards,
    variant_loss_weights,
)

__all__ = [
    "VARIANT_LOSS_WEIGHTS",
    "CausalTransformerEncoder",
    "DelayedOutcomeCreditModel",
    "DocmLossWeights",
    "EncoderArchitecture",
    "SequenceDataset",
    "TrainingSettings",
    "redistributed_rewards",
    "variant_loss_weights",
]
