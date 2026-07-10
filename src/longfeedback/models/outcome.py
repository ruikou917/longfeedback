"""Capacity-matched DOCM variant presets and redistribution helpers."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt

from longfeedback.credit.rudder import redistribute_prefix_values
from longfeedback.models.docm import DocmLossWeights

FloatArray = npt.NDArray[np.float64]

VARIANT_LOSS_WEIGHTS: Final = MappingProxyType(
    {
        "docm_outcome": DocmLossWeights(outcome=1.0),
        "docm_prefix": DocmLossWeights(outcome=1.0, prefix=1.0, telescoping=0.25),
        "docm_credit": DocmLossWeights(outcome=1.0, prefix=1.0, telescoping=0.25, credit=1.0),
    }
)


def variant_loss_weights(name: str) -> DocmLossWeights:
    try:
        return VARIANT_LOSS_WEIGHTS[name]
    except KeyError as error:
        known = ", ".join(sorted(VARIANT_LOSS_WEIGHTS))
        raise ValueError(f"unknown DOCM variant {name!r}; expected one of: {known}") from error


def redistributed_rewards(prefix_values: FloatArray) -> FloatArray:
    """RUDDER-style per-step rewards from prefix values ``V_0..V_T``."""

    return np.asarray(redistribute_prefix_values(prefix_values), dtype=np.float64)
