"""Bootstrap ensembles of DOCM models for epistemic uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from longfeedback.models.docm import (
    DelayedOutcomeCreditModel,
    DocmLossWeights,
    SequenceDataset,
    TrainingSettings,
)
from longfeedback.models.encoders import EncoderArchitecture

FloatArray = npt.NDArray[np.float64]


@dataclass
class BootstrapEnsemble:
    """K capacity-matched members trained on bootstrap-resampled episodes.

    Each member gets an independent initialization seed and an independent
    with-replacement resample of the training episodes. Predictions report
    the member mean and the between-member standard deviation (epistemic
    uncertainty), following the design-doc MVP over evidential alternatives.
    """

    observation_dim: int
    n_actions: int
    horizon: int
    reference_action: int = 0
    architecture: EncoderArchitecture = field(default_factory=EncoderArchitecture)
    loss_weights: DocmLossWeights = field(default_factory=DocmLossWeights)
    members: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.members < 2:
            raise ValueError("an ensemble needs at least two members")
        self._models = [
            DelayedOutcomeCreditModel(
                observation_dim=self.observation_dim,
                n_actions=self.n_actions,
                horizon=self.horizon,
                reference_action=self.reference_action,
                architecture=self.architecture,
                loss_weights=self.loss_weights,
                seed=self.seed + member_index,
            )
            for member_index in range(self.members)
        ]

    def parameter_count(self) -> int:
        """Parameters per member; capacity comparisons stay per-architecture."""

        return self._models[0].parameter_count()

    def fit(
        self,
        dataset: SequenceDataset,
        *,
        training: TrainingSettings | None = None,
    ) -> dict[str, float]:
        last_summary: dict[str, float] = {}
        for member_index, model in enumerate(self._models):
            resampler = np.random.default_rng(self.seed + 1_000 * member_index)
            indices = np.sort(
                resampler.integers(0, dataset.episodes, size=dataset.episodes)
            ).astype(np.int64)
            last_summary = model.fit(dataset.subset(indices), training=training)
        return last_summary

    def predict_outcome_probability(
        self, dataset: SequenceDataset
    ) -> tuple[FloatArray, FloatArray]:
        stacked = np.stack([model.predict_outcome_probability(dataset) for model in self._models])
        return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)

    def predict_logged_credit(self, dataset: SequenceDataset) -> tuple[FloatArray, FloatArray]:
        stacked = np.stack([model.predict_logged_credit(dataset) for model in self._models])
        return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)
