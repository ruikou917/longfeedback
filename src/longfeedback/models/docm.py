"""Delayed Outcome Credit Model (DOCM) MVP with capacity-matched variants.

One architecture carries three heads (terminal outcome, prefix value, action
value). Variants differ only in loss weights, so every comparison between
outcome-only, RUDDER-style prefix, and credit-supervised training is
capacity-matched by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from longfeedback.models.encoders import CausalTransformerEncoder, EncoderArchitecture

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_SE_EPSILON = 1.0e-4


@dataclass(frozen=True, slots=True)
class DocmLossWeights:
    """Loss switches; zero weight disables a head's supervision entirely."""

    outcome: float = 1.0
    prefix: float = 0.0
    telescoping: float = 0.0
    credit: float = 0.0

    def __post_init__(self) -> None:
        for name in ("outcome", "prefix", "telescoping", "credit"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} weight cannot be negative")
        if self.outcome + self.prefix + self.telescoping + self.credit <= 0.0:
            raise ValueError("at least one loss weight must be positive")


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    epochs: int = 60
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0 or self.grad_clip <= 0.0:
            raise ValueError("weight_decay must be >= 0 and grad_clip > 0")


@dataclass(frozen=True)
class SequenceDataset:
    """Fixed-horizon logged episodes with optional oracle-credit labels."""

    observations: FloatArray
    actions: IntArray
    responses: FloatArray
    outcomes: FloatArray
    credit_targets: FloatArray | None = None
    credit_mask: BoolArray | None = None
    credit_se: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.observations.ndim != 3:
            raise ValueError("observations must be [episodes, horizon, features]")
        episodes, horizon, _ = self.observations.shape
        if self.actions.shape != (episodes, horizon):
            raise ValueError("actions must be [episodes, horizon]")
        if self.responses.shape != (episodes, horizon):
            raise ValueError("responses must be [episodes, horizon]")
        if self.outcomes.shape != (episodes,):
            raise ValueError("outcomes must be [episodes]")
        labelled = (self.credit_targets, self.credit_mask, self.credit_se)
        if any(item is not None for item in labelled) and any(item is None for item in labelled):
            raise ValueError("credit targets, mask, and se must be provided together")
        for name in ("credit_targets", "credit_mask", "credit_se"):
            value = getattr(self, name)
            if value is not None and value.shape != (episodes, horizon):
                raise ValueError(f"{name} must be [episodes, horizon]")
        arrays: list[npt.NDArray[Any]] = [self.observations, self.responses, self.outcomes]
        if self.credit_targets is not None and self.credit_mask is not None:
            arrays.append(self.credit_targets[self.credit_mask])
        for array in arrays:
            if array.size and not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
                raise ValueError("dataset arrays must contain only finite values")

    @property
    def episodes(self) -> int:
        return int(self.observations.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[1])

    def subset(self, indices: IntArray) -> SequenceDataset:
        return SequenceDataset(
            observations=self.observations[indices],
            actions=self.actions[indices],
            responses=self.responses[indices],
            outcomes=self.outcomes[indices],
            credit_targets=None if self.credit_targets is None else self.credit_targets[indices],
            credit_mask=None if self.credit_mask is None else self.credit_mask[indices],
            credit_se=None if self.credit_se is None else self.credit_se[indices],
        )


class _DocmNet(nn.Module):
    """Interleaved observation/action tokens with three prediction heads.

    Token layout per episode: ``obs_0, act_0, obs_1, act_1, ...``. The hidden
    state at obs-token ``t`` encodes the history *before* action ``t``
    (including observation ``t``), so prefix values and action values read
    obs-token positions and the terminal outcome reads the final act token.
    """

    def __init__(
        self,
        *,
        observation_dim: int,
        n_actions: int,
        horizon: int,
        architecture: EncoderArchitecture,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or n_actions <= 1 or horizon <= 0:
            raise ValueError("observation_dim, n_actions, and horizon must be positive")
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.horizon = horizon
        token_dim = 2 + observation_dim + n_actions + 1
        self.encoder = CausalTransformerEncoder(
            input_dim=token_dim,
            max_length=2 * horizon,
            architecture=architecture,
        )
        self.outcome_head = nn.Linear(architecture.d_model, 1)
        self.prefix_head = nn.Linear(architecture.d_model, 1)
        self.action_value_head = nn.Linear(architecture.d_model, n_actions)

    def build_tokens(self, observations: Tensor, actions: Tensor, responses: Tensor) -> Tensor:
        batch, horizon, observation_dim = observations.shape
        token_dim = 2 + observation_dim + self.n_actions + 1
        tokens = torch.zeros(batch, 2 * horizon, token_dim, dtype=observations.dtype)
        tokens[:, 0::2, 0] = 1.0
        tokens[:, 0::2, 2 : 2 + observation_dim] = observations
        tokens[:, 1::2, 1] = 1.0
        action_onehot = nn.functional.one_hot(actions, num_classes=self.n_actions)
        tokens[:, 1::2, 2 + observation_dim : 2 + observation_dim + self.n_actions] = (
            action_onehot.to(observations.dtype)
        )
        tokens[:, 1::2, -1] = responses
        return tokens

    def forward(
        self, observations: Tensor, actions: Tensor, responses: Tensor
    ) -> dict[str, Tensor]:
        encoded = self.encoder(self.build_tokens(observations, actions, responses))
        obs_hidden = encoded[:, 0::2, :]
        final_hidden = encoded[:, -1:, :]
        prefix_logits = torch.cat(
            (self.prefix_head(obs_hidden), self.prefix_head(final_hidden)), dim=1
        ).squeeze(-1)
        outcome_prefix_logits = torch.cat(
            (self.outcome_head(obs_hidden), self.outcome_head(final_hidden)), dim=1
        ).squeeze(-1)
        return {
            "outcome_logit": outcome_prefix_logits[:, -1],
            "prefix_logits": prefix_logits,
            "outcome_prefix_logits": outcome_prefix_logits,
            "action_values": self.action_value_head(obs_hidden),
        }


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


@dataclass
class DelayedOutcomeCreditModel:
    """Numpy-facing wrapper that trains and evaluates one DOCM variant."""

    observation_dim: int
    n_actions: int
    horizon: int
    reference_action: int = 0
    architecture: EncoderArchitecture = field(default_factory=EncoderArchitecture)
    loss_weights: DocmLossWeights = field(default_factory=DocmLossWeights)
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.reference_action < self.n_actions:
            raise ValueError("reference_action must index the action space")
        _seed_torch(self.seed)
        self._net = _DocmNet(
            observation_dim=self.observation_dim,
            n_actions=self.n_actions,
            horizon=self.horizon,
            architecture=self.architecture,
        )
        self._feature_mean = np.zeros(self.observation_dim, dtype=np.float64)
        self._feature_scale = np.ones(self.observation_dim, dtype=np.float64)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._net.parameters())

    def _tensors(self, dataset: SequenceDataset) -> tuple[Tensor, Tensor, Tensor]:
        normalized = (dataset.observations - self._feature_mean) / self._feature_scale
        return (
            torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(dataset.actions, dtype=np.int64)),
            torch.from_numpy(np.ascontiguousarray(dataset.responses, dtype=np.float32)),
        )

    def _loss(
        self,
        outputs: dict[str, Tensor],
        outcomes: Tensor,
        credit_targets: Tensor | None,
        credit_weights: Tensor | None,
    ) -> tuple[Tensor, dict[str, float]]:
        weights = self.loss_weights
        components: dict[str, float] = {}
        total = torch.zeros((), dtype=torch.float32)

        outcome_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["outcome_logit"], outcomes
        )
        components["outcome"] = float(outcome_loss.detach())
        total = total + weights.outcome * outcome_loss

        if weights.prefix > 0.0:
            prefix_targets = outcomes.unsqueeze(1).expand_as(outputs["prefix_logits"])
            prefix_loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["prefix_logits"], prefix_targets
            )
            components["prefix"] = float(prefix_loss.detach())
            total = total + weights.prefix * prefix_loss

        if weights.telescoping > 0.0:
            # Per-step rewards are adjacent prefix-value differences, so the
            # telescoping identity holds exactly; this term instead ties the
            # final prefix value to the outcome head.
            telescoping_loss = torch.mean(
                torch.square(
                    torch.sigmoid(outputs["prefix_logits"][:, -1])
                    - torch.sigmoid(outputs["outcome_logit"])
                )
            )
            components["telescoping"] = float(telescoping_loss.detach())
            total = total + weights.telescoping * telescoping_loss

        if weights.credit > 0.0 and credit_targets is not None and credit_weights is not None:
            weight_sum = credit_weights.sum()
            if float(weight_sum) > 0.0:
                predicted = self._credit_from_outputs(outputs)
                credit_loss = (
                    credit_weights * torch.square(predicted - credit_targets)
                ).sum() / weight_sum
                components["credit"] = float(credit_loss.detach())
                total = total + weights.credit * credit_loss

        return total, components

    def _credit_from_outputs(self, outputs: dict[str, Tensor]) -> Tensor:
        action_values = outputs["action_values"]
        logged = outputs["logged_action_values"]
        return logged - action_values[:, :, self.reference_action]

    def fit(
        self,
        dataset: SequenceDataset,
        *,
        training: TrainingSettings | None = None,
    ) -> dict[str, float]:
        settings = training or TrainingSettings()
        _seed_torch(self.seed)
        self._feature_mean = dataset.observations.reshape(-1, self.observation_dim).mean(axis=0)
        scale = dataset.observations.reshape(-1, self.observation_dim).std(axis=0)
        self._feature_scale = np.where(scale > 1.0e-8, scale, 1.0)

        observations, actions, responses = self._tensors(dataset)
        outcomes = torch.from_numpy(np.ascontiguousarray(dataset.outcomes, dtype=np.float32))
        credit_targets: Tensor | None = None
        credit_weights: Tensor | None = None
        if dataset.credit_targets is not None and dataset.credit_se is not None:
            assert dataset.credit_mask is not None
            targets = np.where(dataset.credit_mask, dataset.credit_targets, 0.0)
            precision = dataset.credit_mask / (np.square(dataset.credit_se) + _SE_EPSILON)
            credit_targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
            credit_weights = torch.from_numpy(np.ascontiguousarray(precision, dtype=np.float32))

        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        generator = torch.Generator().manual_seed(self.seed)
        components: dict[str, float] = {}
        final_loss = 0.0
        self._net.train()
        for _ in range(settings.epochs):
            order = torch.randperm(dataset.episodes, generator=generator)
            epoch_loss = 0.0
            for start in range(0, dataset.episodes, settings.batch_size):
                batch = order[start : start + settings.batch_size]
                outputs = self._forward_with_logged_values(
                    observations[batch], actions[batch], responses[batch]
                )
                loss, components = self._loss(
                    outputs,
                    outcomes[batch],
                    None if credit_targets is None else credit_targets[batch],
                    None if credit_weights is None else credit_weights[batch],
                )
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(self._net.parameters(), settings.grad_clip)
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(batch)
            final_loss = epoch_loss / dataset.episodes
        self._net.eval()
        return {"loss": final_loss, **{f"loss_{k}": v for k, v in components.items()}}

    def _forward_with_logged_values(
        self, observations: Tensor, actions: Tensor, responses: Tensor
    ) -> dict[str, Tensor]:
        outputs: dict[str, Tensor] = self._net(observations, actions, responses)
        outputs["logged_action_values"] = (
            outputs["action_values"].gather(2, actions.unsqueeze(-1)).squeeze(-1)
        )
        return outputs

    def _predict(self, dataset: SequenceDataset) -> dict[str, Tensor]:
        observations, actions, responses = self._tensors(dataset)
        self._net.eval()
        with torch.no_grad():
            return self._forward_with_logged_values(observations, actions, responses)

    def predict_outcome_probability(self, dataset: SequenceDataset) -> FloatArray:
        outputs = self._predict(dataset)
        return np.asarray(torch.sigmoid(outputs["outcome_logit"]).numpy(), dtype=np.float64)

    def predict_prefix_values(self, dataset: SequenceDataset) -> FloatArray:
        """Return prefix-value probabilities ``V_0..V_T`` from the prefix head."""

        outputs = self._predict(dataset)
        return np.asarray(torch.sigmoid(outputs["prefix_logits"]).numpy(), dtype=np.float64)

    def predict_outcome_head_prefix_values(self, dataset: SequenceDataset) -> FloatArray:
        """Evaluate the terminal-outcome head on every prefix boundary.

        This is the honest credit diagnostic for an outcome-only variant: the
        head was trained only at the final position.
        """

        outputs = self._predict(dataset)
        return np.asarray(torch.sigmoid(outputs["outcome_prefix_logits"]).numpy(), dtype=np.float64)

    def predict_action_values(self, dataset: SequenceDataset) -> FloatArray:
        outputs = self._predict(dataset)
        return np.asarray(outputs["action_values"].numpy(), dtype=np.float64)

    def predict_logged_credit(self, dataset: SequenceDataset) -> FloatArray:
        """Return ``Q(h_t, a_t) - Q(h_t, a_ref)`` for each logged step."""

        outputs = self._predict(dataset)
        return np.asarray(self._credit_from_outputs(outputs).numpy(), dtype=np.float64)
