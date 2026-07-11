"""Linear softmax policies and REINFORCE against trajectory-level rewards.

Rollouts happen in the real structural world; only the reward is learned.
Action sampling consumes the world's exogenous policy draws, so a fixed seed
plus fixed policy parameters yields byte-identical trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class FeatureFunction(Protocol):
    def __call__(self, observation: Any) -> list[float]: ...


class SoftmaxPolicy(nn.Module):
    """A linear softmax policy over released observation features."""

    def __init__(self, *, feature_dim: int, n_actions: int, seed: int = 0) -> None:
        super().__init__()
        if feature_dim <= 0 or n_actions <= 1:
            raise ValueError("feature_dim must be positive and n_actions > 1")
        torch.manual_seed(seed)
        self.linear = nn.Linear(feature_dim, n_actions)

    def logits(self, features: Tensor) -> Tensor:
        output: Tensor = self.linear(features)
        return output

    def log_probabilities(self, features: Tensor) -> Tensor:
        return torch.log_softmax(self.logits(features), dim=-1)

    def probabilities(self, features: list[float]) -> FloatArray:
        with torch.no_grad():
            tensor = torch.as_tensor(features, dtype=torch.float32)
            return np.asarray(torch.softmax(self.logits(tensor), dim=-1).numpy(), dtype=np.float64)

    def sample(self, features: list[float], uniform_draw: float) -> int:
        """Sample via the cumulative distribution and an external uniform draw."""

        probabilities = self.probabilities(features)
        cumulative = 0.0
        for action_index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if uniform_draw < cumulative:
                return action_index
        return len(probabilities) - 1

    def clone_detached(self) -> SoftmaxPolicy:
        copy = SoftmaxPolicy(
            feature_dim=self.linear.in_features, n_actions=self.linear.out_features
        )
        copy.load_state_dict({k: v.clone() for k, v in self.state_dict().items()})
        for parameter in copy.parameters():
            parameter.requires_grad_(False)
        return copy

    @classmethod
    def behavior_clone(
        cls,
        features: FloatArray,
        actions: IntArray,
        *,
        n_actions: int,
        epochs: int = 120,
        learning_rate: float = 0.05,
        seed: int = 0,
    ) -> SoftmaxPolicy:
        """Fit a clone of logged behavior as the optimization starting point."""

        policy = cls(feature_dim=int(features.shape[-1]), n_actions=n_actions, seed=seed)
        inputs = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        targets = torch.from_numpy(np.ascontiguousarray(actions, dtype=np.int64))
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        for _ in range(epochs):
            loss = nn.functional.cross_entropy(policy.logits(inputs), targets)
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        return policy


@dataclass(frozen=True, slots=True)
class WorldPolicyAdapter:
    """Expose a feature-space policy through the world ``Policy`` protocol."""

    policy: SoftmaxPolicy
    feature_fn: FeatureFunction

    def select_action_index(self, observation: Any, random_value: float) -> int:
        return self.policy.sample(self.feature_fn(observation), random_value)


@dataclass(frozen=True)
class EpisodeBatch:
    """Trajectories collected in the real world under a feature policy."""

    observations: FloatArray  # [episodes, horizon, features]
    actions: IntArray  # [episodes, horizon]
    responses: FloatArray  # [episodes, horizon]
    proxies: FloatArray  # [episodes]
    utilities: FloatArray  # [episodes]

    @property
    def episodes(self) -> int:
        return int(self.observations.shape[0])


def collect_episodes(
    world: Any,
    adapter: WorldPolicyAdapter,
    *,
    episodes: int,
    seed_base: int,
) -> EpisodeBatch:
    """Roll out the current policy on fresh seeded exogenous noise."""

    feature_rows: list[list[list[float]]] = []
    action_rows: list[list[int]] = []
    response_rows: list[list[float]] = []
    proxies: list[float] = []
    utilities: list[float] = []
    for episode_index in range(episodes):
        exogenous = world.sample_exogenous(seed_base + episode_index)
        state = world.initial_state()
        features: list[list[float]] = []
        actions: list[int] = []
        responses: list[float] = []
        for step_index in range(world.horizon):
            observation = world.observe(state)
            step_features = adapter.feature_fn(observation)
            action_index = adapter.select_action_index(
                observation, world.policy_random_value(exogenous[step_index])
            )
            transition = world.step(state, world.action_space[action_index], exogenous[step_index])
            features.append(step_features)
            actions.append(action_index)
            responses.append(transition.info_value("response"))
            state = transition.next_state
        feature_rows.append(features)
        action_rows.append(actions)
        response_rows.append(responses)
        proxies.append(world.terminal_proxy(state))
        utilities.append(world.terminal_utility(state))
    return EpisodeBatch(
        observations=np.asarray(feature_rows, dtype=np.float64),
        actions=np.asarray(action_rows, dtype=np.int64),
        responses=np.asarray(response_rows, dtype=np.float64),
        proxies=np.asarray(proxies, dtype=np.float64),
        utilities=np.asarray(utilities, dtype=np.float64),
    )


def mean_kl_divergence(
    policy: SoftmaxPolicy,
    reference: SoftmaxPolicy,
    observations: FloatArray,
) -> float:
    """Mean KL(policy || reference) over the visited states."""

    flat = torch.from_numpy(
        np.ascontiguousarray(observations.reshape(-1, observations.shape[-1]), dtype=np.float32)
    )
    with torch.no_grad():
        policy_log = policy.log_probabilities(flat)
        reference_log = reference.log_probabilities(flat)
        kl = torch.sum(torch.exp(policy_log) * (policy_log - reference_log), dim=-1)
        return float(kl.mean())


def reinforce_update(
    policy: SoftmaxPolicy,
    optimizer: torch.optim.Optimizer,
    batch: EpisodeBatch,
    rewards: FloatArray,
    *,
    entropy_coefficient: float = 0.01,
    kl_coefficient: float = 0.0,
    reference: SoftmaxPolicy | None = None,
) -> dict[str, float]:
    """One REINFORCE step with a batch-mean baseline.

    ``rewards`` are trajectory-level learned rewards. The optional KL penalty
    against a frozen reference implements the classic regularization baseline.
    """

    if rewards.shape != (batch.episodes,):
        raise ValueError("rewards must have one value per episode")
    if kl_coefficient > 0.0 and reference is None:
        raise ValueError("a reference policy is required for a KL penalty")

    features = torch.from_numpy(np.ascontiguousarray(batch.observations, dtype=np.float32))
    actions = torch.from_numpy(np.ascontiguousarray(batch.actions, dtype=np.int64))
    advantage = torch.from_numpy(np.ascontiguousarray(rewards - rewards.mean(), dtype=np.float32))

    log_probabilities = policy.log_probabilities(features)
    chosen = log_probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    objective = (advantage * chosen.sum(dim=-1)).mean()
    entropy = -(torch.exp(log_probabilities) * log_probabilities).sum(dim=-1).mean()
    loss = -objective - entropy_coefficient * entropy
    if kl_coefficient > 0.0 and reference is not None:
        reference_log = reference.log_probabilities(features)
        kl = (torch.exp(log_probabilities) * (log_probabilities - reference_log)).sum(dim=-1).mean()
        loss = loss + kl_coefficient * kl

    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "objective": float(objective.detach()),
        "entropy": float(entropy.detach()),
        "mean_reward": float(rewards.mean()),
    }
