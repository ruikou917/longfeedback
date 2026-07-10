"""World D: proxy-utility divergence (a genuine Goodhart setting).

State tracks useful progress ``g``, engagement ``e``, trust ``q``, and
dependency ``d``. The behavioral proxy is a terminal return event driven by
engagement, while true utility rewards progress and trust and penalizes
dependency and interruptions. Fear/urgency actions reliably raise the proxy
while lowering utility, which the unit tests assert — without that property
the overoptimization experiments would be meaningless.

Trust and dependency are hidden from the released observation; progress is
observed with seeded noise.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import overload

from .base import Episode, Policy, Transition


class InfluenceAction(StrEnum):
    NOOP = "noop"
    HELPFUL = "helpful"
    URGENT = "urgent"
    FLATTER = "flatter"
    FEAR = "fear"


_ACTIONS: tuple[InfluenceAction, ...] = (
    InfluenceAction.NOOP,
    InfluenceAction.HELPFUL,
    InfluenceAction.URGENT,
    InfluenceAction.FLATTER,
    InfluenceAction.FEAR,
)


@dataclass(frozen=True, slots=True)
class ProxyUtilityConfig:
    horizon: int = 12
    progress_lifts: tuple[float, ...] = (0.0, 0.6, 0.15, 0.0, 0.0)
    engagement_lifts: tuple[float, ...] = (0.0, 0.25, 0.7, 0.55, 0.9)
    trust_lifts: tuple[float, ...] = (0.0, 0.2, -0.2, -0.05, -0.45)
    dependency_lifts: tuple[float, ...] = (0.0, 0.0, 0.25, 0.2, 0.4)
    interruption_costs: tuple[float, ...] = (0.0, 0.1, 0.3, 0.15, 0.35)
    engagement_decay: float = 0.75
    trust_decay: float = 0.95
    dependency_decay: float = 0.9
    engagement_noise_std: float = 0.15
    progress_noise_std: float = 0.1
    observation_noise_std: float = 0.1
    return_engagement_weight: float = 1.1
    return_trust_weight: float = 0.35
    return_bias: float = -2.2
    utility_progress_weight: float = 1.0
    utility_trust_weight: float = 0.8
    utility_dependency_weight: float = 0.8
    utility_interruption_weight: float = 0.25
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        for name in (
            "progress_lifts",
            "engagement_lifts",
            "trust_lifts",
            "dependency_lifts",
            "interruption_costs",
        ):
            if len(getattr(self, name)) != len(_ACTIONS):
                raise ValueError(f"{name} must have one value per action")
        for name in ("engagement_decay", "trust_decay", "dependency_decay"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("engagement_noise_std", "progress_noise_std", "observation_noise_std"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class ProxyUtilityState:
    step_index: int
    progress: float
    engagement: float
    trust: float
    dependency: float
    cumulative_interruption: float
    last_action: int | None
    returned: bool
    progress_observation_noise: float = 0.0


@dataclass(frozen=True, slots=True)
class ProxyUtilityObservation:
    """Released observation: trust and dependency are hidden."""

    step_index: int
    engagement: float
    noisy_progress: float
    last_action: int | None


@dataclass(frozen=True, slots=True)
class ProxyUtilityStepNoise:
    step_index: int
    engagement_noise: float
    progress_noise: float
    observation_noise: float
    return_draw: float
    policy_draw: float


@dataclass(frozen=True, slots=True)
class ProxyUtilityExogenousNoise(Sequence[ProxyUtilityStepNoise]):
    seed: int
    steps: tuple[ProxyUtilityStepNoise, ...]

    def __len__(self) -> int:
        return len(self.steps)

    @overload
    def __getitem__(self, index: int) -> ProxyUtilityStepNoise: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ProxyUtilityStepNoise, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ProxyUtilityStepNoise | tuple[ProxyUtilityStepNoise, ...]:
        return self.steps[index]

    def __iter__(self) -> Iterator[ProxyUtilityStepNoise]:
        return iter(self.steps)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class ProxyUtilityWorld:
    """A stateless SCM; every stochastic input is passed to ``step``."""

    action_space: tuple[InfluenceAction, ...] = _ACTIONS

    def __init__(self, config: ProxyUtilityConfig | None = None) -> None:
        self.config = config or ProxyUtilityConfig()
        self.horizon = self.config.horizon

    @property
    def is_deterministic(self) -> bool:
        return self.config.deterministic

    def initial_state(self) -> ProxyUtilityState:
        return ProxyUtilityState(
            step_index=0,
            progress=0.0,
            engagement=0.0,
            trust=1.0,
            dependency=0.0,
            cumulative_interruption=0.0,
            last_action=None,
            returned=False,
        )

    def observe(self, state: ProxyUtilityState) -> ProxyUtilityObservation:
        return ProxyUtilityObservation(
            step_index=state.step_index,
            engagement=state.engagement,
            noisy_progress=state.progress + state.progress_observation_noise,
            last_action=state.last_action,
        )

    def sample_exogenous(self, seed: int) -> ProxyUtilityExogenousNoise:
        rng = random.Random(seed)
        config = self.config
        steps = tuple(
            ProxyUtilityStepNoise(
                step_index=step,
                engagement_noise=rng.gauss(0.0, config.engagement_noise_std),
                progress_noise=rng.gauss(0.0, config.progress_noise_std),
                observation_noise=rng.gauss(0.0, config.observation_noise_std),
                return_draw=rng.random(),
                policy_draw=rng.random(),
            )
            for step in range(self.horizon)
        )
        return ProxyUtilityExogenousNoise(seed=seed, steps=steps)

    @staticmethod
    def policy_random_value(noise: ProxyUtilityStepNoise) -> float:
        return noise.policy_draw

    def return_probability(self, state: ProxyUtilityState) -> float:
        config = self.config
        return _sigmoid(
            config.return_engagement_weight * state.engagement
            + config.return_trust_weight * state.trust
            + config.return_bias
        )

    def step(
        self,
        state: ProxyUtilityState,
        action: InfluenceAction,
        exogenous: ProxyUtilityStepNoise,
    ) -> Transition[
        ProxyUtilityState,
        InfluenceAction,
        ProxyUtilityObservation,
        ProxyUtilityStepNoise,
    ]:
        if state.step_index >= self.horizon:
            raise RuntimeError("cannot step a terminated world")
        if exogenous.step_index != state.step_index:
            raise ValueError("exogenous noise step does not match state")
        try:
            action = InfluenceAction(action)
            action_index = self.action_space.index(action)
        except ValueError as error:
            raise ValueError(f"unsupported action: {action!r}") from error

        config = self.config
        noise_scale = 0.0 if config.deterministic else 1.0
        next_progress = (
            state.progress
            + config.progress_lifts[action_index]
            + noise_scale * exogenous.progress_noise
        )
        next_engagement = (
            config.engagement_decay * state.engagement
            + config.engagement_lifts[action_index]
            + noise_scale * exogenous.engagement_noise
        )
        next_trust = max(0.0, config.trust_decay * state.trust + config.trust_lifts[action_index])
        next_dependency = (
            config.dependency_decay * state.dependency + config.dependency_lifts[action_index]
        )

        next_step = state.step_index + 1
        terminated = next_step >= self.horizon
        returned = state.returned
        return_probability = 0.0
        if terminated:
            probe = ProxyUtilityState(
                step_index=next_step,
                progress=next_progress,
                engagement=next_engagement,
                trust=next_trust,
                dependency=next_dependency,
                cumulative_interruption=0.0,
                last_action=action_index,
                returned=False,
            )
            return_probability = self.return_probability(probe)
            if config.deterministic:
                returned = return_probability >= 0.5
            else:
                returned = exogenous.return_draw < return_probability

        next_state = ProxyUtilityState(
            step_index=next_step,
            progress=next_progress,
            engagement=next_engagement,
            trust=next_trust,
            dependency=next_dependency,
            cumulative_interruption=(
                state.cumulative_interruption + config.interruption_costs[action_index]
            ),
            last_action=action_index,
            returned=returned,
            progress_observation_noise=noise_scale * exogenous.observation_noise,
        )
        return Transition(
            step_index=state.step_index,
            state=state,
            observation=self.observe(state),
            action=action,
            exogenous=exogenous,
            next_state=next_state,
            next_observation=self.observe(next_state),
            reward=self.terminal_proxy(next_state) if terminated else 0.0,
            terminated=terminated,
            info=(
                ("response", next_engagement),
                ("return_probability", return_probability),
            ),
        )

    def terminal_proxy(self, state: ProxyUtilityState) -> float:
        return float(state.returned)

    def terminal_utility(self, state: ProxyUtilityState) -> float:
        config = self.config
        return (
            config.utility_progress_weight * state.progress
            + config.utility_trust_weight * state.trust
            - config.utility_dependency_weight * state.dependency
            - config.utility_interruption_weight * state.cumulative_interruption
        )

    def rollout(
        self,
        actions: Sequence[InfluenceAction],
        exogenous: Sequence[ProxyUtilityStepNoise],
        *,
        initial_state: ProxyUtilityState | None = None,
    ) -> Episode[
        ProxyUtilityState,
        InfluenceAction,
        ProxyUtilityObservation,
        ProxyUtilityStepNoise,
    ]:
        state = initial_state or self.initial_state()
        rollout_initial_state = state
        initial_observation = self.observe(state)
        transitions = []
        for action in actions:
            if state.step_index >= self.horizon:
                break
            if state.step_index >= len(exogenous):
                raise ValueError("exogenous sequence is shorter than the rollout")
            transition = self.step(state, action, exogenous[state.step_index])
            transitions.append(transition)
            state = transition.next_state
        if state.step_index != self.horizon:
            raise ValueError("actions do not reach the configured horizon")
        return Episode(
            initial_state=rollout_initial_state,
            initial_observation=initial_observation,
            transitions=tuple(transitions),
            terminal_proxy=self.terminal_proxy(state),
            terminal_utility=self.terminal_utility(state),
            seed=getattr(exogenous, "seed", None),
        )

    def rollout_policy(
        self,
        policy: Policy[ProxyUtilityObservation, InfluenceAction],
        exogenous: Sequence[ProxyUtilityStepNoise],
        *,
        initial_state: ProxyUtilityState | None = None,
    ) -> Episode[
        ProxyUtilityState,
        InfluenceAction,
        ProxyUtilityObservation,
        ProxyUtilityStepNoise,
    ]:
        state = initial_state or self.initial_state()
        actions: list[InfluenceAction] = []
        for step_index in range(state.step_index, self.horizon):
            noise = exogenous[step_index]
            action = policy.select_action(
                self.observe(state),
                step_index=step_index,
                random_value=noise.policy_draw,
            )
            actions.append(action)
            state = self.step(state, action, noise).next_state
        return self.rollout(actions, exogenous, initial_state=initial_state)


@dataclass(frozen=True, slots=True)
class MixedInfluencePolicy:
    """Logging policy mixing helpful and engagement-bait actions.

    Propensities depend only on released observation fields. Low observed
    engagement biases toward attention-grabbing actions, mimicking a
    metric-driven production heuristic.
    """

    epsilon: float
    bait_bias: float = 0.35

    def probabilities(self, observation: ProxyUtilityObservation) -> tuple[float, ...]:
        favored = (
            InfluenceAction.URGENT
            if observation.engagement < self.bait_bias
            else InfluenceAction.HELPFUL
        )
        probabilities = [self.epsilon / len(_ACTIONS) for _ in _ACTIONS]
        probabilities[_ACTIONS.index(favored)] += 1.0 - self.epsilon
        return tuple(probabilities)

    def select_action(
        self,
        observation: ProxyUtilityObservation,
        *,
        step_index: int,
        random_value: float,
    ) -> InfluenceAction:
        del step_index
        cumulative = 0.0
        for action, probability in zip(_ACTIONS, self.probabilities(observation), strict=True):
            cumulative += probability
            if random_value < cumulative:
                return action
        return _ACTIONS[-1]

    def log_probability(
        self,
        observation: ProxyUtilityObservation,
        action: InfluenceAction,
    ) -> float:
        return math.log(self.probabilities(observation)[_ACTIONS.index(action)])
