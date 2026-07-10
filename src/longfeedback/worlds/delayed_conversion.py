"""World C: delayed conversion with competing causes and saturation.

Each outreach action spawns a latent impulse whose effect arrives after a
random action-dependent delay and then decays. The per-step conversion hazard
sums the arrived impulses, so several earlier actions compete as causes of one
delayed outcome. Later actions saturate (diminishing quality). Pending
impulses are hidden from the released observation.

Proxy and utility: ``Y = converted``, ``U = value * converted - cost * sends``.
The proxy/utility gap here is deliberately small (World D owns the Goodhart
setting); World C's difficulty is long, variable credit delay.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import overload

from .base import Episode, Policy, Transition


class ConversionAction(StrEnum):
    NONE = "none"
    SOFT = "soft"
    STRONG = "strong"


_ACTIONS: tuple[ConversionAction, ...] = (
    ConversionAction.NONE,
    ConversionAction.SOFT,
    ConversionAction.STRONG,
)


@dataclass(frozen=True, slots=True)
class DelayedConversionConfig:
    horizon: int = 16
    base_hazard: float = 0.003
    impulse_qualities: tuple[float, float, float] = (0.0, 0.1, 0.2)
    delay_geometric_p: tuple[float, float, float] = (1.0, 0.5, 0.25)
    max_delay: int = 6
    kernel_decay: float = 0.6
    saturation_rate: float = 0.6
    conversion_value: float = 2.0
    action_cost: float = 0.04
    deterministic: bool = False
    deterministic_threshold: float = 0.15

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.base_hazard < 0.0:
            raise ValueError("base_hazard cannot be negative")
        if len(self.impulse_qualities) != len(_ACTIONS):
            raise ValueError("impulse_qualities must have one value per action")
        if len(self.delay_geometric_p) != len(_ACTIONS):
            raise ValueError("delay_geometric_p must have one value per action")
        for probability in self.delay_geometric_p:
            if not 0.0 < probability <= 1.0:
                raise ValueError("delay_geometric_p values must lie in (0, 1]")
        if self.max_delay < 1:
            raise ValueError("max_delay must be at least 1")
        if not 0.0 <= self.kernel_decay < 1.0:
            raise ValueError("kernel_decay must lie in [0, 1)")
        if self.saturation_rate < 0.0 or self.action_cost < 0.0:
            raise ValueError("saturation_rate and action_cost cannot be negative")


@dataclass(frozen=True, slots=True)
class ConversionImpulse:
    arrival_step: int
    quality: float


@dataclass(frozen=True, slots=True)
class DelayedConversionState:
    step_index: int
    converted: bool
    sends: int
    last_action: int | None
    steps_since_send: int
    impulses: tuple[ConversionImpulse, ...]


@dataclass(frozen=True, slots=True)
class DelayedConversionObservation:
    """Released observation; pending impulses stay hidden."""

    step_index: int
    converted: bool
    sends: int
    last_action: int | None
    steps_since_send: int


@dataclass(frozen=True, slots=True)
class DelayedConversionStepNoise:
    step_index: int
    delay_draw: float
    conversion_draw: float
    policy_draw: float


@dataclass(frozen=True, slots=True)
class DelayedConversionExogenousNoise(Sequence[DelayedConversionStepNoise]):
    seed: int
    steps: tuple[DelayedConversionStepNoise, ...]

    def __len__(self) -> int:
        return len(self.steps)

    @overload
    def __getitem__(self, index: int) -> DelayedConversionStepNoise: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[DelayedConversionStepNoise, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> DelayedConversionStepNoise | tuple[DelayedConversionStepNoise, ...]:
        return self.steps[index]

    def __iter__(self) -> Iterator[DelayedConversionStepNoise]:
        return iter(self.steps)


class DelayedConversionWorld:
    """A stateless SCM; every stochastic input is passed to ``step``."""

    action_space: tuple[ConversionAction, ...] = _ACTIONS

    def __init__(self, config: DelayedConversionConfig | None = None) -> None:
        self.config = config or DelayedConversionConfig()
        self.horizon = self.config.horizon

    @property
    def is_deterministic(self) -> bool:
        return self.config.deterministic

    def initial_state(self) -> DelayedConversionState:
        return DelayedConversionState(
            step_index=0,
            converted=False,
            sends=0,
            last_action=None,
            steps_since_send=self.horizon,
            impulses=(),
        )

    def observe(self, state: DelayedConversionState) -> DelayedConversionObservation:
        return DelayedConversionObservation(
            step_index=state.step_index,
            converted=state.converted,
            sends=state.sends,
            last_action=state.last_action,
            steps_since_send=state.steps_since_send,
        )

    def sample_exogenous(self, seed: int) -> DelayedConversionExogenousNoise:
        rng = random.Random(seed)
        steps = tuple(
            DelayedConversionStepNoise(
                step_index=step,
                delay_draw=rng.random(),
                conversion_draw=rng.random(),
                policy_draw=rng.random(),
            )
            for step in range(self.horizon)
        )
        return DelayedConversionExogenousNoise(seed=seed, steps=steps)

    @staticmethod
    def policy_random_value(noise: DelayedConversionStepNoise) -> float:
        return noise.policy_draw

    def _sample_delay(self, action_index: int, draw: float) -> int:
        """Map a uniform draw to a truncated geometric arrival delay (>=1)."""

        probability = self.config.delay_geometric_p[action_index]
        if self.config.deterministic or probability >= 1.0:
            return max(1, min(self.config.max_delay, round(1.0 / probability)))
        # Inverse CDF of a geometric distribution starting at 1.
        delay = 1 + int(math.log(max(1.0 - draw, 1.0e-12)) / math.log(1.0 - probability))
        return max(1, min(delay, self.config.max_delay))

    def hazard(self, state: DelayedConversionState) -> float:
        """Per-step conversion hazard from base rate plus arrived impulses."""

        config = self.config
        total = config.base_hazard
        for impulse in state.impulses:
            age = state.step_index - impulse.arrival_step
            if age >= 0:
                total += impulse.quality * (config.kernel_decay**age)
        return total

    def step(
        self,
        state: DelayedConversionState,
        action: ConversionAction,
        exogenous: DelayedConversionStepNoise,
    ) -> Transition[
        DelayedConversionState,
        ConversionAction,
        DelayedConversionObservation,
        DelayedConversionStepNoise,
    ]:
        if state.step_index >= self.horizon:
            raise RuntimeError("cannot step a terminated world")
        if exogenous.step_index != state.step_index:
            raise ValueError("exogenous noise step does not match state")
        try:
            action = ConversionAction(action)
            action_index = self.action_space.index(action)
        except ValueError as error:
            raise ValueError(f"unsupported action: {action!r}") from error

        config = self.config
        impulses = state.impulses
        sends = state.sends
        is_send = action is not ConversionAction.NONE
        if is_send and not state.converted:
            quality = config.impulse_qualities[action_index] * math.exp(
                -config.saturation_rate * state.sends
            )
            delay = self._sample_delay(action_index, exogenous.delay_draw)
            impulses = (
                *impulses,
                ConversionImpulse(arrival_step=state.step_index + delay, quality=quality),
            )
        if is_send:
            sends += 1

        hazard = self.hazard(state)
        conversion_probability = 1.0 - math.exp(-hazard)
        if state.converted:
            converted = True
            response = 0.0
        elif config.deterministic:
            converted = hazard >= config.deterministic_threshold
            response = float(converted)
        else:
            converted = exogenous.conversion_draw < conversion_probability
            response = float(converted)

        next_state = DelayedConversionState(
            step_index=state.step_index + 1,
            converted=converted,
            sends=sends,
            last_action=action_index,
            steps_since_send=0 if is_send else state.steps_since_send + 1,
            impulses=impulses,
        )
        terminated = next_state.step_index >= self.horizon
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
                ("response", response),
                ("hazard", hazard),
                ("conversion_probability", conversion_probability),
            ),
        )

    def terminal_proxy(self, state: DelayedConversionState) -> float:
        return float(state.converted)

    def terminal_utility(self, state: DelayedConversionState) -> float:
        config = self.config
        return config.conversion_value * float(state.converted) - config.action_cost * state.sends

    def rollout(
        self,
        actions: Sequence[ConversionAction],
        exogenous: Sequence[DelayedConversionStepNoise],
        *,
        initial_state: DelayedConversionState | None = None,
    ) -> Episode[
        DelayedConversionState,
        ConversionAction,
        DelayedConversionObservation,
        DelayedConversionStepNoise,
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
        policy: Policy[DelayedConversionObservation, ConversionAction],
        exogenous: Sequence[DelayedConversionStepNoise],
        *,
        initial_state: DelayedConversionState | None = None,
    ) -> Episode[
        DelayedConversionState,
        ConversionAction,
        DelayedConversionObservation,
        DelayedConversionStepNoise,
    ]:
        state = initial_state or self.initial_state()
        actions: list[ConversionAction] = []
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
class SpacedOutreachPolicy:
    """Heuristic sender with exact released-observation propensities.

    Sends STRONG on a fixed cadence, SOFT halfway between, and stops sending
    once conversion is observed.
    """

    epsilon: float
    cadence: int = 6

    def probabilities(self, observation: DelayedConversionObservation) -> tuple[float, ...]:
        if observation.converted:
            favored = 0
        elif observation.step_index % self.cadence == 0:
            favored = 2
        elif observation.step_index % self.cadence == self.cadence // 2:
            favored = 1
        else:
            favored = 0
        probabilities = [self.epsilon / len(_ACTIONS) for _ in _ACTIONS]
        probabilities[favored] += 1.0 - self.epsilon
        return tuple(probabilities)

    def select_action(
        self,
        observation: DelayedConversionObservation,
        *,
        step_index: int,
        random_value: float,
    ) -> ConversionAction:
        del step_index
        cumulative = 0.0
        for action, probability in zip(_ACTIONS, self.probabilities(observation), strict=True):
            cumulative += probability
            if random_value < cumulative:
                return action
        return _ACTIONS[-1]

    def log_probability(
        self,
        observation: DelayedConversionObservation,
        action: ConversionAction,
    ) -> float:
        return math.log(self.probabilities(observation)[_ACTIONS.index(action)])
