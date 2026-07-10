"""World A: fatigue and habit dynamics with explicit seeded noise."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import overload

from .base import Episode, Policy, Transition


class FatigueAction(StrEnum):
    NOOP = "noop"
    HELPFUL = "helpful"
    URGENT = "urgent"


class FatigueObservability(StrEnum):
    """How much latent state the released observation reveals."""

    ORACLE = "oracle"
    NOISY = "noisy"
    PARTIAL = "partial"


_ACTIONS: tuple[FatigueAction, ...] = (
    FatigueAction.NOOP,
    FatigueAction.HELPFUL,
    FatigueAction.URGENT,
)


@dataclass(frozen=True, slots=True)
class FatigueHabitConfig:
    horizon: int = 8
    initial_habit: float = 0.0
    initial_fatigue: float = 0.0
    habit_decay: float = 0.90
    fatigue_decay: float = 0.80
    base_response_logit: float = -0.25
    response_lifts: tuple[float, float, float] = (0.0, 1.0, 1.6)
    action_intensities: tuple[float, float, float] = (0.0, 0.5, 1.0)
    habituation_increments: tuple[float, float, float] = (0.0, 0.25, 0.45)
    habituation_decay: float = 0.75
    fatigue_sensitivity: float = 0.55
    fatigue_response_penalty: float = 0.80
    habituation_response_penalty: float = 0.50
    habit_gain: float = 0.65
    fatigue_utility_cost: float = 0.08
    action_utility_cost: float = 0.04
    proxy_threshold: float = 1.73
    response_noise_std: float = 0.0
    habit_noise_std: float = 0.0
    fatigue_noise_std: float = 0.0
    observability: FatigueObservability = FatigueObservability.ORACLE
    observation_noise_std: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        for name in ("response_lifts", "action_intensities", "habituation_increments"):
            if len(getattr(self, name)) != len(_ACTIONS):
                raise ValueError(f"{name} must have one value per action")
        for name in ("habit_decay", "fatigue_decay", "habituation_decay"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "response_noise_std",
            "habit_noise_std",
            "fatigue_noise_std",
            "observation_noise_std",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        object.__setattr__(self, "observability", FatigueObservability(self.observability))

    @classmethod
    def effects_disabled(cls, *, horizon: int = 8) -> FatigueHabitConfig:
        """Return a fixture where action identity has no structural effect."""

        return cls(
            horizon=horizon,
            response_lifts=(0.0, 0.0, 0.0),
            action_intensities=(0.0, 0.0, 0.0),
            habituation_increments=(0.0, 0.0, 0.0),
        )

    @classmethod
    def stochastic(
        cls,
        *,
        horizon: int = 12,
        observability: FatigueObservability | str = FatigueObservability.ORACLE,
        response_noise_std: float = 0.6,
        habit_noise_std: float = 0.05,
        fatigue_noise_std: float = 0.05,
        observation_noise_std: float = 0.1,
        proxy_threshold: float = 2.1,
    ) -> FatigueHabitConfig:
        """Return the Gate A stochastic preset for World A.

        The default proxy threshold keeps the horizon-12 behavioral proxy
        near a balanced positive rate under the Gate A behavior policy.
        """

        return cls(
            horizon=horizon,
            observability=FatigueObservability(observability),
            response_noise_std=response_noise_std,
            habit_noise_std=habit_noise_std,
            fatigue_noise_std=fatigue_noise_std,
            observation_noise_std=observation_noise_std,
            proxy_threshold=proxy_threshold,
        )


@dataclass(frozen=True, slots=True)
class FatigueHabitState:
    step_index: int
    habit: float
    fatigue: float
    habituation: tuple[float, float, float]
    last_response: float
    cumulative_fatigue: float
    cumulative_action_intensity: float
    habit_observation_noise: float = 0.0
    fatigue_observation_noise: float = 0.0


@dataclass(frozen=True, slots=True)
class FatigueHabitObservation:
    """Released observation; hidden fields are ``None``, never zero-filled."""

    step_index: int
    habit: float
    fatigue: float | None
    habituation: tuple[float, float, float] | None
    last_response: float


@dataclass(frozen=True, slots=True)
class FatigueHabitStepNoise:
    step_index: int
    response_noise: float
    habit_noise: float
    fatigue_noise: float
    policy_draw: float
    habit_observation_noise: float = 0.0
    fatigue_observation_noise: float = 0.0


@dataclass(frozen=True, slots=True)
class FatigueHabitExogenousNoise(Sequence[FatigueHabitStepNoise]):
    seed: int
    steps: tuple[FatigueHabitStepNoise, ...]

    def __len__(self) -> int:
        return len(self.steps)

    @overload
    def __getitem__(self, index: int) -> FatigueHabitStepNoise: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FatigueHabitStepNoise, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> FatigueHabitStepNoise | tuple[FatigueHabitStepNoise, ...]:
        return self.steps[index]

    def __iter__(self) -> Iterator[FatigueHabitStepNoise]:
        return iter(self.steps)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class FatigueHabitWorld:
    """A stateless SCM; every stochastic input is passed to ``step``."""

    action_space: tuple[FatigueAction, ...] = _ACTIONS

    def __init__(self, config: FatigueHabitConfig | None = None) -> None:
        self.config = config or FatigueHabitConfig()
        self.horizon = self.config.horizon

    @property
    def is_deterministic(self) -> bool:
        return (
            self.config.response_noise_std == 0.0
            and self.config.habit_noise_std == 0.0
            and self.config.fatigue_noise_std == 0.0
            and self.config.observation_noise_std == 0.0
        )

    def initial_state(self) -> FatigueHabitState:
        return FatigueHabitState(
            step_index=0,
            habit=self.config.initial_habit,
            fatigue=self.config.initial_fatigue,
            habituation=(0.0, 0.0, 0.0),
            last_response=0.0,
            cumulative_fatigue=0.0,
            cumulative_action_intensity=0.0,
        )

    def observe(self, state: FatigueHabitState) -> FatigueHabitObservation:
        observability = self.config.observability
        if observability is FatigueObservability.ORACLE:
            return FatigueHabitObservation(
                step_index=state.step_index,
                habit=state.habit,
                fatigue=state.fatigue,
                habituation=state.habituation,
                last_response=state.last_response,
            )
        noisy_habit = state.habit + state.habit_observation_noise
        if observability is FatigueObservability.NOISY:
            return FatigueHabitObservation(
                step_index=state.step_index,
                habit=noisy_habit,
                fatigue=state.fatigue + state.fatigue_observation_noise,
                habituation=state.habituation,
                last_response=state.last_response,
            )
        return FatigueHabitObservation(
            step_index=state.step_index,
            habit=noisy_habit,
            fatigue=None,
            habituation=None,
            last_response=state.last_response,
        )

    def sample_exogenous(self, seed: int) -> FatigueHabitExogenousNoise:
        rng = random.Random(seed)
        # Observation noise uses a separate derived stream so that enabling it
        # never perturbs the dynamics/policy draws of existing seeds.
        observation_rng = random.Random(f"observation:{seed}")
        steps = tuple(
            FatigueHabitStepNoise(
                step_index=step,
                response_noise=rng.gauss(0.0, self.config.response_noise_std),
                habit_noise=rng.gauss(0.0, self.config.habit_noise_std),
                fatigue_noise=rng.gauss(0.0, self.config.fatigue_noise_std),
                policy_draw=rng.random(),
                habit_observation_noise=observation_rng.gauss(
                    0.0, self.config.observation_noise_std
                ),
                fatigue_observation_noise=observation_rng.gauss(
                    0.0, self.config.observation_noise_std
                ),
            )
            for step in range(self.horizon)
        )
        return FatigueHabitExogenousNoise(seed=seed, steps=steps)

    @staticmethod
    def policy_random_value(noise: FatigueHabitStepNoise) -> float:
        return noise.policy_draw

    def step(
        self,
        state: FatigueHabitState,
        action: FatigueAction,
        exogenous: FatigueHabitStepNoise,
    ) -> Transition[
        FatigueHabitState,
        FatigueAction,
        FatigueHabitObservation,
        FatigueHabitStepNoise,
    ]:
        if state.step_index >= self.horizon:
            raise RuntimeError("cannot step a terminated world")
        if exogenous.step_index != state.step_index:
            raise ValueError("exogenous noise step does not match state")
        try:
            action = FatigueAction(action)
            action_index = self.action_space.index(action)
        except ValueError as error:
            raise ValueError(f"unsupported action: {action!r}") from error

        config = self.config
        response_logit = (
            config.base_response_logit
            + config.response_lifts[action_index]
            - config.fatigue_response_penalty * state.fatigue
            - config.habituation_response_penalty * state.habituation[action_index]
            + exogenous.response_noise
        )
        response = _sigmoid(response_logit)
        intensity = config.action_intensities[action_index]
        next_fatigue = max(
            0.0,
            config.fatigue_decay * state.fatigue
            + config.fatigue_sensitivity * intensity
            + exogenous.fatigue_noise,
        )

        habituation_values = [config.habituation_decay * value for value in state.habituation]
        habituation_values[action_index] += config.habituation_increments[action_index]
        next_habituation = (
            habituation_values[0],
            habituation_values[1],
            habituation_values[2],
        )
        next_habit = max(
            0.0,
            config.habit_decay * state.habit + config.habit_gain * response + exogenous.habit_noise,
        )
        next_state = FatigueHabitState(
            step_index=state.step_index + 1,
            habit=next_habit,
            fatigue=next_fatigue,
            habituation=next_habituation,
            last_response=response,
            cumulative_fatigue=state.cumulative_fatigue + next_fatigue,
            cumulative_action_intensity=state.cumulative_action_intensity + intensity,
            habit_observation_noise=exogenous.habit_observation_noise,
            fatigue_observation_noise=exogenous.fatigue_observation_noise,
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
                ("response_logit", response_logit),
                ("action_intensity", intensity),
            ),
        )

    def terminal_proxy(self, state: FatigueHabitState) -> float:
        return float(state.habit >= self.config.proxy_threshold)

    def terminal_utility(self, state: FatigueHabitState) -> float:
        return (
            state.habit
            - self.config.fatigue_utility_cost * state.cumulative_fatigue
            - self.config.action_utility_cost * state.cumulative_action_intensity
        )

    def rollout(
        self,
        actions: Sequence[FatigueAction],
        exogenous: Sequence[FatigueHabitStepNoise],
        *,
        initial_state: FatigueHabitState | None = None,
    ) -> Episode[
        FatigueHabitState,
        FatigueAction,
        FatigueHabitObservation,
        FatigueHabitStepNoise,
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
        policy: Policy[FatigueHabitObservation, FatigueAction],
        exogenous: Sequence[FatigueHabitStepNoise],
        *,
        initial_state: FatigueHabitState | None = None,
    ) -> Episode[
        FatigueHabitState,
        FatigueAction,
        FatigueHabitObservation,
        FatigueHabitStepNoise,
    ]:
        state = initial_state or self.initial_state()
        actions: list[FatigueAction] = []
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
