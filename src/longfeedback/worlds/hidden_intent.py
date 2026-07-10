"""World B: hidden Markov intent with exogenous shifts and confounded logging.

The latent intent drifts exogenously (never caused by actions), step progress
depends on the intent/action match, and the terminal behavioral proxy adds an
exogenous shock so that outcome stochasticity and proxy/utility separation are
both present. A privileged noisy intent signal is carried in the observation
for logging policies only; it must never enter released model features.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import overload

from .base import Episode, Policy, Transition


class IntentAction(StrEnum):
    FOCUS_A = "focus_a"
    FOCUS_B = "focus_b"
    FOCUS_C = "focus_c"


_ACTIONS: tuple[IntentAction, ...] = (
    IntentAction.FOCUS_A,
    IntentAction.FOCUS_B,
    IntentAction.FOCUS_C,
)
_N_INTENTS = len(_ACTIONS)


@dataclass(frozen=True, slots=True)
class HiddenIntentConfig:
    horizon: int = 12
    stay_probability: float = 0.85
    match_logit: float = 1.2
    mismatch_logit: float = -1.8
    progress_shock_scale: float = 1.0
    proxy_threshold: float = 6.5
    signal_accuracy: float = 0.85
    deterministic: bool = False
    deterministic_cycle_period: int = 4

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 <= self.stay_probability <= 1.0:
            raise ValueError("stay_probability must be in [0, 1]")
        if not 0.0 <= self.signal_accuracy <= 1.0:
            raise ValueError("signal_accuracy must be in [0, 1]")
        if self.progress_shock_scale < 0.0:
            raise ValueError("progress_shock_scale cannot be negative")
        if self.deterministic_cycle_period <= 0:
            raise ValueError("deterministic_cycle_period must be positive")


@dataclass(frozen=True, slots=True)
class HiddenIntentState:
    step_index: int
    intent: int
    progress: float
    last_action: int | None
    last_response: float
    privileged_signal: int
    shock: float


@dataclass(frozen=True, slots=True)
class HiddenIntentObservation:
    """Released history fields plus one explicitly privileged channel.

    ``privileged_signal`` exists only for logging policies. Released feature
    builders and serialized observation payloads must exclude it; the property
    tests enforce this.
    """

    step_index: int
    last_action: int | None
    last_response: float
    cumulative_progress: float
    privileged_signal: int


@dataclass(frozen=True, slots=True)
class HiddenIntentStepNoise:
    step_index: int
    intent_draw: float
    response_draw: float
    signal_draw: float
    policy_draw: float
    shock_noise: float


@dataclass(frozen=True, slots=True)
class HiddenIntentExogenousNoise(Sequence[HiddenIntentStepNoise]):
    seed: int
    steps: tuple[HiddenIntentStepNoise, ...]

    def __len__(self) -> int:
        return len(self.steps)

    @overload
    def __getitem__(self, index: int) -> HiddenIntentStepNoise: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[HiddenIntentStepNoise, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> HiddenIntentStepNoise | tuple[HiddenIntentStepNoise, ...]:
        return self.steps[index]

    def __iter__(self) -> Iterator[HiddenIntentStepNoise]:
        return iter(self.steps)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _other_intent(current: int, quantile: float) -> int:
    """Map a uniform draw to one of the intents other than ``current``."""

    others = [intent for intent in range(_N_INTENTS) if intent != current]
    index = min(int(quantile * len(others)), len(others) - 1)
    return others[index]


class HiddenIntentWorld:
    """A stateless SCM; every stochastic input is passed to ``step``."""

    action_space: tuple[IntentAction, ...] = _ACTIONS

    def __init__(self, config: HiddenIntentConfig | None = None) -> None:
        self.config = config or HiddenIntentConfig()
        self.horizon = self.config.horizon

    @property
    def is_deterministic(self) -> bool:
        return self.config.deterministic

    def initial_state(self) -> HiddenIntentState:
        return HiddenIntentState(
            step_index=0,
            intent=0,
            progress=0.0,
            last_action=None,
            last_response=0.0,
            privileged_signal=0,
            shock=0.0,
        )

    def observe(self, state: HiddenIntentState) -> HiddenIntentObservation:
        return HiddenIntentObservation(
            step_index=state.step_index,
            last_action=state.last_action,
            last_response=state.last_response,
            cumulative_progress=state.progress,
            privileged_signal=state.privileged_signal,
        )

    def sample_exogenous(self, seed: int) -> HiddenIntentExogenousNoise:
        rng = random.Random(seed)
        steps = tuple(
            HiddenIntentStepNoise(
                step_index=step,
                intent_draw=rng.random(),
                response_draw=rng.random(),
                signal_draw=rng.random(),
                policy_draw=rng.random(),
                shock_noise=rng.gauss(0.0, 1.0),
            )
            for step in range(self.horizon)
        )
        return HiddenIntentExogenousNoise(seed=seed, steps=steps)

    @staticmethod
    def policy_random_value(noise: HiddenIntentStepNoise) -> float:
        return noise.policy_draw

    def _next_intent(self, state: HiddenIntentState, exogenous: HiddenIntentStepNoise) -> int:
        if self.config.deterministic:
            next_step = state.step_index + 1
            if next_step % self.config.deterministic_cycle_period == 0:
                return (state.intent + 1) % _N_INTENTS
            return state.intent
        if exogenous.intent_draw < self.config.stay_probability:
            return state.intent
        remaining = (exogenous.intent_draw - self.config.stay_probability) / max(
            1.0 - self.config.stay_probability, 1.0e-12
        )
        return _other_intent(state.intent, remaining)

    def _next_signal(self, next_intent: int, exogenous: HiddenIntentStepNoise) -> int:
        if self.config.deterministic:
            return next_intent
        if exogenous.signal_draw < self.config.signal_accuracy:
            return next_intent
        remaining = (exogenous.signal_draw - self.config.signal_accuracy) / max(
            1.0 - self.config.signal_accuracy, 1.0e-12
        )
        return _other_intent(next_intent, remaining)

    def step(
        self,
        state: HiddenIntentState,
        action: IntentAction,
        exogenous: HiddenIntentStepNoise,
    ) -> Transition[
        HiddenIntentState,
        IntentAction,
        HiddenIntentObservation,
        HiddenIntentStepNoise,
    ]:
        if state.step_index >= self.horizon:
            raise RuntimeError("cannot step a terminated world")
        if exogenous.step_index != state.step_index:
            raise ValueError("exogenous noise step does not match state")
        try:
            action = IntentAction(action)
            action_index = self.action_space.index(action)
        except ValueError as error:
            raise ValueError(f"unsupported action: {action!r}") from error

        config = self.config
        matched = action_index == state.intent
        response_probability = _sigmoid(config.match_logit if matched else config.mismatch_logit)
        if config.deterministic:
            response = 1.0 if response_probability >= 0.5 else 0.0
        else:
            response = 1.0 if exogenous.response_draw < response_probability else 0.0

        next_intent = self._next_intent(state, exogenous)
        next_signal = self._next_signal(next_intent, exogenous)
        next_step = state.step_index + 1
        terminated = next_step >= self.horizon
        shock = 0.0
        if terminated and not config.deterministic:
            shock = exogenous.shock_noise

        next_state = HiddenIntentState(
            step_index=next_step,
            intent=next_intent,
            progress=state.progress + response,
            last_action=action_index,
            last_response=response,
            privileged_signal=next_signal,
            shock=shock,
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
                ("response", response),
                ("response_probability", response_probability),
                ("intent_matched", float(matched)),
            ),
        )

    def terminal_proxy(self, state: HiddenIntentState) -> float:
        score = state.progress + self.config.progress_shock_scale * state.shock
        return float(score > self.config.proxy_threshold)

    def terminal_utility(self, state: HiddenIntentState) -> float:
        return state.progress

    def rollout(
        self,
        actions: Sequence[IntentAction],
        exogenous: Sequence[HiddenIntentStepNoise],
        *,
        initial_state: HiddenIntentState | None = None,
    ) -> Episode[
        HiddenIntentState,
        IntentAction,
        HiddenIntentObservation,
        HiddenIntentStepNoise,
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
        policy: Policy[HiddenIntentObservation, IntentAction],
        exogenous: Sequence[HiddenIntentStepNoise],
        *,
        initial_state: HiddenIntentState | None = None,
    ) -> Episode[
        HiddenIntentState,
        IntentAction,
        HiddenIntentObservation,
        HiddenIntentStepNoise,
    ]:
        state = initial_state or self.initial_state()
        actions: list[IntentAction] = []
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
class RepeatSuccessPolicy:
    """Clean-regime logging policy with exact released-observation propensities.

    Repeats the last action after a success, otherwise sweeps intents by step
    index. It never reads ``privileged_signal``, so its propensities are a
    function of released history only.
    """

    epsilon: float

    def probabilities(self, observation: HiddenIntentObservation) -> tuple[float, ...]:
        if observation.last_response >= 0.5 and observation.last_action is not None:
            favored = observation.last_action
        else:
            favored = observation.step_index % _N_INTENTS
        probabilities = [self.epsilon / _N_INTENTS for _ in _ACTIONS]
        probabilities[favored] += 1.0 - self.epsilon
        return tuple(probabilities)

    def select_action(
        self,
        observation: HiddenIntentObservation,
        *,
        step_index: int,
        random_value: float,
    ) -> IntentAction:
        del step_index
        cumulative = 0.0
        for action, probability in zip(_ACTIONS, self.probabilities(observation), strict=True):
            cumulative += probability
            if random_value < cumulative:
                return action
        return _ACTIONS[-1]

    def log_probability(
        self,
        observation: HiddenIntentObservation,
        action: IntentAction,
    ) -> float:
        return math.log(self.probabilities(observation)[_ACTIONS.index(action)])


@dataclass(frozen=True, slots=True)
class PrivilegedIntentPolicy:
    """Confounded logging policy that follows the privileged intent signal.

    Its full propensity conditions on ``privileged_signal``, which is hidden
    from the learner, so released-history propensities are not computable in
    closed form and generated datasets must be flagged as confounded.
    """

    epsilon: float

    def probabilities(self, observation: HiddenIntentObservation) -> tuple[float, ...]:
        favored = observation.privileged_signal
        probabilities = [self.epsilon / _N_INTENTS for _ in _ACTIONS]
        probabilities[favored] += 1.0 - self.epsilon
        return tuple(probabilities)

    def select_action(
        self,
        observation: HiddenIntentObservation,
        *,
        step_index: int,
        random_value: float,
    ) -> IntentAction:
        del step_index
        cumulative = 0.0
        for action, probability in zip(_ACTIONS, self.probabilities(observation), strict=True):
            cumulative += probability
            if random_value < cumulative:
                return action
        return _ACTIONS[-1]

    def log_probability_full(
        self,
        observation: HiddenIntentObservation,
        action: IntentAction,
    ) -> float:
        """Full propensity ``log mu(a | h, z-signal)`` for oracle analysis only."""

        return math.log(self.probabilities(observation)[_ACTIONS.index(action)])
