"""Typed, immutable contracts for controlled structural worlds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
ObservationT = TypeVar("ObservationT")
NoiseT = TypeVar("NoiseT")
PolicyObservationT = TypeVar("PolicyObservationT", contravariant=True)
PolicyActionT = TypeVar("PolicyActionT", covariant=True)


@runtime_checkable
class Policy(Protocol[PolicyObservationT, PolicyActionT]):
    """A policy whose randomness is supplied explicitly by the world."""

    def select_action(
        self,
        observation: PolicyObservationT,
        *,
        step_index: int,
        random_value: float,
    ) -> PolicyActionT: ...


@dataclass(frozen=True, slots=True)
class Transition(Generic[StateT, ActionT, ObservationT, NoiseT]):
    """One immutable Gymnasium-like transition."""

    step_index: int
    state: StateT
    observation: ObservationT
    action: ActionT
    exogenous: NoiseT
    next_state: StateT
    next_observation: ObservationT
    reward: float
    terminated: bool
    truncated: bool = False
    info: tuple[tuple[str, float], ...] = ()

    def info_value(self, name: str) -> float:
        for key, value in self.info:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class Episode(Generic[StateT, ActionT, ObservationT, NoiseT]):
    """An immutable rollout, including counterfactual suffix rollouts."""

    initial_state: StateT
    initial_observation: ObservationT
    transitions: tuple[Transition[StateT, ActionT, ObservationT, NoiseT], ...]
    terminal_proxy: float
    terminal_utility: float
    seed: int | None = None

    @property
    def actions(self) -> tuple[ActionT, ...]:
        return tuple(transition.action for transition in self.transitions)

    @property
    def exogenous(self) -> tuple[NoiseT, ...]:
        return tuple(transition.exogenous for transition in self.transitions)

    @property
    def final_state(self) -> StateT:
        if not self.transitions:
            return self.initial_state
        return self.transitions[-1].next_state

    def state_before(self, step_index: int) -> StateT:
        for transition in self.transitions:
            if transition.step_index == step_index:
                return transition.state
        raise IndexError(f"episode has no transition at step {step_index}")


@runtime_checkable
class StructuralWorld(Protocol[StateT, ActionT, ObservationT, NoiseT]):
    """Stateless world API with explicit exogenous randomness."""

    horizon: int
    action_space: tuple[ActionT, ...]

    @property
    def is_deterministic(self) -> bool: ...

    def initial_state(self) -> StateT: ...

    def observe(self, state: StateT) -> ObservationT: ...

    def sample_exogenous(self, seed: int) -> Sequence[NoiseT]: ...

    def policy_random_value(self, noise: NoiseT) -> float: ...

    def step(
        self,
        state: StateT,
        action: ActionT,
        exogenous: NoiseT,
    ) -> Transition[StateT, ActionT, ObservationT, NoiseT]: ...

    def terminal_proxy(self, state: StateT) -> float: ...

    def terminal_utility(self, state: StateT) -> float: ...
