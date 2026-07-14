"""GiGPO-style macro/micro group-relative credit for anchor states.

Follows the published group-in-group construction: macro credit is the
group-relative terminal advantage over episodes that share an initial game;
micro credit is the group-relative discounted return over repeated anchor
states identified by exact state hash. Anchor equality by canonical state
hash is the frozen ALFWorld adaptation; anchors with fewer than two visits
contribute zero micro credit (ordinary grouped credit is never silently
substituted).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from longfeedback.credit.branching import EpisodeRecord

_STD_FLOOR = 1.0e-8


@dataclass(frozen=True, slots=True)
class GigpoSettings:
    discount: float = 0.95
    micro_weight: float = 1.0
    normalize_by_std: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if self.micro_weight < 0.0:
            raise ValueError("micro_weight cannot be negative")


@dataclass(frozen=True, slots=True)
class GigpoStepCredit:
    episode_id: str
    step_index: int
    state_hash: str
    macro_advantage: float
    micro_advantage: float
    combined_advantage: float
    anchor_group_size: int


def _group_relative(values: list[float], *, normalize: bool) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    if not normalize:
        return centered
    variance = sum(value**2 for value in centered) / len(values)
    std = variance**0.5
    if std < _STD_FLOOR:
        # Declared all-success/all-failure convention: zero advantage.
        return [0.0 for _ in values]
    return [value / std for value in centered]


def gigpo_step_credits(
    episodes: Sequence[EpisodeRecord], settings: GigpoSettings
) -> list[GigpoStepCredit]:
    """Macro + micro credits for every step of the provided episode group."""

    macro_groups: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for episode in episodes:
        macro_groups[episode.game.game_id].append(episode)
    macro_advantage: dict[str, float] = {}
    for game_id in sorted(macro_groups):
        group = sorted(macro_groups[game_id], key=lambda episode: episode.episode_id)
        advantages = _group_relative(
            [float(episode.success) for episode in group], normalize=settings.normalize_by_std
        )
        for episode, advantage in zip(group, advantages, strict=True):
            macro_advantage[episode.episode_id] = advantage

    anchor_returns: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for episode in episodes:
        horizon = len(episode.steps)
        for step in episode.steps:
            discounted = float(episode.success) * settings.discount ** (
                horizon - step.step_index - 1
            )
            anchor_returns[step.observation.state_hash].append(
                (episode.episode_id, step.step_index, discounted)
            )

    micro_advantage: dict[tuple[str, int], tuple[float, int]] = {}
    for state_hash in sorted(anchor_returns):
        visits = sorted(anchor_returns[state_hash])
        advantages = _group_relative(
            [value for _, _, value in visits], normalize=settings.normalize_by_std
        )
        for (episode_id, step_index, _), advantage in zip(visits, advantages, strict=True):
            micro_advantage[(episode_id, step_index)] = (advantage, len(visits))

    credits: list[GigpoStepCredit] = []
    for episode in episodes:
        for step in episode.steps:
            micro, group_size = micro_advantage.get((episode.episode_id, step.step_index), (0.0, 1))
            macro = macro_advantage.get(episode.episode_id, 0.0)
            credits.append(
                GigpoStepCredit(
                    episode_id=episode.episode_id,
                    step_index=step.step_index,
                    state_hash=step.observation.state_hash,
                    macro_advantage=macro,
                    micro_advantage=micro,
                    combined_advantage=macro + settings.micro_weight * micro,
                    anchor_group_size=group_size,
                )
            )
    return credits


def anchor_diagnostics(credits: Sequence[GigpoStepCredit]) -> dict[str, float | int]:
    """Anchor availability and effective grouped sample size."""

    total = len(credits)
    grouped = sum(1 for credit in credits if credit.anchor_group_size >= 2)
    return {
        "steps": total,
        "steps_with_anchor_group": grouped,
        "anchor_availability": grouped / total if total else 0.0,
        "mean_anchor_group_size": (
            sum(credit.anchor_group_size for credit in credits) / total if total else 0.0
        ),
    }
