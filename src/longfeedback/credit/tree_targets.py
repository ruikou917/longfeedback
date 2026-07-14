"""Parent-Q / child-V tree-consistency target construction (design section 3.6).

For a forced edge ``(h_t, a, h_{t+1})`` with zero intermediate reward:

- terminal edge: the target is the observed terminal success;
- child with direct unforced rollouts: the target is ``child_v_hat`` weighted
  by its Monte Carlo precision;
- otherwise: a stop-gradient EMA target-network value, filled in by the model
  at training time (never from locked-reference rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from longfeedback.credit.branching import BranchTargetRow

_SE_EPSILON = 1.0e-4


class TreeTargetKind(StrEnum):
    TERMINAL = "terminal"
    CHILD_DIRECT = "child_direct"
    BOOTSTRAP = "bootstrap"


@dataclass(frozen=True, slots=True)
class TreeTarget:
    kind: TreeTargetKind
    value: float
    weight: float


def tree_target_for_row(row: BranchTargetRow) -> TreeTarget:
    """Static part of the tree target; bootstrap values are model-filled."""

    if row.target_role == "locked_reference":
        raise ValueError("locked-reference rows must never produce training tree targets")
    if row.forced_done:
        return TreeTarget(
            kind=TreeTargetKind.TERMINAL, value=row.forced_terminal_success, weight=1.0
        )
    if row.child_unforced_rollout_count > 0:
        precision = 1.0 / (row.child_v_se**2 + _SE_EPSILON)
        return TreeTarget(kind=TreeTargetKind.CHILD_DIRECT, value=row.child_v_hat, weight=precision)
    return TreeTarget(kind=TreeTargetKind.BOOTSTRAP, value=0.0, weight=1.0)
