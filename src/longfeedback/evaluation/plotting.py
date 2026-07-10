"""Small non-interactive E0 plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


def plot_outcome_vs_credit(
    outcome_scores: ArrayLike,
    credit_scores: ArrayLike,
    path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    outcome_label: str = "Outcome score",
    credit_label: str = "Credit score",
    title: str = "Outcome accuracy vs. credit recovery",
) -> Path:
    """Save a deterministic scatter plot without an interactive backend."""

    outcomes = np.asarray(outcome_scores, dtype=np.float64).reshape(-1)
    credits = np.asarray(credit_scores, dtype=np.float64).reshape(-1)
    if outcomes.shape != credits.shape or outcomes.size == 0:
        raise ValueError("outcome_scores and credit_scores must be equal-length, non-empty arrays")
    if not np.all(np.isfinite(outcomes)) or not np.all(np.isfinite(credits)):
        raise ValueError("plot scores must contain only finite values")
    if labels is not None and len(labels) != outcomes.size:
        raise ValueError("labels must match the number of plotted points")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6.0, 4.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(1, 1, 1)
    axes.scatter(outcomes, credits, color="#35618f", edgecolor="white", linewidth=0.6)
    if labels is not None:
        for x_value, y_value, label in zip(outcomes, credits, labels, strict=True):
            axes.annotate(
                str(label),
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
            )
    axes.set_xlabel(outcome_label)
    axes.set_ylabel(credit_label)
    axes.set_title(title)
    axes.grid(alpha=0.25)
    axes.margins(x=0.18, y=0.12)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, metadata={"Software": "LongFeedback"})
    figure.clear()
    return output
