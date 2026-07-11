"""Small non-interactive experiment plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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


def plot_optimization_curves(
    curves_by_variant: Mapping[str, Mapping[str, Sequence[float]]],
    budgets: Sequence[int],
    path: str | Path,
    *,
    series_labels: Mapping[str, str],
    title: str = "Reward overoptimization curves",
) -> Path:
    """One panel per reward variant; named series versus optimization budget."""

    if not curves_by_variant:
        raise ValueError("curves_by_variant must be non-empty")
    budget_array = np.asarray(budgets, dtype=np.float64)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    variants = list(curves_by_variant)
    figure = Figure(figsize=(4.6 * len(variants), 3.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    colors = {"learned_reward": "#35618f", "observed_proxy": "#c58a2c", "true_utility": "#3d8f5f"}
    axes_list: Any = figure.subplots(1, len(variants), sharex=True)
    if len(variants) == 1:
        axes_list = [axes_list]
    for axes, variant in zip(axes_list, variants, strict=True):
        for series_name, label in series_labels.items():
            values = np.asarray(curves_by_variant[variant][series_name], dtype=np.float64)
            if values.shape != budget_array.shape:
                raise ValueError(f"series {series_name!r} must align with budgets")
            axes.plot(
                budget_array,
                values,
                label=label,
                color=colors.get(series_name),
                linewidth=1.6,
            )
        axes.set_title(variant)
        axes.set_xlabel("optimizer updates")
        axes.grid(alpha=0.25)
    axes_list[0].set_ylabel("value")
    axes_list[0].legend(loc="best", fontsize=8)
    figure.suptitle(title)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, metadata={"Software": "LongFeedback"})
    figure.clear()
    return output
