"""Utilities for inspecting fit results by hand by plotting them individually."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.axes._base import _AxesBase

from .utils import simple_ax_grid

__all__ = (
    "plot_fit",
    "plot_fits",
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lmfit as lf
    from numpy.typing import NDArray


def plot_fit(model_result: lf.Model, ax: Axes | None = None) -> None:
    """Performs a straightforward plot of the data, residual, and fit to an axis."""
    if ax is None:
        _, ax = plt.subplots()
    assert isinstance(ax, _AxesBase)
    x = model_result.userkws[model_result.model.independent_vars[0]]
    ax2 = ax.twinx()
    assert isinstance(ax2, Axes)
    ax2.grid(visible=False)
    ax2.axhline(0, color="green", linestyle="--", alpha=0.5)

    ax.scatter(x, model_result.data, s=10, edgecolors="blue", marker="s", c="white", linewidth=1.5)
    ax.plot(x, model_result.best_fit, color="red", linewidth=1.5)

    ax2.scatter(
        x,
        model_result.residual,
        edgecolors="green",
        alpha=0.5,
        s=12,
        marker="s",
        c="white",
        linewidth=1.5,
    )
    ylim = np.max(np.abs(np.asarray(ax2.get_ylim()))) * 1.5
    ax2.set_ylim(bottom=-ylim, top=ylim)
    ax.set_xlim(left=np.min(x), right=np.max(x))


def plot_fits(model_results: list[lf.Model], axs: NDArray[np.object_] | None = None) -> None:
    """Plots several fits onto a grid of axes."""
    n_results = len(model_results)
    axs = axs if axs else simple_ax_grid(n_results, sharex="col", sharey="row")[1]

    for axi, model_result in zip(axs, model_results, strict=False):
        plot_fit(model_result, ax=axi)
