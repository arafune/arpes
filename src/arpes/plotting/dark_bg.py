"""Module for contextmanager for dark background."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Literal, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from .utils import get_colorbars

__all__ = ("dark_background",)

# Only actual rcParam keys (dot separated)
RcParamKey = Literal[
    "axes.edgecolor",
    "xtick.color",
    "ytick.color",
    "axes.facecolor",
    "text.color",
    "figure.facecolor",
    "savefig.facecolor",
    "grid.color",
]


DEFAULT_DARK_MODE: dict[RcParamKey, str] = {
    "axes.edgecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.facecolor": "none",
    "text.color": "white",
    "figure.facecolor": "none",
    "savefig.facecolor": "none",
    "grid.color": "gray",
}


def apply_dark_to_colorbar(cbar: Colorbar) -> None:
    """Force colorabar element to dark-mode styling."""
    if cbar.outline:
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_facecolor("none")

    cbar.ax.tick_params(colors="white", which="both")

    for label in cbar.ax.get_yticklabels() + cbar.ax.get_yticklabels():
        label.set_color("white")

    if cbar.ax.xaxis.label:
        cbar.ax.xaxis.label.set_color("white")
    if cbar.ax.yaxis.label:
        cbar.ax.yaxis.label.set_color("white")
    cbar.ax.set_facecolor("none")


def apply_dark_to_ax(ax: Axes) -> None:
    ax.set_facecolor("none")

    ax.tick_params(colors="white", which="both")

    for spine in ax.spines.values():
        spine.set_color("white")

    if ax.get_title():
        ax.set_title(ax.get_title(), color="white")

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def apply_dark_to_figure(fig: Figure) -> None:
    """Apply dark mode to all Axes and Colorbars in the Figure.

    Set the figure background to tranparent "none".
    """
    fig.patch.set_facecolor("none")

    for ax in fig.get_axes():
        apply_dark_to_ax(ax)
    for cbar in get_colorbars(fig):
        apply_dark_to_colorbar(cbar)


def get_dark_mode_params(
    overrides: Mapping[RcParamKey, str] | None = None,
) -> dict[RcParamKey, str]:
    """Return a safe copy of the dark-mode rcParams."""
    params = DEFAULT_DARK_MODE.copy()
    if overrides:
        params.update(overrides)
    return params


@contextmanager
def dark_background(
    overrides: Mapping[RcParamKey, str] | None = None,
    fig: Figure | None = None,
) -> Iterator[None]:
    """Apply dark-mode rcParams temporarily.

    Optionally updates an Axes and Colorbars of Figure for dark mode.

    Args:
        overrides: Optional dict of rcParams to override defaults.
        fig: optional Figure to update Axes and Colorbars for dark mode.
    """
    params = get_dark_mode_params(overrides)
    with plt.rc_context(cast("dict[str, object]", params)):
        fig = plt.gcf() if fig is None else fig

        yield

        apply_dark_to_figure(fig)
