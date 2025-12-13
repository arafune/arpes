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


def apply_dark_to_colorbar(
    cbar: Colorbar,
    *,
    transparent: bool = True,
) -> None:
    """Force colorabar element to dark-mode styling.

    Args:
        cbar: Colorbar for dark mode.
        transparent: if True, set figure background to 'none'.
    """
    if cbar.outline.get_visible():
        cbar.outline.set_edgecolor("white")
        if transparent:
            cbar.outline.set_facecolor("none")
        else:
            cbar.outline.set_facecolor("black")

    cbar.ax.tick_params(colors="white", which="both")

    for label in cbar.ax.get_yticklabels() + cbar.ax.get_yticklabels():
        label.set_color("white")

    if cbar.ax.xaxis.label.get_text():
        cbar.ax.xaxis.label.set_color("white")
    if cbar.ax.yaxis.label.get_text():
        cbar.ax.yaxis.label.set_color("white")
    if transparent:
        cbar.ax.set_facecolor("none")
    else:
        cbar.ax.set_facecolor("black")


def apply_dark_to_ax(
    ax: Axes,
    *,
    transparent: bool = True,
) -> None:
    """Apply dark mode to a single Axes.

    Args:
        ax: Axes for dark mode.
        transparent: if True, set figure background to 'none'.
    """
    if transparent:
        ax.set_facecolor("none")
    else:
        ax.set_facecolor("black")

    ax.tick_params(colors="white", which="both")

    for spine in ax.spines.values():
        spine.set_color("white")

    if ax.get_title():
        ax.set_title(ax.get_title(), color="white")

    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


def apply_dark_to_figure(
    fig: Figure,
    *,
    transparent: bool = True,
) -> None:
    """Apply dark mode to all Axes and Colorbars in the Figure.

    Set the figure background to tranparent "none".

    Args:
        fig: optional Figure to update Axes and Colorbars for dark mode.
        transparent: if True, set figure background to 'none'.
    """
    if transparent:
        fig.patch.set_facecolor("none")
    else:
        fig.patch.set_facecolor("black")

    for ax in fig.get_axes():
        apply_dark_to_ax(ax, transparent=transparent)
    for cbar in get_colorbars(fig):
        apply_dark_to_colorbar(cbar, transparent=transparent)


def get_dark_mode_params(
    overrides: Mapping[RcParamKey, str] | None = None,
    *,
    transparent: bool = True,
) -> dict[RcParamKey, str]:
    """Return a safe copy of the dark-mode rcParams.

    Args:
        overrides: Optional dict of rcParams to override defaults.
        transparent: if True, set figure background to 'none'.
    """
    params = DEFAULT_DARK_MODE.copy()
    if not transparent:
        params["axes.facecolor"] = "black"
        params["figure.facecolor"] = "black"
        params["savefig.facecolor"] = "black"
    if overrides:
        params.update(overrides)
    return params


@contextmanager
def dark_background(
    overrides: Mapping[RcParamKey, str] | None = None,
    fig: Figure | None = None,
    *,
    transparent: bool = True,
) -> Iterator[None]:
    """Apply dark-mode rcParams temporarily.

    Optionally updates an Axes and Colorbars of Figure for dark mode.

    Args:
        overrides: Optional dict of rcParams to override defaults.
        fig: optional Figure to update Axes and Colorbars for dark mode.
        transparent: if True, set figure background to 'none'.
    """
    params = get_dark_mode_params(overrides, transparent=transparent)
    with plt.rc_context(cast("dict[str, object]", params)):
        fig = plt.gcf() if fig is None else fig

        yield

        apply_dark_to_figure(fig, transparent=transparent)
