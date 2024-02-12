"""Annotations onto plots for experimental conditions or locations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Unpack

import matplotlib as mpl
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from arpes.constants import TWO_DIMENSION

from .utils import name_for_dim, unit_for_dim

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from arpes._typing import EXPERIMENTINFO, DataType, MPLTextParam

__all__ = (
    "annotate_cuts",
    "annotate_point",
    "annotate_experimental_conditions",
)


# TODO @<R.Arafune>: Useless: Revision required
# * In order not to use data axis, set transform = ax.Transform
def annotate_experimental_conditions(
    ax: Axes,
    data: DataType,
    desc: list[str | float] | float | str,
    *,
    show: bool = False,
    orientation: Literal["top", "bottom"] = "top",
    **kwargs: Unpack[MPLTextParam],
) -> None:
    """Renders information about the experimental conditions onto a set of axes.

    Also adjust the axes limits and hides the axes.

    data should be the dataset described, and desc should be one of

    'temp',
    'photon',
    'photon polarization',
    'polarization',

    or a number to act as a spacer in units of the axis coordinates

    or a list of such items.
    """
    if isinstance(desc, str | int | float):
        desc = [desc]

    ax.grid(visible=False)
    ax.set_ylim(bottom=0, top=100)
    ax.set_xlim(left=0, right=100)
    if not show:
        ax.set_axis_off()
        ax.patch.set_alpha(0)

    delta: float = -1
    current = 100.0
    if orientation == "bottom":
        delta = 1
        current = 0

    fontsize_keyword: (
        float
        | Literal[
            "xx-small",
            "x-small",
            "small",
            "medium",
            "large",
            "x-large",
            "xx-large",
            "smaller",
        ]
    ) = kwargs.get("fontsize", 16)
    if isinstance(fontsize_keyword, float):
        fontsize = fontsize_keyword
    elif fontsize_keyword in (
        "xx-small",
        "x-small",
        "small",
        "medium",
        "large",
        "x-large",
        "xx-large",
        "smaller",
    ):
        font_scalings = {  # see matplotlib.font_manager
            "xx-small": 0.579,
            "x-small": 0.694,
            "small": 0.833,
            "medium": 1.0,
            "large": 1.200,
            "x-large": 1.440,
            "xx-large": 1.728,
            "larger": 1.2,
            "smaller": 0.833,
        }
        fontsize = mpl.rc_params()["font.size"] * font_scalings[fontsize_keyword]
    else:
        err_msg = "Incorrect font size setting"
        raise RuntimeError(err_msg)
    delta = fontsize * delta

    conditions: EXPERIMENTINFO = data.S.experimental_conditions

    renderers = {
        "temp": lambda c: "\\textbf{T = " + "{:.3g}".format(c["temp"]) + " K}",
        "photon": _render_photon,
        "hv": _render_photon,
        "photon polarization": lambda c: _render_photon(c) + ", " + _render_polarization(c),
        "polarization": _render_polarization,
    }

    for item in desc:
        if isinstance(item, float):
            current += item + delta
            continue

        item_replaced = item.replace("_", " ").lower()

        ax.text(0, current, renderers[item_replaced](conditions), **kwargs)
        current += delta


def _render_polarization(conditions: dict[str, str]) -> str:
    pol = conditions["polarization"]
    if pol in ["lc", "rc"]:
        return "\\textbf{" + pol.upper() + "}"

    symbol_pol: dict[str, str] = {
        "s": "",
        "p": "",
        "s-p": "",
        "p-s": "",
    }

    prefix = ""
    if pol in ["s-p", "p-s"]:
        prefix = "\\textbf{Linear Dichroism, }"

    symbol = symbol_pol[pol]
    if symbol:
        return prefix + "$" + symbol + "$/\\textbf{" + pol + "}"

    return prefix + "\\textbf{" + pol + "}"


def _render_photon(c: dict[str, float]) -> str:
    return "\\textbf{" + str(c["hv"]) + " eV}"


def annotate_cuts(
    ax: Axes,
    data: DataType,
    plotted_axes: NDArray[np.object_],
    *,
    include_text_labels: bool = False,
    **kwargs: tuple | list | NDArray[np.float_],
) -> None:
    """Annotates a cut location onto a plot.

    Example:
        >>> annotate_cuts(ax, conv, ['kz', 'ky'], hv=80)  # doctest: +SKIP

    Args:
        ax: The axes to plot onto
        data: The original data
        plotted_axes: The dimension names which were plotted
        include_text_labels: Whether to include text labels
        kwargs: Defines the coordinates of the cut location
    """
    from arpes.utilities.conversion.forward import convert_coordinates_to_kspace_forward

    converted_coordinates = convert_coordinates_to_kspace_forward(data)
    assert converted_coordinates, xr.Dataset | xr.DataArray
    assert len(plotted_axes) == TWO_DIMENSION

    for k, v in kwargs.items():
        selected = converted_coordinates.sel({k: v}, method="nearest")

        for coords_dict, obj in selected.G.iterate_axis(k):
            css = [obj[d].values for d in plotted_axes]
            ax.plot(*css, color="red", ls="--", linewidth=1, dashes=(5, 5))

            if include_text_labels:
                idx = np.argmin(css[1])

                ax.text(
                    css[0][idx] + 0.05,
                    css[1][idx],
                    f"{name_for_dim(k)} = {coords_dict[k].item()} {unit_for_dim(k)}",
                    color="red",
                    size="medium",
                )


def annotate_point(
    ax: Axes | Axes3D,
    location: Sequence[float],
    delta: tuple[float, ...] = (),
    **kwargs: Unpack[MPLTextParam],
) -> None:
    """Annotates a point or high symmetry location into a plot."""
    if "label" in kwargs:
        label = {
            "G": "$\\Gamma$",
            "X": r"\textbf{X}",
            "Y": r"\textbf{Y}",
            "K": r"\textbf{K}",
            "M": r"\textbf{M}",
        }.get(kwargs["label"], "")
        kwargs.pop("label")

    if not delta:
        delta = (
            -0.05,
            0.05,
        )
    if "color" not in kwargs:
        kwargs["color"] = "red"

    if len(delta) == TWO_DIMENSION:
        assert isinstance(ax, Axes)
        dx, dy = tuple(delta)
        pos_x, pos_y = tuple(location)
        ax.plot([pos_x], [pos_y], "o", c=kwargs["color"])
        ax.text(pos_x + dx, pos_y + dy, s=label, **kwargs)
    else:
        assert isinstance(ax, Axes3D)
        dx, dy, dz = tuple(delta)
        pos_x, pos_y, pos_z = tuple(location)
        ax.plot([pos_x], [pos_y], [pos_z], "o", c=kwargs["color"])
        ax.text(pos_x + dx, pos_y + dy, pos_z + dz, label, **kwargs)
