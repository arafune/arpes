"""Plotting routines for making the classic stacked line plots.

Think the album art for "Unknown Pleasures".
"""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Literal, Unpack

import matplotlib as mpl
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from arpes.analysis import rebin
from arpes.constants import TWO_DIMENSION
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

from .tof import scatter_with_std
from .utils import (
    fancy_labels,
    label_for_dim,
    path_for_plot,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure
    from matplotlib.typing import ColorType, RGBAColorType, RGBColorType
    from numpy.typing import NDArray

    from arpes._typing import LEGENDLOCATION, ColorbarParam, MPLPlotKwargsBasic
__all__ = (
    "stack_dispersion_plot",
    "flat_stack_plot",
    "offset_scatter_plot",
)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


@save_plot_provenance
def offset_scatter_plot(  # noqa: PLR0913
    data: xr.Dataset,
    name_to_plot: str = "",
    stack_axis: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    scale_coordinate: float = 0.5,
    ylim: tuple[float, float] | tuple[()] = (),
    fermi_level: float | None = None,
    loc: LEGENDLOCATION = "upper left",
    figsize: tuple[float, float] = (11, 5),
    *,
    color: Colormap | str = "black",
    aux_errorbars: bool = True,
    **kwargs: Unpack[ColorbarParam],
) -> Path | tuple[Figure | None, Axes]:
    """Makes a stack plot (scatters version).

    Args:
        data(xr.Dataset): _description_
        name_to_plot(str): name of the spectrum (in many case 'spectrum' is set), by default ""
        stack_axis(str): _description_, by default ""
        ax(Axes | None):  _description_, by default None
        out(str | Path):  _description
        scale_coordinate(float):  _description_, by default 0.5
        ylim(tuple[float, float]):  _description_, by default ()
        fermi_level(float | None): Value corresponds the Fermi level to draw the line,
            by default None (not drawn)
        figsize (tuple[float, float]) : figure size. Used in plt.subplots
        loc: Legend Location
        color: Colormap
        aux_errorbars(bool):  _description_, by default True
        kwargs: kwargs passing to args of Colorbar

    Returns:
        Path | tuple[Figure | None, Axes]: _description_

    Raises:
        ValueError
    """
    assert isinstance(data, xr.Dataset)

    if not name_to_plot:
        var_names = [k for k in data.data_vars if "_std" not in str(k)]  # => ["spectrum"]
        assert len(var_names) == 1
        name_to_plot = str(var_names[0])
        assert (name_to_plot + "_std") in data.data_vars, "Has 'mean_and_deviation' been applied?"

    msg = "In order to produce a stack plot, data must be image-like."
    msg += "Passed data included dimensions:"
    msg += f" {data.data_vars[name_to_plot].dims}"
    assert len(data.data_vars[name_to_plot].dims) == TWO_DIMENSION, msg

    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    inset_ax = inset_axes(ax, width="40%", height="5%", loc=loc)

    assert isinstance(ax, Axes)

    stack_axis = stack_axis if stack_axis else str(data.data_vars[name_to_plot].dims[0])

    skip_colorbar = True
    other_dim = next(str(d) for d in data.dims if d != stack_axis)

    if "eV" in data.dims and stack_axis != "eV" and fermi_level is not None:
        ax.axhline(fermi_level, linestyle="--", color="red")
        ax.fill_betweenx([-1e6, 1e6], 0, 0.2, color="black", alpha=0.07)

    if not ylim:
        ax.set_ylim(auto=True)
    else:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ylim = ax.get_ylim()

    # real plotting here
    for i, (_, value) in enumerate(data.G.iterate_axis(stack_axis)):
        delta = data.G.stride(generic_dim_names=False)[other_dim]
        data_for = value.copy(deep=True)
        data_for.coords[other_dim].values -= i * delta * scale_coordinate / 10

        scatter_with_std(
            data_for,
            name_to_plot,
            ax=ax,
            color=_color_for_plot(color, i, len(data.coords[stack_axis])),
        )

        if aux_errorbars:
            data_for = data_for.copy(deep=True)
            flattened = data_for.data_vars[name_to_plot].copy(deep=True)
            flattened.values = ylim[0] * np.ones(flattened.values.shape)
            data_for = data_for.assign(**{name_to_plot: flattened})
            scatter_with_std(
                data_for,
                name_to_plot,
                ax=ax,
                color=_color_for_plot(color, i, len(data.coords[stack_axis])),
            )

    ax.set_xlabel(other_dim)
    ax.set_ylabel(name_to_plot)
    fancy_labels(ax)
    kwargs = _set_default_kwags(kwargs, data=data, stack_axis=stack_axis)

    if isinstance(color, Colormap):
        kwargs.setdefault("cmap", color)
    if inset_ax and not skip_colorbar:
        inset_ax.set_xlabel(stack_axis, fontsize=16)
        fancy_labels(inset_ax)
        matplotlib.colorbar.Colorbar(
            inset_ax,
            **kwargs,
        )

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


def _set_default_kwags(
    kwargs: ColorbarParam,
    data: xr.Dataset,
    stack_axis: str,
) -> ColorbarParam:
    kwargs.setdefault("orientation", "horizontal")
    kwargs.setdefault(
        "label",
        label_for_dim(data, stack_axis),
    )
    kwargs.setdefault(
        "norm",
        matplotlib.colors.Normalize(
            vmin=data.coords[stack_axis].min().item(),
            vmax=data.coords[stack_axis].max().item(),
        ),
    )
    kwargs.setdefault("ticks", matplotlib.ticker.MaxNLocator(2))
    return kwargs


@save_plot_provenance
def flat_stack_plot(  # noqa: PLR0913
    data: xr.DataArray,
    *,
    stack_axis: str = "",
    ax: Axes | None = None,
    mode: Literal["line", "scatter"] = "line",
    fermi_level: float | None = None,
    figsize: tuple[float, float] = (7, 5),
    title: str = "",
    max_stacks: int = 200,
    out: str | Path = "",
    loc: LEGENDLOCATION = "upper left",
    **kwargs: Unpack[MPLPlotKwargsBasic],
) -> Path | tuple[Figure | None, Axes]:
    """Generates a stack plot with all the lines distinguished by color rather than offset.

    Args:
        data(DataType): ARPES data (xr.DataArray is prepfered)
        stack_axis(str): axis for stacking, by default ""
        ax (Axes | None): matplotlib Axes, by default None
        mode(Literal["line", "scatter"]): plot style (line/scatter), by default "line"
        fermi_level(float|None): Value of the Fermi level to Draw the line, by default None.
                                 (Not drawn)
        figsize (tuple[float, float]): figure size
        title(str): Title string, by default ""
        max_stacks(int): maximum number of the staking spectra
        out(str | Path): Path to the figure, by default ""
        loc: Legend location
        **kwargs: pass to subplot if figsize is set, and ticks is set, and the others to be passed
                  ax.plot

    Returns:
        Path | tuple[Figure | None, Axes]

    Raises:
        ValueError
            _description_
        NotImplementedError
            _description_
    """
    data = _rebinning(
        data,
        stack_axis=stack_axis,
        max_stacks=max_stacks,
        method="mean",
    )[0]

    if len(data.dims) != TWO_DIMENSION:
        msg = "In order to produce a stack plot, data must be image-like."
        msg += f"Passed data included dimensions: {data.dims}"
        raise ValueError(
            msg,
        )

    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax_inset = inset_axes(ax, width="40%", height="5%", loc=loc)
    assert isinstance(ax, Axes)
    if not stack_axis:
        stack_axis = str(data.dims[0])

    horizontal_dim = next(str(d) for d in data.dims if d != stack_axis)
    horizontal = data.coords[horizontal_dim]

    if "eV" in data.dims and stack_axis != "eV" and fermi_level is not None:
        ax.axvline(
            fermi_level,
            color="red",
            alpha=0.8,
            linestyle="--",
            linewidth=1,
        )

    color = kwargs.pop("color", "viridis")

    for i, (_, marginal) in enumerate(data.G.iterate_axis(stack_axis)):
        if mode == "line":
            ax.plot(
                horizontal,
                marginal.values,
                color=_color_for_plot(
                    color,
                    i,
                    len(data.coords[stack_axis]),
                ),
                **kwargs,
            )
        else:
            assert mode == "scatter"
            ax.scatter(
                horizontal,
                marginal.values,
                color=_color_for_plot(
                    color,
                    i,
                    len(data.coords[stack_axis]),
                ),
                **kwargs,
            )
    matplotlib.colorbar.Colorbar(
        ax_inset,
        orientation="horizontal",
        label=label_for_dim(data, stack_axis),
        norm=matplotlib.colors.Normalize(
            vmin=data.coords[stack_axis].min().values,
            vmax=data.coords[stack_axis].max().values,
        ),
        ticks=matplotlib.ticker.MaxNLocator(2),
        cmap=color,
    )
    ax.set_xlabel(label_for_dim(data, horizontal_dim))
    ax.set_ylabel("Spectrum Intensity (arb).")
    ax.set_title(title, fontsize=14)
    ax.set_xlim(left=horizontal.min().item(), right=horizontal.max().item())

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


@save_plot_provenance
def stack_dispersion_plot(  # noqa: PLR0913
    data: xr.DataArray,
    *,
    stack_axis: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    max_stacks: int = 100,
    scale_factor: float = 0,
    mode: Literal["line", "fill_between", "hide_line", "scatter"] = "line",
    offset_correction: Literal["zero", "constant", "constant_right"] | None = "zero",
    shift: float = 0,
    negate: bool = False,
    figsize: tuple[float, float] = (7, 7),
    title: str = "",
    **kwargs: Unpack[MPLPlotKwargsBasic],
) -> Path | tuple[Figure | None, Axes]:
    """Generates a stack plot with all the lines distinguished by offset (and color).

    Args:
        data(XrTypes): ARPES data
        stack_axis(str): stack axis. e.g. "phi" , "eV", ...
        ax(Axes): matplotlib Axes object
        out(str | Path): Path for output figure
        max_stacks(int): maximum number of the stacking spectra
        scale_factor(float): scale factor
        mode(Literal["liine", "fill_between", "hide_line", "scatter"]): Draw mode
        offset_correction(Literal["zero", "constant", "constant_right"] | None): offset correction
                                                                                 mode (default to
                                                                                 "zero")
        shift(float): shift of the plot along the horizontal direction
        figsize (tuple[float, float]): figure size, default is (7,7)
        title (str, optional): title of figure
        negate(bool): _description_
        **kwargs:
            Passed to ax.plot / fill_between. Can set linewidth etc., here.
            (See _typing/MPLPlotKwagsBasic)
    """
    data_arr, stack_axis, other_axis = _rebinning(
        data,
        stack_axis=stack_axis,
        max_stacks=max_stacks,
    )

    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    assert isinstance(ax, Axes)
    if not title:
        title = f"{data_arr.S.label.replace('_', ' ')} Stack"
    assert isinstance(title, str)
    max_intensity_over_stacks = np.nanmax(data_arr.values)

    cvalues: NDArray[np.float_] = data_arr.coords[other_axis].values

    if not scale_factor:
        scale_factor = _scale_factor(
            data_arr,
            stack_axis=stack_axis,
            offset_correction=offset_correction,
            negate=negate,
        )

    iteration_order = -1  # might need to fiddle with this in certain cases
    lim = [np.inf, -np.inf]

    color = kwargs.pop("color", "black")
    for i, (coord_dict, marginal) in enumerate(
        list(data_arr.G.iterate_axis(stack_axis))[::iteration_order],
    ):
        coord_value = coord_dict[stack_axis]
        ys = _y_shifted(
            offset_correction=offset_correction,
            coord_value=coord_value,
            marginal=marginal,
            scale_parameters=(scale_factor, max_intensity_over_stacks, negate),
        )

        xs = cvalues - i * shift

        lim = [min(lim[0], float(np.min(xs))), max(lim[1], float(np.max(xs)))]
        if mode == "line":
            ax.plot(
                xs,
                ys,
                color=_color_for_plot(color, i, len(data_arr.coords[stack_axis])),
                **kwargs,
            )
        elif mode == "hide_line":
            ax.plot(
                xs,
                ys,
                color=_color_for_plot(color, i, len(data_arr.coords[stack_axis])),
                **kwargs,
                zorder=i * 2 + 1,
            )
            ax.fill_between(xs, ys, coord_value, color="white", alpha=1, zorder=i * 2, **kwargs)
        elif mode == "fill_between":
            ax.fill_between(
                xs,
                ys,
                coord_value,
                color=_color_for_plot(color, i, len(data_arr.coords[stack_axis])),
                alpha=1,
                zorder=i * 2,
                **kwargs,
            )
        else:
            ax.scatter(
                xs,
                ys,
                color=_color_for_plot(color, i, len(data_arr.coords[stack_axis])),
                **kwargs,
            )

    x_label = other_axis
    y_label = stack_axis

    yticker = matplotlib.ticker.MaxNLocator(5)
    y_tick_region = [
        i
        for i in yticker.tick_values(
            data_arr.coords[stack_axis].min().values,
            data_arr.coords[stack_axis].max().values,
        )
        if (
            i > data_arr.coords[stack_axis].min().values
            and i < data_arr.coords[stack_axis].max().values
        )
    ]

    ax.set_yticks(np.array(y_tick_region))
    ax.set_ylabel(label_for_dim(data_arr, y_label))
    ylims = ax.get_ylim()
    median_along_stack_axis = y_tick_region[2]

    ax.yaxis.set_label_coords(
        -0.09,
        1 / (ylims[1] - ylims[0]) * (median_along_stack_axis - ylims[0]),
    )

    ax.set_xlabel(label_for_dim(data_arr, x_label))
    # set xlim with margin
    # 11/10 is the good value for margine
    axis_min, axis_max = min(lim), max(lim)
    middle = (axis_min + axis_max) / 2
    ax.set_xlim(
        left=middle - (axis_max - axis_min) / 2 * 11 / 10,
        right=middle + (axis_max - axis_min) / 2 * 11 / 10,
    )

    ax.set_title(title)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


def _y_shifted(
    offset_correction: Literal["zero", "constant", "constant_right"] | None,
    marginal: xr.DataArray,
    coord_value: NDArray[np.float_],
    scale_parameters: tuple[float, float, bool],
) -> NDArray[np.float_]:
    scale_factor = scale_parameters[0]
    max_intensity_over_stacks = scale_parameters[1]
    negate = scale_parameters[2]

    marginal_values = -marginal.values if negate else marginal.values
    marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

    if offset_correction == "zero":
        true_ys = marginal_values / max_intensity_over_stacks
    elif offset_correction == "constant":
        true_ys = (marginal_values - marginal_offset) / max_intensity_over_stacks
    elif offset_correction == "constant_right":
        true_ys = (marginal_values - right_marginal_offset) / max_intensity_over_stacks
    else:  # is this procedure phyically correct?
        true_ys = (
            marginal_values
            - np.linspace(marginal_offset, right_marginal_offset, len(marginal_values))
        ) / max_intensity_over_stacks
    return scale_factor * true_ys + coord_value


def _scale_factor(
    data_arr: xr.DataArray,
    stack_axis: str,
    *,
    offset_correction: Literal["zero", "constant", "constant_right"] | None = "zero",
    negate: bool = False,
) -> float:
    """Determine the scale factor."""
    maximum_deviation = -np.inf

    for _, marginal in data_arr.G.iterate_axis(stack_axis):
        marginal_values = -marginal.values if negate else marginal.values
        marginal_offset, right_marginal_offset = marginal_values[0], marginal_values[-1]

        if offset_correction == "zero":
            true_ys = marginal_values
        elif offset_correction is not None and offset_correction.startswith("constant"):
            true_ys = marginal_values - marginal_offset
        else:
            true_ys = marginal_values - np.linspace(
                marginal_offset,
                right_marginal_offset,
                len(marginal_values),
            )

        maximum_deviation = np.max([maximum_deviation, *list(np.abs(true_ys))])

    return float(
        10
        * (data_arr.coords[stack_axis].max().values - data_arr.coords[stack_axis].min().values)
        / maximum_deviation,
    )


def _rebinning(
    data: xr.DataArray,
    stack_axis: str,
    max_stacks: int,
    method: Literal["sum", "mean"] = "sum",
) -> tuple[xr.DataArray, str, str]:
    """Preparation for stack plot.

    1. rebinning
    2. determine the stack axis
    3. determine the name of the other.
    """
    data_arr = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(data_arr, xr.DataArray)
    assert len(data_arr.dims) == TWO_DIMENSION
    if not stack_axis:
        stack_axis = str(data_arr.dims[0])

    other_axes = list(data_arr.dims)
    other_axes.remove(stack_axis)
    horizontal_axis = str(other_axes[0])

    stack_coord: xr.DataArray = data_arr.coords[stack_axis]
    if len(stack_coord.values) > max_stacks:
        return (
            rebin(
                data_arr,
                bin_width={stack_axis: int(np.ceil(len(stack_coord.values) / max_stacks))},
                method=method,
            ),
            stack_axis,
            horizontal_axis,
        )
    return data_arr, stack_axis, horizontal_axis


def _color_for_plot(
    color: Colormap | ColorType,
    i: int,
    num_plot: int,
) -> RGBAColorType | RGBColorType:
    if isinstance(color, Colormap):
        cmap = color
        return cmap(np.abs(i / num_plot))
    if isinstance(color, str):
        try:
            cmap = mpl.colormaps[color]
            return cmap(np.abs(i / num_plot))
        except KeyError:  # not in the colormap name, assume the color name
            return color
    return color  # color is tuple representing the color
