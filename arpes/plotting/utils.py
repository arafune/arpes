"""Contains many common utility functions for managing matplotlib."""

from __future__ import annotations

import contextlib
import datetime
import errno
import itertools
import json
import pickle
import re
import warnings
from collections import Counter
from collections.abc import Generator, Hashable, Iterable, Iterator, Sequence
from datetime import UTC
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Unpack

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colorbar, colors, gridspec
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, colorConverter
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from arpes import VERSION
from arpes._typing import IMshowParam, XrTypes
from arpes.config import CONFIG, SETTINGS, attempt_determine_workspace, is_using_tex
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.jupyter import get_notebook_name, get_recent_history

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete
    from lmfit.model import Model
    from matplotlib.font_manager import FontProperties
    from matplotlib.image import AxesImage
    from matplotlib.typing import ColorType, RGBAColorType, RGBColorType
    from numpy.typing import NDArray

    from arpes._typing import ColorbarParam, DataType, MPLPlotKwargs, PLTSubplotParam, XrTypes

__all__ = (
    # General + IO
    "path_for_plot",
    "path_for_holoviews",
    "name_for_dim",
    "unit_for_dim",
    "load_data_for_figure",
    "savefig",
    "AnchoredHScaleBar",
    "calculate_aspect_ratio",
    # context managers
    "dark_background",
    # color related
    "temperature_colormap",
    "polarization_colorbar",
    "temperature_colormap_around",
    "temperature_colorbar",
    "temperature_colorbar_around",
    "generic_colorbarmap",
    "generic_colorbarmap_for_data",
    "colorbarmaps_for_axis",
    # Axis generation
    "dos_axes",
    "simple_ax_grid",
    # matplotlib 'macros'
    "invisible_axes",
    "no_ticks",
    "get_colorbars",
    "remove_colorbars",
    "frame_with",
    "unchanged_limits",
    "imshow_arr",
    "imshow_mask",
    "lineplot_arr",  # 1D version of imshow_arr
    "plot_arr",  # generic dimension version of imshow_arr, plot_arr
    # insets related
    "inset_cut_locator",
    "swap_xaxis_side",
    "swap_yaxis_side",
    "swap_axis_sides",
    # units related
    "data_to_axis_units",
    "axis_to_data_units",
    "daxis_ddata_units",
    "ddata_daxis_units",
    # TeX related
    "quick_tex",
    "latex_escape",
    # Decorating + labeling
    "label_for_colorbar",
    "label_for_dim",
    "label_for_symmetry_point",
    "sum_annotation",
    "mean_annotation",
    "fancy_labels",
    "mod_plot_to_ax",
    # Data summaries
    "summarize",
    "transform_labels",
    "v_gradient_fill",
    "h_gradient_fill",
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


TwoDimensional = 2


@contextlib.contextmanager
def unchanged_limits(ax: Axes) -> Iterator[None]:
    """Context manager that retains axis limits."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    yield

    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])


def mod_plot_to_ax(
    data_arr: xr.DataArray,
    ax: Axes,
    mod: Model,
    **kwargs: Unpack[MPLPlotKwargs],
) -> None:
    """Plots a model onto an axis using the data range from the passed data.

    Args:
        data_arr (xr.DataArray): ARPES data
        ax (Axes): matplotlib Axes object
        mod (lmfit.model.Model): Fitting model function
        **kwargs(): pass to "ax.plot"
    """
    assert isinstance(data_arr, xr.DataArray)
    assert isinstance(ax, Axes)
    with unchanged_limits(ax):
        xs: NDArray[np.float_] = data_arr.coords[data_arr.dims[0]].values
        ys: NDArray[np.float_] = mod.eval(x=xs)
        ax.plot(xs, ys, **kwargs)


class GradientFillParam(IMshowParam):
    step: Literal["pre", "mid", "post", None]


def h_gradient_fill(
    x1: float,
    x2: float,
    x_solid: float | None,
    fill_color: RGBColorType = "red",
    ax: Axes | None = None,
    **kwargs: Unpack[GradientFillParam],
) -> AxesImage:  # <== checkme!
    """Fills a gradient between x1 and x2.

    If x_solid is not None, the gradient will be extended
    at the maximum opacity from the closer limit towards x_solid.

    Args:
        x1(float): lower side of x
        x2(float): height side of x
        x_solid: If x_solid is not None, the gradient will be extended at the maximum opacity from
                 the closer limit towards x_solid.
        fill_color (str): Color name, pass it as "c" in mpl.colors.to_rgb
        ax(Axes): matplotlib Axes object
        **kwargs: Pass to imshow  (Z order can be set here.)

    Returns:
        The result of the inner imshow.
    """
    if ax is None:
        ax = plt.gca()
    assert isinstance(ax, Axes)

    alpha = float(kwargs.get("alpha", 1.0))
    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("origin", "lower")
    step = kwargs.pop("step", None)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert fill_color

    z = np.empty((1, 100, 4), dtype=float)

    rgb = colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[None, :]
    assert x1 < x2
    xmin, xmax, (ymin, ymax) = x1, x2, ylim
    kwargs.setdefault("extent", (xmin, xmax, ymin, ymax))

    im: AxesImage = ax.imshow(
        z,
        **kwargs,
    )

    if x_solid is not None:
        xlow, xhigh = (x2, x_solid) if x_solid > x2 else (x_solid, x1)
        ax.fill_betweenx(
            ylim,
            xlow,
            xhigh,
            color=fill_color,
            alpha=alpha,
            step=step,
        )

    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    return im


def v_gradient_fill(
    y1: float,
    y2: float,
    y_solid: float | None,
    fill_color: RGBColorType = "red",
    ax: Axes | None = None,
    **kwargs: Unpack[GradientFillParam],
) -> AxesImage:
    """Fills a gradient vertically between y1 and y2.

    If y_solid is not None, the gradient will be extended
    at the maximum opacity from the closer limit towards y_solid.

    Args:
        y1(float): Lower side for limit to fill.
        y2(float): Higher side for to fill.
        y_solid (float|solid): If y_solid is not None, the gradient will be extended at the maximum
            opacity from the closer limit towards y_solid.
        fill_color (str): Color name, pass it as "c" in mpl.colors.to_rgb  (Default "red")
        ax(Axes): matplotlib Axes object
        **kwargs: (str|float): pass to ax.imshow

    Returns:
        The result of the inner imshow call.
    """
    if ax is None:
        ax = plt.gca()

    alpha = float(kwargs.get("alpha", 1.0))
    assert isinstance(ax, Axes)
    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("origin", "lower")
    step = kwargs.pop("step", None)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert fill_color

    z = np.empty((100, 1, 4), dtype=float)

    rgb = colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]
    assert y1 < y2
    (xmin, xmax), ymin, ymax = xlim, y1, y2
    kwargs.setdefault("extent", (xmin, xmax, ymin, ymax))
    im: AxesImage = ax.imshow(
        z,
        **kwargs,
    )

    if y_solid is not None:
        ylow, yhigh = (y2, y_solid) if y_solid > y2 else (y_solid, y1)
        ax.fill_between(
            xlim,
            ylow,
            yhigh,
            color=fill_color,
            alpha=alpha,
            step=step,
        )

    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    return im


def simple_ax_grid(
    n_axes: int,
    figsize: tuple[float, float] = (0, 0),
    **kwargs: Unpack[PLTSubplotParam],
) -> tuple[Figure, NDArray[np.object_], NDArray[np.object_]]:
    """Generates a square-ish set of axes and hides the extra ones.

    It would be nice to accept an "aspect ratio" item that will attempt to fix the
    grid dimensions to get an aspect ratio close to the desired one.

    Args:
        n_axes(int): number of axis # <== checkme!
        figsize (tuple[float, float]): Pass to figsize in plt.subplots.
        kwargs: pass to plg.subplot

    Returns:
        The figure, the first n axis which are shown, and the remaining hidden axes.
    """
    width = int(np.ceil(np.sqrt(n_axes)))
    height = width - 1
    if width * height < n_axes:
        height += 1

    if figsize == (0, 0):
        figsize = (
            3 * max(width, 5),
            3 * max(height, 5),
        )

    fig, ax = plt.subplots(height, width, figsize=figsize, **kwargs)
    if n_axes == 1:
        ax = np.array([ax])

    ax, ax_rest = ax.ravel()[:n_axes], ax.ravel()[n_axes:]
    for axi in ax_rest:
        invisible_axes(axi)

    return fig, ax, ax_rest


@contextlib.contextmanager
def dark_background(overrides: dict[str, Incomplete]) -> Generator[None, None, None]:
    """Context manager for plotting "dark mode"."""
    defaults = {
        "axes.edgecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.labelcolor": "white",
        "text.color": "white",
    }
    defaults.update(overrides)

    with plt.rc_context(defaults):
        yield


def data_to_axis_units(
    points: tuple[float, float],
    ax: Axes | None = None,
) -> NDArray[np.float_]:
    """Converts between data and axis units."""
    if ax is None:
        ax = plt.gca()
    assert isinstance(ax, Axes)
    return ax.transAxes.inverted().transform(ax.transData.transform(points))


def axis_to_data_units(
    points: tuple[float, float],
    ax: Axes | None = None,
) -> NDArray[np.float_]:
    """Converts between axis and data units."""
    if ax is None:
        ax = plt.gca()
    assert isinstance(ax, Axes)
    return ax.transData.inverted().transform(ax.transAxes.transform(points))


def ddata_daxis_units(ax: Axes | None = None) -> NDArray[np.float_]:
    """Gives the derivative of data units with respect to axis units."""
    if ax is None:
        ax = plt.gca()

    dp1 = axis_to_data_units((1.0, 1.0), ax)
    dp0 = axis_to_data_units((0.0, 0.0), ax)
    return dp1 - dp0


def daxis_ddata_units(ax: Axes | None = None) -> NDArray[np.float_]:
    """Gives the derivative of axis units with respect to data units."""
    if ax is None:
        ax = plt.gca()
    isinstance(ax, Axes)
    dp1 = data_to_axis_units((1.0, 1.0), ax)
    dp0 = data_to_axis_units((0.0, 0.0), ax)
    return dp1 - dp0


def swap_xaxis_side(ax: Axes) -> None:
    """Swaps the x axis to the top of the figure."""
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")


def swap_yaxis_side(ax: Axes) -> None:
    """Swaps the y axis to the right of the figure."""
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


def swap_axis_sides(ax: Axes) -> None:
    """Swaps the y axis to the right of the figure and the x axis to the top."""
    swap_xaxis_side(ax)
    swap_yaxis_side(ax)


def transform_labels(
    transform_fn: Callable[[str, bool], str],
    fig: Figure | None = None,
    *,
    include_titles: bool = True,
) -> None:
    """Apply a function to all axis labeled in a figure."""
    if fig is None:
        fig = plt.gcf()
    assert isinstance(fig, Figure)
    axes = list(fig.get_axes())
    for ax in axes:
        try:
            ax.set_xlabel(transform_fn(ax.get_xlabel(), is_title=False))
            ax.set_ylabel(transform_fn(ax.get_xlabel(), is_title=False))
            if include_titles:
                ax.set_title(transform_fn(ax.get_title(), is_title=True))
        except TypeError:
            ax.set_xlabel(transform_fn(ax.get_xlabel()))
            ax.set_ylabel(transform_fn(ax.get_xlabel()))
            if include_titles:
                ax.set_title(transform_fn(ax.get_title()))


def summarize(data: DataType, axes: NDArray[np.object_] | None = None) -> NDArray[np.object_]:
    """Makes a summary plot with different marginal plots represented."""
    data_arr = normalize_to_spectrum(data)
    assert isinstance(data_arr, xr.DataArray)
    axes_shapes_for_dims = {
        1: (1, 1),
        2: (1, 1),
        3: (2, 2),  # one extra here
        4: (3, 2),  # corresponds to 4 choose 2 axes
    }
    assert len(data_arr.dims) <= len(axes_shapes_for_dims)
    if axes is None:
        _, axes = plt.subplots(
            axes_shapes_for_dims.get(len(data_arr.dims), (3, 2)),
            figsize=(8, 8),
        )
    assert isinstance(axes, np.ndarray)
    flat_axes = axes.ravel()
    combinations = list(itertools.combinations(data_arr.dims, 2))
    for axi, combination in zip(flat_axes, combinations, strict=False):
        assert isinstance(axi, Axes)
        data_arr.sum(combination).plot(ax=axi)
        fancy_labels(axi)

    for i in range(len(combinations), len(flat_axes)):
        flat_axes[i].set_axis_off()

    return axes


def sum_annotation(
    eV: slice | None = None,  # noqa: N803
    phi: slice | None = None,
) -> str:
    """Annotates that a given axis was summed over by listing the integration range."""
    eV_annotation, phi_annotation = "", ""

    assert "use_tex" in SETTINGS

    def to_str(bound: float | None) -> str:
        if bound is None:
            return ""

        return f"{bound:.2f}"

    if eV is not None:
        if SETTINGS["use_tex"]:
            eV_annotation = "$\\text{E}_{" + to_str(eV.start) + "}^{" + to_str(eV.stop) + "}$"
        else:
            eV_annotation = to_str(eV.start) + " < E < " + to_str(eV.stop)
    if phi is not None:
        if SETTINGS["use_tex"]:
            phi_annotation = "$\\phi_{" + to_str(phi.start) + "}^{" + to_str(phi.stop) + "}$"
        else:
            phi_annotation = to_str(phi.start) + " < φ < " + to_str(phi.stop)

    return eV_annotation + phi_annotation


def mean_annotation(eV: slice | None = None, phi: slice | None = None) -> str:  # noqa: N803
    """Annotates that a given axis was meant (summed) over by listing the integration range."""
    eV_annotation, phi_annotation = "", ""

    assert "use_tex" in SETTINGS

    def to_str(bound: float | None) -> str:
        if bound is None:
            return ""

        return f"{bound:.2f}"

    if eV is not None:
        if SETTINGS["use_tex"]:
            eV_annotation = (
                "$\\bar{\\text{E}}_{" + to_str(eV.start) + "}^{" + to_str(eV.stop) + "}$"
            )
        else:
            eV_annotation = "Mean<" + to_str(eV.start) + " < E < " + to_str(eV.stop) + ">"
    if phi is not None:
        if SETTINGS["use_tex"]:
            phi_annotation = "$\\bar{\\phi}_{" + to_str(phi.start) + "}^{" + to_str(phi.stop) + "}$"
        else:
            phi_annotation = "Mean<" + to_str(phi.start) + " < φ < " + to_str(phi.stop) + ">"

    return eV_annotation + phi_annotation


def frame_with(ax: Axes, color: RGBColorType = "red", linewidth: float = 2) -> None:
    """Makes thick, visually striking borders on a matplotlib plot.

    Very useful for color coding results in a slideshow.
    """
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(linewidth)


LATEX_ESCAPE_MAP = {
    "_": r"\_",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "{": r"\{",
    "}": r"\}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
    "\\": r"\textbackslash{}",
}
LATEX_ESCAPE_REGEX = re.compile(
    "|".join(
        re.escape(str(k)) for k in sorted(LATEX_ESCAPE_MAP.keys(), key=lambda item: -len(item))
    ),
)


def latex_escape(text: str, *, force: bool = False) -> str:
    """Conditionally escapes a string based on the matplotlib settings.

    If you need the escaped string even if you are not using matplotlib with LaTeX
    support, you can pass `force=True`.

    Adjusted from suggestions at:
    https://stackoverflow.com/questions/16259923/how-can-i-escape-latex-special-characters-inside-django-templates

    Args:
        text: The contents which should be escaped
        force: Whether we should perform escaping even if matplotlib is
          not being used with LaTeX support.

    Returns:
        The escaped string which should appear in LaTeX with the same
        contents as the original.
    """
    if not is_using_tex() and not force:
        return text

    # otherwise, we need to escape
    return LATEX_ESCAPE_REGEX.sub(lambda match: LATEX_ESCAPE_MAP[match.group()], text)


def quick_tex(latex_fragment: str, ax: Axes | None = None, fontsize: int = 30) -> Axes:
    """Sometimes you just need to render some LaTeX.

    Getting a LaTex session running is far too much effort.
    Also just go to the KaTeX website and can work well.

    Args:
        latex_fragment: The fragment to render
        ax (Axes): matploglib Axes ofbject
        fontsize(int): font size

    Returns:
        The axes generated.
    """
    if ax is None:
        _, ax = plt.subplots()
    assert isinstance(ax, Axes)

    invisible_axes(ax)
    ax.text(0.2, 0.2, latex_fragment, fontsize=fontsize)
    return ax


def lineplot_arr(
    arr: XrTypes,
    ax: Axes | None = None,
    method: Literal["plot", "scatter"] = "plot",
    mask: list[slice] | None = None,
    mask_kwargs: Incomplete | None = None,
    **kwargs: Incomplete,
) -> Axes:
    """Convenience method to plot an array with a mask over some other data."""
    if mask_kwargs is None:
        mask_kwargs = {}
    assert isinstance(arr, xr.DataArray)
    if ax is None:
        _, ax = plt.subplots()
    assert isinstance(ax, Axes)

    xs = None
    if arr is not None:
        fn = plt.plot
        if method == "scatter":
            fn = plt.scatter

        xs = arr.coords[arr.dims[0]].values
        fn(xs, arr.values, **kwargs)

    if mask is not None:
        y_lim = ax.get_ylim()
        if isinstance(mask, list) and isinstance(mask[0], slice):
            for slice_mask in mask:
                ax.fill_betweenx(y_lim, slice_mask.start, slice_mask.stop, **mask_kwargs)
        else:
            raise NotImplementedError
        ax.set_ylim(bottom=y_lim[0], top=y_lim[1])

    return ax


def plot_arr(
    arr: XrTypes,
    ax: Axes | None = None,
    over: AxesImage | None = None,
    mask: DataType | None = None,
    **kwargs: Incomplete,
) -> Axes | None:
    """Convenience method to plot an array with a mask over some other data."""
    to_plot = arr if mask is None else mask
    assert isinstance(to_plot, xr.Dataset | xr.Dataset)
    try:
        n_dims = len(to_plot.dims)
    except AttributeError:
        n_dims = 1

    if n_dims == TwoDimensional:
        quad = None
        if arr is not None:
            ax, quad = imshow_arr(arr, ax=ax, over=over, **kwargs)
        if mask is not None:
            over = quad if over is None else over
            imshow_mask(mask, ax=ax, over=over, **kwargs)
    if n_dims == 1:
        ax = lineplot_arr(arr, ax=ax, mask=mask, **kwargs)

    return ax


def imshow_mask(
    mask: XrTypes,
    ax: Axes | None = None,
    over: AxesImage | None = None,
    **kwargs: Incomplete,
) -> None:
    """Plots a mask by using a fixed color and transparency."""
    assert over is not None

    if ax is None:
        ax = plt.gca()
    assert isinstance(ax, Axes)

    default_kwargs = {
        "origin": "lower",
        "aspect": ax.get_aspect(),
        "alpha": 1.0,
        "vmin": 0,
        "vmax": 1,
        "cmap": "Reds",
        "extent": over.get_extent(),
        "interpolation": "none",
    }
    default_kwargs.update(kwargs)
    kwargs = default_kwargs

    if isinstance(kwargs["cmap"], str):
        kwargs["cmap"] = mpl.colormaps.get_cmap(cmap=kwargs["cmap"])

    assert isinstance(kwargs["cmap"], Colormap)
    kwargs["cmap"].set_bad("k", alpha=0)

    ax.imshow(
        mask.values,
        **kwargs,
    )


def imshow_arr(
    arr: XrTypes,
    ax: Axes | None = None,
    over: AxesImage | None = None,
    **kwargs: Unpack[IMshowParam],
) -> tuple[Figure, Axes]:
    """Similar to plt.imshow but users different default origin, and sets appropriate extents.

    Args:
        arr (XrTypes): ARPES data
        ax (Axes): [TODO:description]
        over ([TODO:type]): [TODO:description]
        kwargs: pass to ax.imshow

    Returns:
        The axes and quadmesh instance.
    """
    if ax is None:
        fig, ax = plt.subplots()
    assert isinstance(ax, Axes)

    x, y = arr.coords[arr.dims[0]].values, arr.coords[arr.dims[1]].values
    default_kwargs = {
        "origin": "lower",
        "aspect": "auto",
        "alpha": 1.0,
        "vmin": arr.min().item(),
        "vmax": arr.max().item(),
        "cmap": "viridis",
        "extent": (y[0], y[-1], x[0], x[-1]),
    }
    default_kwargs.update(kwargs)
    kwargs = default_kwargs
    if over is None:
        if kwargs["alpha"] != 1:
            if isinstance(kwargs["cmap"], str):
                cmap = mpl.colormaps.get_cmap(kwargs["cmap"])
            norm = colors.Normalize(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
            mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            mapped_colors = mappable.to_rgba(arr.values)
            mapped_colors[:, :, 3] = kwargs["alpha"]
            quad = ax.imshow(
                mapped_colors,
                **kwargs,
            )
        else:
            quad = ax.imshow(
                arr.values,
                **kwargs,
            )
        ax.grid(visible=False)
        ax.set_xlabel(str(arr.dims[1]))
        ax.set_ylabel(str(arr.dims[0]))
    else:
        kwargs["extent"] = over.get_extent()
        kwargs["aspect"] = ax.get_aspect()
        quad = ax.imshow(
            arr.values,
            extent=over.get_extent(),
            aspect=ax.get_aspect(),
            **kwargs,
        )

    return fig, quad


def dos_axes(
    orientation: str = "horiz",
    figsize: tuple[int, int] | tuple[()] = (),
) -> tuple[Figure, tuple[Axes, ...]]:
    """Makes axes corresponding to density of states data.

    This has one image like region and one small marginal for an EDC.
    Orientation option should be 'horiz' or 'vert'.

    Args:
        orientation: orientation of the Axes
        figsize: figure size

    Returns:
        The generated figure and axes as a tuple.
    """
    if not figsize:
        figsize = (12, 9) if orientation == "vert" else (9, 9)
    fig = plt.figure(figsize=figsize)
    gridspec.GridSpec(4, 4, wspace=0.0, hspace=0.0)
    if orientation.startswith("horiz"):  # "horizontal" is also ok
        fig.subplots_adjust(hspace=0.00)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
        axes = (ax0, plt.subplot(gs[1], sharex=ax0))
        plt.setp(axes[0].get_xticklabels(), visible=False)
    else:
        fig.subplots_adjust(wspace=0.00)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
        ax0 = plt.subplot(gs[1])
        axes = (ax0, plt.subplot(gs[0], sharey=ax0))
        plt.setp(axes[0].get_yticklabels(), visible=False)
    return fig, axes


def inset_cut_locator(
    data: DataType,
    reference_data: DataType,
    ax: Axes,
    location: dict[str, Incomplete],
    color: ColorType = "red",
    **kwargs: Incomplete,
) -> None:
    """Plots a reference cut location over a figure.

    Another approach is to separately plot the locator and add it in Illustrator or
    another tool.

    Args:
        data: The data you are plotting
        reference_data: The reference data containing the location of the cut
        ax: The axes to plot on
        location: The location in the cut
        color: The color to use for the indicator line
        kwargs: Passed to ax.plot when making the indicator lines
    """
    quad = data.plot(ax=ax)
    assert isinstance(ax, Axes)

    ax.set_xlabel("")
    ax.set_ylabel("")
    with contextlib.suppress(Exception):
        quad.colorbar.remove()

    assert isinstance(data, xr.Dataset | xr.DataArray)
    assert isinstance(reference_data, xr.Dataset | xr.DataArray)
    # add more as necessary
    missing_dim_resolvers = {
        "theta": lambda: reference_data.S.theta,
        "beta": lambda: reference_data.S.beta,
        "phi": lambda: reference_data.S.phi,
    }
    missing_dims = [dim for dim in data.dims if dim not in location]
    missing_values = {dim: missing_dim_resolvers[dim]() for dim in missing_dims}
    ordered_selector = [location.get(dim, missing_values.get(dim)) for dim in data.dims]

    n = 200

    def resolve(name: str, value: slice | int) -> NDArray[np.float_]:
        if isinstance(value, slice):
            low = value.start
            high = value.stop

            if low is None:
                low = data.coords[name].min().item()
            if high is None:
                high = data.coords[name].max().item()

            return np.linspace(low, high, n)

        return np.ones((n,)) * value

    n_cut_dims = len([d for d in ordered_selector if isinstance(d, Iterable | slice)])
    ordered_selector = [
        resolve(d, v)
        for d, v in zip(
            data.dims,
            ordered_selector,
            strict=True,
        )
    ]

    if missing_dims:
        assert reference_data is not None
        logger.info(missing_dims)

    if n_cut_dims == TwoDimensional:
        # a region cut, illustrate with a rect or by suppressing background
        return

    if n_cut_dims == 1:
        # a line cut, illustrate with a line
        ax.plot(*ordered_selector[::-1], color=color, **kwargs)
    elif n_cut_dims == 0:
        # a single point cut, illustrate with a marker
        pass


def generic_colormap(low: float, high: float) -> Callable[..., RGBAColorType]:
    """Generates a colormap from the cm.Blues palette, suitable for most purposes."""
    delta = high - low
    low = low - delta / 6
    high = high + delta / 6

    def get_color(value: float) -> RGBAColorType:
        return mpl.colormaps.get_cmap("Blues")(float((value - low) / (high - low)))

    return get_color


def phase_angle_colormap(
    low: float = 0,
    high: float = np.pi * 2,
) -> Callable[[float], RGBAColorType]:
    """Generates a colormap suitable for angular data or data on a unit circle like a phase."""

    def get_color(value: float) -> RGBAColorType:
        return mpl.colormaps.get_cmap("twilight_shifted")(float((value - low) / (high - low)))

    return get_color


def delay_colormap(low: float = -1, high: float = 1) -> Callable[[float], RGBAColorType]:
    """Generates a colormap suitable for pump-probe delay data."""

    def get_color(value: float) -> RGBAColorType:
        return mpl.colormaps.get_cmap("coolwarm")(float((value - low) / (high - low)))

    return get_color


def temperature_colormap(
    low: float = 0,
    high: float = 300,
    cmap: Colormap = mpl.colormaps["Blues_r"],
) -> Callable[[float], RGBAColorType]:
    """Generates a colormap suitable for temperature data with fixed extent."""

    def get_color(value: float) -> RGBAColorType:
        return cmap(float((value - low) / (high - low)))

    return get_color


def temperature_colormap_around(
    central: float,
    region: float = 50,
) -> Callable[[float], RGBAColorType]:
    """Generates a colormap suitable for temperature data around a central value."""

    def get_color(value: float) -> RGBAColorType:
        return mpl.colormaps.get_cmap("RdBu_r")(float((value - central) / region))

    return get_color


def generic_colorbar(
    low: float,
    high: float,
    ax: Axes,
    **kwargs: Unpack[ColorbarParam],
) -> colorbar.Colorbar:
    """Generate colorbar.

    Args:
        low(float): value for lowest value of the colorbar
        high(float): value for hightst value of the colorbar
        ax(Axes): Matplotlib Axes object
        **kwargs: Pass to ColoarbarBase
    """
    kwargs.setdefault("cmap", mpl.colormaps.get_cmap("Blues"))
    kwargs.setdefault("norm", colors.Normalize(vmin=low, vmax=high))
    kwargs.setdefault("ticks", [low, high])
    kwargs.setdefault("orientation", "horizontal")

    delta = high - low
    low = low - delta / 6
    high = high + delta / 6

    return colorbar.Colorbar(ax, **kwargs)


def phase_angle_colorbar(
    low: float = 0,
    high: float = np.pi * 2,
    ax: Axes | None = None,
    **kwargs: Incomplete,
) -> colorbar.Colorbar:
    """Generates a colorbar suitable for plotting an angle or value on a unit circle."""
    assert isinstance(ax, Axes)
    assert "use_tex" in SETTINGS

    kwargs.setdefault("cmap", mpl.colormaps.get_cmap("Blues_r"))
    kwargs.setdefault("norm", colors.Normalize(vmin=low, vmax=high))
    kwargs.setdefault("label", "Angle (rad)")
    kwargs.setdefault("ticks", ["0", r"$\pi$", r"$2\pi$"])
    kwargs.setdefault("orientation", "horizontal")

    if not SETTINGS["use_tex"]:
        kwargs["ticks"] = ["0", "π", "2π"]

    return colorbar.Colorbar(ax, **kwargs)


def temperature_colorbar(
    low: float = 0,
    high: float = 300,
    ax: Axes | None = None,
    **kwargs: Incomplete,
) -> colorbar.Colorbar:
    """Generates a colorbar suitable for temperature data with fixed extent."""
    assert isinstance(ax, Axes)
    kwargs.setdefault("cmap", mpl.colormaps.get_cmap("Blues_r"))
    kwargs.setdefault("norm", colors.Normalize(vmin=low, vmax=high))
    kwargs.setdefault("label", "Temperature  (K)")
    kwargs.setdefault("ticks", [low, high])
    kwargs.setdefault("orientation", "horizontal")

    if isinstance(kwargs["cmap"], str):
        kwargs["cmap"] = mpl.colormaps.get_cmap(kwargs["cmap"])
    return colorbar.Colorbar(
        ax,
        **kwargs,
    )


def delay_colorbar(
    low: float = -1,
    high: float = 1,
    ax: Axes | None = None,
    **kwargs: Incomplete,
) -> colorbar.Colorbar:
    assert isinstance(ax, Axes)
    """Generates a colorbar suitable for delay data.

    TODO make this nonsequential for use in case where you want to have a long time period after the
    delay or before.
    """
    kwargs.setdefault("cmap", mpl.colormaps.get_cmap("coolwarm"))
    kwargs.setdefault("norm", colors.Normalize(vmin=low, vmax=high))
    kwargs.setdefault("label", "Probe pulse delay (fs)")
    kwargs.setdefault("ticks", [low, 0, high])
    kwargs.setdefault("orientation", "horizontal")
    return colorbar.Colorbar(ax, **kwargs)


def temperature_colorbar_around(
    central: float,
    temperature_range: float = 50,
    ax: Axes | None = None,
    **kwargs: Incomplete,
) -> colorbar.Colorbar:
    """Generates a colorbar suitable for temperature axes around a central value."""
    assert isinstance(ax, Axes)
    kwargs.setdefault("cmap", mpl.colormaps.get_cmap("RdBu_r"))
    kwargs.setdefault(
        "norm",
        colors.Normalize(
            vmin=central - temperature_range,
            vmax=central + temperature_range,
        ),
    )
    kwargs.setdefault("label", "Temperature  (K)")
    kwargs.setdefault("ticks", [central - temperature_range, central + temperature_range])
    kwargs.setdefault("orientation", "horizontal")
    return colorbar.Colorbar(ax, **kwargs)


colorbarmaps_for_axis: dict[
    str,
    tuple[
        Callable[..., colorbar.Colorbar],
        Callable[
            ...,
            Callable[..., RGBAColorType],
        ],
    ],
] = {
    "temp": (
        temperature_colorbar,
        temperature_colormap,
    ),
    "delay": (
        delay_colorbar,
        delay_colormap,
    ),
    "theta": (
        phase_angle_colorbar,
        phase_angle_colormap,
    ),
    "volts": (
        generic_colorbar,
        generic_colormap,
    ),
}


def get_colorbars(fig: Figure | None = None) -> list[Axes]:
    """Collects likely colorbars in a figure."""
    if fig is None:
        fig = plt.gcf()
    assert isinstance(fig, Figure)
    return [ax for ax in fig.axes if ax.get_aspect() == 20]  # noqa: PLR2004


def remove_colorbars(fig: Figure | None = None) -> None:
    """Removes colorbars from given (or, if no given figure, current) matplotlib figure.

    Args:
        fig: The figure to modify, by default uses the current figure (`plt.gcf()`)
    """
    # TODO: after colorbar removal, plots should be relaxed/rescaled to occupy space previously
    # allocated to colorbars for now, can follow this with plt.tight_layout()
    try:
        if fig is not None:
            for ax in fig.axes:
                if ax.get_aspect() >= 20:  # a bit of a hack
                    ax.remove()
        else:
            remove_colorbars(plt.gcf())
    except Exception as err:
        logger.debug(f"Exception occurs: {err=}, {type(err)=}")


generic_colorbarmap = (
    generic_colorbar,
    generic_colormap,
)


def generic_colorbarmap_for_data(
    data: xr.DataArray,
    ax: Axes,
    *,
    keep_ticks: bool = True,
    **kwargs: Incomplete,
) -> tuple[colorbar.Colorbar, Callable[..., RGBAColorType]]:
    """Generates a colorbar and colormap which is useful in general context.

    Args:
        data(xr.DataArray): data of coords. Note that not ARPES data itself.
        ax(Axes): matplotlib.Axes object
        keep_ticks(bool): if True, use coord values for ticks.
        **kwargs: pass to generic_colorbar then to colorbar.Colorbar

    Returns:
        tuple[]
    """
    low, high = data.min().item(), data.max().item()
    ticks = None
    if keep_ticks:
        ticks = data.values
    return (
        generic_colorbar(low=low, high=high, ax=ax, ticks=kwargs.get("ticks", ticks)),
        generic_colormap(low=low, high=high),
    )


def polarization_colorbar(ax: Axes | None = None) -> colorbar.Colorbar:
    """Makes a colorbar which is appropriate for "polarization" (e.g. spin) data."""
    assert isinstance(ax, Axes)
    return colorbar.Colorbar(
        ax,
        cmap="RdBu",
        norm=colors.Normalize(vmin=-1, vmax=1),
        orientation="horizontal",
        label="Polarization",
        ticks=[-1, 0, 1],
    )


def calculate_aspect_ratio(data: DataType) -> float:
    """Calculate the aspect ratio which should be used for plotting some data based on extent."""
    data_arr = normalize_to_spectrum(data)
    assert isinstance(data_arr, xr.DataArray)
    assert len(data.dims) == TwoDimensional

    x_extent = np.ptp(data_arr.coords[data_arr.dims[0]].values)
    y_extent = np.ptp(data_arr.coords[data_arr.dims[1]].values)

    return y_extent / x_extent


class AnchoredHScaleBar(mpl.offsetbox.AnchoredOffsetbox):
    """Provides an anchored scale bar on the X axis.

    Modified from `this StackOverflow question <https://stackoverflow.com/questions/43258638/>`_
    as alternate to the one provided through matplotlib.
    """

    def __init__(
        self,
        size: float = 1,
        extent: float = 0.03,
        label: str = "",
        loc: int = 2,
        ax: Axes | None = None,
        pad: float = 0.4,
        borderpad: float = 0.5,
        ppad: float = 0,
        sep: int = 2,
        prop: FontProperties | None = None,
        label_color: ColorType | None = None,
        *,
        frameon: bool = True,
        **kwargs: Incomplete,
    ) -> None:
        """Setup the scale bar and coordinate transforms to the parent axis."""
        if not ax:
            ax = plt.gca()
        assert isinstance(ax, Axes)
        trans = ax.get_xaxis_transform()

        size_bar = mpl.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **kwargs)
        vline1 = Line2D([0, 0], [-extent / 2.0, extent / 2.0], **kwargs)
        vline2 = Line2D([size, size], [-extent / 2.0, extent / 2.0], **kwargs)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = mpl.offsetbox.TextArea(
            label,
            minimumdescent=False,
            textprops={
                "color": label_color,
            },
        )
        self.vpac = mpl.offsetbox.VPacker(
            children=[size_bar, txt],
            align="center",
            pad=ppad,
            sep=sep,
        )
        mpl.offsetbox.AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=self.vpac,
            prop=prop,
            frameon=frameon,
        )


def load_data_for_figure(p: str | Path) -> None:
    """Tries to load the data associated with a given figure by unpickling the saved data."""
    path = str(p)
    stem = str(Path(path).parent / Path(path).stem)
    if stem.endswith("-PAPER"):
        stem = stem[:-6]

    pickle_file = stem + ".pickle"

    if not Path(pickle_file).exists():
        msg = "No saved data matching figure."
        raise ValueError(msg)

    with Path(pickle_file).open("rb") as f:
        return pickle.load(f)


def savefig(
    desired_path: str | Path,
    dpi: int = 400,
    data: list[DataType] | tuple[DataType, ...] | set[DataType] | None = None,
    save_data=None,
    *,
    paper: bool = False,
    **kwargs: Incomplete,
) -> None:
    """The PyARPES preferred figure saving routine.

    Provides a number of conveniences over matplotlib's `savefig`:

    #. Output is scoped per project and per day, which aids organization
    #. The dpi is set to a reasonable value for the year 2021.
    #. By omitting a file extension you will get high and low res formats in .png and .pdf
       which is useful for figure drafting in external software (Adobe Illustrator)
    #. Data and plot provenenace is tracked, which makes it easier to find your analysis
       after the fact if you have many many plots.

    """
    desired_path = Path(desired_path)
    assert isinstance(desired_path, Path)
    if not desired_path.suffix:
        paper = True

    if save_data is None:
        if paper:
            msg = "You must supply save_data when outputting in paper mode."
            msg += "This is for your own good so you can more easily regenerate the figure later!"
            raise ValueError(
                msg,
            )
    else:
        output_location = path_for_plot(desired_path.parent / desired_path.stem)
        with Path(str(output_location) + ".pickle").open("wb") as f:
            pickle.dump(save_data, f)

    if paper:
        # automatically generate useful file formats
        high_dpi = max(dpi, 400)
        formats_for_paper = ["pdf", "png"]  # not including SVG anymore because files too large

        for the_format in formats_for_paper:
            savefig(
                f"{desired_path}-PAPER.{the_format}",
                dpi=high_dpi,
                data=data,
                paper=False,
                **kwargs,
            )

        savefig(f"{desired_path}-low-PAPER.pdf", dpi=200, data=data, paper=False, **kwargs)

        return

    full_path = path_for_plot(desired_path)
    provenance_path = str(full_path) + ".provenance.json"
    provenance_context = {
        "VERSION": VERSION,
        "time": datetime.datetime.now(UTC).isoformat(),
        "jupyter_notebook_name": get_notebook_name(),
        "name": "savefig",
    }

    def extract(for_data: DataType) -> dict[str, Any]:
        return for_data.attrs.get("provenance", {})

    if data is not None:
        assert isinstance(
            data,
            list | tuple | set,
        )
        provenance_context.update(
            {
                "jupyter_context": get_recent_history(1),
                "data": [extract(d) for d in data],
            },
        )
    else:
        # get more recent history because we don't have the data
        provenance_context.update(
            {
                "jupyter_context": get_recent_history(5),
            },
        )

    with Path(provenance_path).open("w") as f:
        json.dump(provenance_context, f, indent=2)
    plt.savefig(full_path, dpi=dpi, **kwargs)


def path_for_plot(desired_path: str | Path) -> Path:
    """Provides workspace and date scoped path generation for plots.

    This is used to ensure that analysis products are grouped together
    and organized in a reasonable way (by each day, together).

    This will be used automatically if you use `arpes.plotting.utils.savefig`
    instead of the one from matplotlib.
    """
    if not CONFIG["WORKSPACE"]:
        attempt_determine_workspace()

    workspace = CONFIG["WORKSPACE"]

    if not workspace:
        warnings.warn("Saving locally, no workspace found.", stacklevel=2)
        return Path.cwd() / desired_path

    try:
        import arpes.config

        figure_path = arpes.config.FIGURE_PATH
        if figure_path is None:
            figure_path = Path(workspace["path"]) / "figures"

        filename = (
            Path(figure_path)
            / workspace["name"]
            / datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat()
            / desired_path
        )
        filename = Path(filename).absolute()
        parent_directory = Path(filename).parent
        if not Path(parent_directory).exists():
            try:
                Path(parent_directory).mkdir(parents=True)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        return filename
    except Exception as e:
        warnings.warn(f"Misconfigured FIGURE_PATH saving locally: {e}", stacklevel=2)
        return Path.cwd() / desired_path


def path_for_holoviews(desired_path: str) -> str:
    """Determines an appropriate output path for a holoviews save."""
    skip_paths = [".svg", ".png", ".jpeg", ".jpg", ".gif"]

    ext = str(Path(desired_path).suffix)
    prefix = str(Path(desired_path).parent / Path(desired_path).stem)

    if ext in skip_paths:
        return prefix

    return prefix + ext


def name_for_dim(dim_name: str, *, escaped: bool = True) -> str:
    """Alternate variant of `label_for_dim`."""
    assert "use_tex" in SETTINGS

    if SETTINGS["use_tex"]:
        name = {
            "temperature": "Temperature",
            "beta": r"$\beta$",
            "theta": r"$\theta$",
            "chi": r"$\chi$",
            "alpha": r"$\alpha$",
            "psi": r"$\psi$",
            "phi": r"$\phi",
            "eV": r"$\textnormal{E}$",
            "kx": r"$\textnormal{k}_\textnormal{x}$",
            "ky": r"$\textnormal{k}_\textnormal{y}$",
            "kz": r"$\textnormal{k}_\textnormal{z}$",
            "kp": r"$\textnormal{k}_\textnormal{\parallel}$",
            "hv": r"$h\nu$",
        }.get(dim_name, "")
    else:
        name = {
            "temperature": "Temperature",
            "beta": "β",
            "theta": "θ",
            "chi": "χ",
            "alpha": "a",
            "psi": "ψ",
            "phi": "φ",
            "eV": "E",
            "kx": "Kx",
            "ky": "Ky",
            "kz": "Kz",
            "kp": "Kp",
            "hv": "Photon Energy",
        }.get(dim_name, "")

    if not escaped:
        name = name.replace("$", "")

    return name


def unit_for_dim(dim_name: str, *, escaped: bool = True) -> str:
    """Calculate LaTeX or fancy display label for the unit associated to a dimension."""
    assert "use_tex" in SETTINGS
    if SETTINGS["use_tex"]:
        unit = {
            "temperature": "K",
            "theta": r"rad",
            "beta": r"rad",
            "psi": r"rad",
            "chi": r"rad",
            "alpha": r"rad",
            "phi": r"rad",
            "eV": r"eV",
            "kx": r"$\AA^{-1}$",
            "ky": r"$\AA^{-1}$",
            "kz": r"$\AA^{-1}$",
            "kp": r"$\AA^{-1}$",
            "hv": r"eV",
        }.get(dim_name, "")
    else:
        unit = {
            "temperature": "K",
            "theta": r"rad",
            "beta": r"rad",
            "psi": r"rad",
            "chi": r"rad",
            "alpha": r"rad",
            "phi": r"rad",
            "eV": r"eV",
            "kx": "1/Å",
            "ky": "1/Å",
            "kz": "1/Å",
            "kp": "1/Å",
            "hv": "eV",
        }.get(dim_name, "")

    if not escaped:
        unit = unit.replace("$", "")

    return unit


def label_for_colorbar(data: DataType) -> str:
    """Returns an appropriate label for an ARPES intensity colorbar."""
    if not data.S.is_differentiated:
        return r"Spectrum Intensity (arb.)"

    # determine which axis was differentiated
    hist = data.S.history
    records = [h["record"] for h in hist if isinstance(h, dict)]
    if "curvature" in [r["by"] for r in records]:
        curvature_record = next(r for r in records if r["by"] == "curvature")
        directions = curvature_record["directions"]
        return rf"Curvature along {name_for_dim(directions[0])} and {name_for_dim(directions[1])}"

    derivative_records = [r for r in records if r["by"] == "dn_along_axis"]
    c = Counter(itertools.chain(*[[d["axis"]] * d["order"] for d in derivative_records]))

    partial_frag = r""
    if sum(c.values()) > 1:
        partial_frag = r"^" + str(sum(c.values()))

    return (
        r"$\frac{\partial"
        + partial_frag
        + r" \textnormal{Int.}}{"
        + r"".join(
            [rf"\partial {name_for_dim(item, escaped=False)}^{n}" for item, n in c.items()],
        )
        + "}$ (arb.)"
    )


def label_for_dim(
    data: DataType | None = None,
    dim_name: Hashable = "",
    *,
    escaped: bool = True,
) -> str:
    """Generates a fancy label for a dimension according to standard conventions.

    If available, LaTeX is used

    Args:
        data(DataType | None): Source data, used to calculate names, typically you can leave this
            empty <== for backward compatibility ?
        dim_name(str): name of dimension (axis)
        escaped(bool) : if True, remove $

    Returns:
        str

    Todo: Think about removing data argument

    """
    if SETTINGS.get("use_tex", False):
        raw_dim_names = {
            "temperature": "Temperature ( K )",
            "theta": r"$\theta$",
            "beta": r"$\beta$",
            "chi": r"$\chi$",
            "alpha": r"$\alpha$",
            "psi": r"$\psi$",
            "phi": r"$\varphi$",
            "eV": r"Binding Energy ( eV )",
            "angle": r"Interp. Angle",
            "kinetic": r"Kinetic Energy ( eV )",
            "temp": r"Temperature",
            "kp": r"$k_\parallel$",
            "kx": r"$k_\text{x}$",
            "ky": r"$k_\text{y}$",
            "kz": r"$k_\perp$",
            "hv": "Photon Energy",
            "x": "X ( mm )",
            "y": "Y ( mm )",
            "z": "Z ( mm )",
            "spectrum": "Intensity ( arb. )",
        }
        if isinstance(data, xr.Dataset | xr.DataArray):
            if data.S.energy_notation == "Kinetic":
                raw_dim_names["eV"] = r"Final State Energy ( eV )"
            else:
                raw_dim_names["eV"] = r"Binding Energy ( eV )"
    else:
        raw_dim_names = {
            "temperature": "Temperature ( K )",
            "beta": "β",
            "theta": "θ",
            "chi": "χ",
            "alpha": "a",
            "psi": "ψ",
            "phi": "φ",
            "eV": "Binding Energy ( eV )",
            "angle": "Interp. Angle",
            "kinetic": "Kinetic Energy ( eV )",
            "temp": "Temperature ( K )",
            "kp": "Kp",
            "kx": "Kx",
            "ky": "Ky",
            "kz": "Kz",
            "hv": "Photon Energy ( eV )",
            "x": "X ( mm )",
            "y": "Y ( mm )",
            "z": "Z ( mm )",
            "spectrum": "Intensity ( arb. )",
        }
        if isinstance(data, xr.DataArray | xr.Dataset):
            if data.S.energy_notation == "Kinetic":
                raw_dim_names["eV"] = "Final State Energy ( eV )"
            else:
                raw_dim_names["eV"] = "Binding Energy ( eV )"
    if dim_name in raw_dim_names:
        label_dim_name = raw_dim_names.get(str(dim_name), "")
        if not escaped:
            label_dim_name = label_dim_name.replace("$", "")
        return label_dim_name

    try:
        from titlecase import titlecase
    except ImportError:
        warnings.warn(
            "Using alternative titlecase, for better results `pip install titlecase`.",
            stacklevel=2,
        )

        def titlecase(s: str) -> str:
            """Poor man's titlecase.

            Args:
                s: The input string

            Returns:
                The titlecased string.
            """
            return s.title()

    return titlecase(str(dim_name).replace("_", " "))


def fancy_labels(
    ax_or_ax_set: Axes | Sequence[Axes],
    data: DataType | None = None,
) -> None:
    """Attaches better display axis labels for all axes.

    Axes are determined by those that can be traversed in the passed figure or axes.

    Args:
        ax_or_ax_set: The axis to search for subaxes
        data: The source data, used to calculate names, typically you can leave this empty
    """
    if isinstance(ax_or_ax_set, Sequence):
        for ax in ax_or_ax_set:
            fancy_labels(ax)
        return

    ax = ax_or_ax_set
    assert isinstance(ax, Axes)
    ax.set_xlabel(label_for_dim(data=data, dim_name=ax.get_xlabel()))

    with contextlib.suppress(Exception):
        ax.set_ylabel(label_for_dim(data=data, dim_name=ax.get_ylabel()))


def label_for_symmetry_point(point_name: str) -> str:
    """Determines the LaTeX label for a symmetry point shortcode."""
    assert "use_tex" in SETTINGS
    if SETTINGS["use_tex"]:
        proper_names = {"G": r"$\Gamma$", "X": r"X", "Y": r"Y", "K": r"K"}
    else:
        proper_names = {"G": r"Γ", "X": r"X", "Y": r"Y", "K": r"K"}

    return proper_names.get(point_name, point_name)


class CoincidentLinesPlot:
    """Helper to allow drawing lines at the same location.

    Will draw n lines offset so that their center appears at the data center,
    and the lines will end up nonoverlapping.

    Only works for straight lines.

    Technique adapted from `StackOverflow
    <https://stackoverflow.com/questions/19394505/matplotlib-expand-the-line-with-specified-width-in-data-unit>`_.
    """

    linewidth = 3

    def __init__(self, **kwargs: Incomplete) -> None:
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = kwargs.pop("fig", plt.gcf())
        self.extra_kwargs = kwargs
        self.ppd = 72.0 / self.fig.dpi
        self.has_drawn = False

        self.events = {
            "resize_event": self.ax.figure.canvas.mpl_connect("resize_event", self._resize),
            "motion_notify_event": self.ax.figure.canvas.mpl_connect(
                "motion_notify_event",
                self._resize,
            ),
            "button_release_event": self.ax.figure.canvas.mpl_connect(
                "button_release_event",
                self._resize,
            ),
        }
        self.handles = []
        self.lines = []  # saved args and kwargs for plotting, does not verify coincidence

    def add_line(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Adds an additional line into the collection to be drawn."""
        assert not self.has_drawn
        self.lines.append(
            (
                args,
                kwargs,
            ),
        )

    def draw(self) -> None:
        """Draw all of the lines after offsetting them slightly."""
        self.has_drawn = True

        offset_in_data_units = self.data_units_per_pixel * self.linewidth
        self.offsets = [
            offset_in_data_units * (o - (len(self.lines) - 1) / 2) for o in range(len(self.lines))
        ]

        for offset, (line_args, line_kwargs) in zip(self.offsets, self.lines, strict=True):
            normalized_line_args = self.normalize_line_args(line_args)
            normalized_line_args[1] = np.array(normalized_line_args[1]) + offset
            handle = self.ax.plot(*normalized_line_args, **line_kwargs)
            self.handles.append(handle)

    @property
    def data_units_per_pixel(self) -> tuple[float, float]:
        """Gets the data/pixel conversion ratio."""
        trans = self.ax.transData.transform
        inverse = (trans((1, 1)) - trans((0, 0))) * self.ppd
        return (1 / inverse[0], 1 / inverse[1])

    def normalize_line_args(self, args: Sequence[object]) -> list[object]:
        def is_data_type(value: object) -> bool:
            return isinstance(value, np.array | np.ndarray | list | tuple)

        assert is_data_type(args[0])

        if len(args) > 1 and is_data_type(args[1]) and len(args[0]) == len(args[1]):
            # looks like we have x and y data
            return args

        # otherwise we should pad the args with the x data
        return [range(len(args[0])), *args]


def invisible_axes(ax: Axes) -> None:
    """Make a Axes instance completely invisible."""
    ax.grid(visible=False)
    ax.set_axis_off()
    ax.patch.set_alpha(0)


def no_ticks(ax: Axes) -> None:
    """Remove all axis ticks."""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
