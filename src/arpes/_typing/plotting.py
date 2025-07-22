"""Specialized type annotations for use in PyARPES.

In particular, `DataType` refers to either an xarray.DataArray or xarray.Dataset

`NormalizableDataType` referes to anything that can be tuned into datase,
such as by loading from the cache using an ID.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import numpy as np
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import Event
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.path import Path as mpl_Path
    from matplotlib.patheffects import AbstractPathEffect
    from matplotlib.ticker import Locator
    from matplotlib.transforms import BboxBase, Transform
    from matplotlib.typing import (
        CapStyleType,
        ColorType,
        DrawStyleType,
        FillStyleType,
        JoinStyleType,
        LineStyleType,
        MarkerType,
        MarkEveryType,
    )
    from numpy.typing import ArrayLike, NDArray

    from .base import Orientation

LEGENDLOCATION = Literal[
    # Numeric values (0 to 10) are for the backward compatibility.
    "best",
    "upper right",
    "upper left",
    "lower left",
    "lower right",
    "right",
    "center left",
    "center right",
    "lower center",
    "upper center",
    "center",
]


class Line2DProperty(TypedDict, total=False):
    agg_filter: Callable[[NDArray[np.float64], int], tuple[NDArray[np.float64], int, int]]
    alpha: float | None
    animated: bool
    antialiased: bool | list[bool]
    clip_box: BboxBase | None
    clip_on: bool
    clip_path: mpl_Path | Patch | Transform | None
    color: ColorType
    c: ColorType
    dash_capstyle: CapStyleType
    dash_joinstyle: JoinStyleType
    dashes: LineStyleType
    drawstyle: DrawStyleType
    ds: DrawStyleType
    figure: Figure
    fillstyle: FillStyleType
    gapcolor: ColorType | None
    gid: str
    in_layout: bool
    label: Any
    linestyle: LineStyleType
    ls: LineStyleType
    marker: MarkerType
    markeredgecolor: ColorType
    mec: ColorType
    markeredgewidth: float
    mew: ColorType
    markerfacecoloralt: ColorType
    mfcalt: ColorType
    markersize: float
    ms: float
    markevery: MarkEveryType
    mouseover: bool
    path_effects: list[AbstractPathEffect]
    picker: float | Callable[[Artist, Event], tuple[bool, dict]]
    pickradius: float
    rasterized: bool
    sketch_params: tuple[float, float, float]
    snap: bool | None
    solid_capstyle: CapStyleType
    solid_joinstyle: JoinStyleType
    url: str
    visible: bool
    zorder: float


class PolyCollectionProperty(Line2DProperty, total=False):
    array: ArrayLike | None
    clim: tuple[float, float]
    cmap: Colormap | str | None
    edgecolor: ColorType | list[ColorType]
    ec: ColorType | list[ColorType]
    facecolor: ColorType | list[ColorType]
    fc: ColorType | list[ColorType]
    hatch: Literal["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    norm: Normalize | None
    offset_transform: Transform
    # offsets: (N, 2) or (2, ) array-like
    sizes: NDArray[np.float64] | None
    transform: Transform
    urls: list[str] | None


class MPLPlotKwargsBasic(TypedDict, total=False):
    """Kwargs for Axes.plot & Axes.fill_between."""

    agg_filter: Callable[[NDArray[np.float64], int], tuple[NDArray[np.float64], int, int]]
    alpha: float | None
    animated: bool
    antialiased: bool | list[bool]
    aa: bool | list[bool]
    clip_box: BboxBase | None
    clip_on: bool
    color: ColorType
    c: ColorType
    figure: Figure
    gid: str
    in_layout: bool
    label: str
    ls: LineStyleType
    linestyle: LineStyleType
    linewidth: float
    lw: float
    mouseover: bool
    path_effects: list[AbstractPathEffect]
    pickradius: float
    rasterized: bool
    sketch_params: tuple[float, float, float]
    snap: bool | None
    transform: Transform
    url: str
    visible: bool


class MPLPlotKwargs(MPLPlotKwargsBasic, total=False):
    scalex: bool
    scaley: bool
    fmt: str
    dash_capstyle: CapStyleType
    dash_joinstyle: JoinStyleType
    dashes: Sequence[float | None]
    drawstyle: DrawStyleType
    fillstyle: FillStyleType
    gapcolor: ColorType | None
    marker: MarkerType
    markeredgecolor: ColorType
    mec: ColorType
    markeredgewidth: float
    mew: float
    markerfacecolor: ColorType
    mfc: ColorType
    markerfacecoloralt: ColorType
    mfcalt: ColorType
    markersize: float
    ms: float
    markevery: MarkEveryType
    picker: float | Callable[[Artist, Event], tuple[bool, dict]]
    scale: float
    length: float
    randomness: float
    solid_capstyle: CapStyleType
    solid_joinstyle: JoinStyleType
    xdata: NDArray[np.float64]
    ydata: NDArray[np.float64]
    zorder: float


class ColorbarParam(TypedDict, total=False):
    alpha: float
    orientation: None | Orientation
    ticklocation: Literal["auto", "right", "top", "bottom"]
    extend: Literal["neither", "both", "min", "max"]
    extendfrac: None | Literal["auto"] | float | tuple[float, float] | list[float]
    spacing: Literal["uniform", "proportional"]
    ticks: None | Sequence[float] | Locator
    format: str | None
    drawedges: bool
    label: str
    boundaries: None | Sequence[float]
    values: None | Sequence[float]
    location: None | Literal["left", "right", "top", "bottom"]
    cmap: Colormap
    norm: Normalize


_FONTSIZES = Literal[
    "xx-small",
    "x-small",
    "small",
    "medium",
    "large",
    "x-large",
    "xx-large",
]

_FONTSTRETCHS = Literal[
    "ultra-condensed",
    "extra-condensed",
    "condensed",
    "semi-condensed",
    "normal",
    "semi-expanded",
    "expanded",
    "extra-expanded",
    "ultra-expanded",
]

_FONTWEIGHTS = Literal[
    "ultralight",
    "light",
    "normal",
    "regular",
    "book",
    "medium",
    "roman",
    "semibold",
    "demibold",
    "demi",
    "bold",
    "heavy",
    "extra bold",
    "black",
]


class MPLTextParam(TypedDict, total=False):
    agg_filter: Callable[[NDArray[np.float64], int], tuple[NDArray[np.float64], int, int]]
    alpha: float | None
    animated: bool
    antialiased: bool
    backgroundcolor: ColorType
    color: ColorType
    c: ColorType
    figure: Figure
    fontfamily: str
    family: str
    fontname: str
    fontproperties: str | Path
    font: str | Path
    font_properties: str | Path
    fontsize: float | _FONTSIZES
    size: float | _FONTSIZES
    fontstretch: float | _FONTSTRETCHS
    stretch: float | _FONTSTRETCHS
    fontstyle: Literal["normal", "italic", "oblique"]
    style: Literal["normal", "italic", "oblique"]
    fontvariant: Literal["normal", "small-caps"]
    variant: Literal["normal", "small-caps"]
    fontweight: float | _FONTWEIGHTS
    weight: float | _FONTWEIGHTS
    gid: str
    horizontalalignment: Literal["left", "center", "right"]
    ha: Literal["left", "center", "right"]
    in_layout: bool
    label: str
    linespacing: float
    math_fontfamily: str
    mouseover: bool
    multialignment: Literal["left", "center", "right"]
    ma: Literal["left", "center", "right"]
    parse_math: bool
    path_effects: list[AbstractPathEffect]
    picker: None | bool | float | Callable
    position: tuple[float, float]
    rasterized: bool
    rotation: float | Orientation
    rotation_mode: Literal["default", "anchor"] | None
    sketch_params: tuple[float, float, float]
    scale: float
    length: float
    randomness: float
    snap: bool | None
    text: str
    transform: Transform
    transform_rotates_text: bool
    url: str
    usetex: bool | None
    verticalalignment: Literal["bottom", "baseline", "center", "center_baseline", "top"]
    va: Literal["bottom", "baseline", "center", "center_baseline", "top"]
    visible: bool
    wrap: bool
    zorder: float


class PLTSubplotParam(TypedDict, total=False):
    sharex: bool | Literal["none", "all", "row", "col"]
    sharey: bool | Literal["none", "all", "row", "col"]
    squeeze: bool
    width_ratios: Sequence[float] | None
    height_ratios: Sequence[float] | None
    subplot_kw: dict
    gridspec_kw: dict


class AxesImageParam(TypedDict, total=False):
    cmap: str | Colormap
    norm: str | Normalize
    interpolation: Literal[
        "none",
        "antialiased",
        "nearest",
        "bilinear",
        "bicubic",
        "spline16",
        "spline36",
        "hanning",
        "hamming",
        "hermite",
        "kaiser",
        "quadric",
        "catrom",
        "gaussian",
        "bessel",
        "mitchell",
        "sinc",
        "lanczos",
        "blackman",
    ]
    interpolation_stage: Literal["data", "rgba"]
    origin: Literal["upper", "lower"]
    extent: tuple[float, float, float, float]
    filternorm: bool
    filterrad: float
    resample: bool


class IMshowParam(AxesImageParam, total=False):
    aspect: Literal["equal", "auto"] | float | None
    alpha: float
    vmin: float
    vmax: float
    url: str


class QuadmeshParam(TypedDict, total=False):
    agg_filter: Callable[..., tuple[NDArray[np.int_], float, float]]
    alpha: float
    animated: bool
    antialiased: bool
    aa: bool | list[bool]
    antialiaseds: bool | list[bool]
    array: ArrayLike
    capstyle: CapStyleType
    clim: tuple[float, float]
    clip_box: BboxBase | None
    clip_on: bool
    clip_path: Patch | Transform | None
    cmap: Colormap | str | None
    color: ColorType
    edgecolor: ColorType
    ec: ColorType
    edgecolors: ColorType
    facecolor: ColorType
    facecolors: ColorType
    fc: ColorType
    figure: Figure
    gid: str
    hatch: Literal["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    in_layout: bool
    joinstyle: JoinStyleType
    label: str
    linestyle: LineStyleType
    dashes: LineStyleType
    linestyles: LineStyleType
    ls: LineStyleType
    linewidth: float | list[float]
    linewidths: float | list[float]
    lw: float | list[float]
    mouseover: bool
    norm: Normalize | None
    offset_transform: Transform
    transOffset: Transform
    offsets: ArrayLike
    path_effects: list[AbstractPathEffect]
    picker: None | bool | float
    pickradius: float
    rasterized: bool
    sketch_params: tuple[float, float, float]
    scale: float
    randomness: float
    snap: bool | None
    transform: Transform
    url: str
    urls: list[str | None]
    visible: bool
    zorder: float


class PColorMeshKwargs(QuadmeshParam, total=False):
    vmin: float
    vmax: float
    shading: Literal["flat", "nearest", "gouraud", "auto"]


class ProfileViewParam(TypedDict):
    """Kwargs for profile_view."""

    width: int
    height: int
    cmap: str
    log: bool
    profile_view_height: int
    colorbar: bool


class SliceAlongPathKwags(TypedDict, total=False):
    axis_name: str
    resolution: float
    n_points: int
    extend_to_edge: bool


class PlotParamKwargs(MPLPlotKwargs, total=False):
    ax: Axes | None
    shift: float
    x_shift: float
    two_sigma: bool
    figsize: tuple[float, float]


class LabeledFermiSurfaceParam(TypedDict, total=False):
    include_symmetry_points: bool
    include_bz: bool
    fermi_energy: float
    out: str | Path


class HvRefScanParam(LabeledFermiSurfaceParam):
    """Parameter for hf_ref_scan."""

    e_cut: float
    bkg_subtraction: float
