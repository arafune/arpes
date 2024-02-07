"""Specialized type annotations for use in PyARPES.

In particular, we frequently allow using the `DataType` annotation,
which refers to either an xarray.DataArray or xarray.Dataset.

Additionally, we often use `NormalizableDataType` which
means essentially anything that can be turned into a dataset,
for instance by loading from the cache using an ID, or which is
literally already data.
"""

from __future__ import annotations

import uuid
from typing import (
    TYPE_CHECKING,
    Literal,
    Required,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import matplotlib as mpl
    import numpy as np
    from _typeshed import Incomplete
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import Event
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch
    from matplotlib.patheffects import AbstractPathEffect
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
    from PySide6.QtCore.Qt import Orientation, WindowType
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtWidgets import (
        QWidget,
    )


__all__ = [
    "DataType",
    "NormalizableDataType",
    "xr_types",
    "SPECTROMETER",
    "MOMENTUM",
    "EMISSION_ANGLE",
    "ANGLE",
    "CONFIGTYPE",
    "WORKSPACETYPE",
    "ANALYZERINFO",
]

DataType = TypeVar("DataType", xr.DataArray, xr.Dataset)
NormalizableDataType: TypeAlias = DataType | str | uuid.UUID

xr_types = (xr.DataArray, xr.Dataset)


MOMENTUM = Literal["kp", "kx", "ky", "kz"]
EMISSION_ANGLE = Literal["phi", "psi"]
ANGLE = Literal["alpha", "beta", "chi", "theta"] | EMISSION_ANGLE


class KspaceCoords(TypedDict, total=False):
    eV: NDArray[np.float_]
    kp: NDArray[np.float_]
    kx: NDArray[np.float_]
    ky: NDArray[np.float_]
    kz: NDArray[np.float_]


class ConfigSettings(TypedDict, total=False):
    """TypedDict for arpes.config.SETTINGS."""

    interactive: dict[str, float | str]
    xarray_repr_mod: bool
    use_tex: bool


class WORKSPACETYPE(TypedDict, total=False):
    """TypedDict for arpes.config.CONFIG["WORKSPACE"]."""

    path: str | Path
    name: str


class CURRENTCONTEXT(TypedDict, total=False):
    selected_components: list[float]  # in widget.py, selected_components is [0, 1] is default
    selected_indices: list[int]
    sum_data: Incomplete
    map_data: Incomplete
    selector: Incomplete
    integration_region: dict[Incomplete, Incomplete]
    original_data: xr.DataArray | xr.Dataset
    data: xr.DataArray | xr.Dataset
    widgets: list[mpl.widgets.AxesWidget]
    points: list[Incomplete]
    rect_next: bool
    #
    axis_button: mpl.widgets.Button
    axis_X_input: mpl.widgets.TextBox
    axis_Y_input: mpl.widgets.TextBox


class CONFIGTYPE(TypedDict, total=False):
    """TypedDict for arpes.config.CONFIG."""

    WORKSPACE: Required[WORKSPACETYPE]
    CURRENT_CONTEXT: CURRENTCONTEXT | None  # see widgets.py
    ENABLE_LOGGING: Required[bool]
    LOGGING_STARTED: Required[bool]
    LOGGING_FILE: Required[str | Path | None]


#
# TypedDict for ARPES.attrs
#
class COORDINATES(TypedDict, total=False):
    """TypedDict for attrs."""

    x: NDArray[np.float_] | float
    y: NDArray[np.float_] | float
    z: NDArray[np.float_] | float
    alpha: NDArray[np.float_] | float
    beta: NDArray[np.float_] | float
    chi: NDArray[np.float_] | float
    theta: NDArray[np.float_] | float
    psi: NDArray[np.float_] | float
    phi: NDArray[np.float_] | float


class ANALYZERINFO(TypedDict, total=False):
    """TypedDict for attrs.

    see analyzer_info in xarray_extensions.py
    """

    lens_mode: str | None
    lens_mode_name: str | None
    acquisition_mode: str
    pass_energy: float
    slit_shape: str | None
    slit_width: float
    slit_number: str | int
    lens_table: None
    analyzer_type: str | None
    mcp_voltage: float
    work_function: float
    #
    analyzer_radius: float
    analyzer: str | None
    analyzer_name: str | None
    parallel_deflectors: bool
    perpendicular_deflectors: bool
    #
    is_slit_vertical: bool


class _PUMPINFO(TypedDict, total=False):
    """TypedDict for attrs.

    see pump_info in xarray_extensions.py
    """

    pump_wavelength: float
    pump_energy: float
    pump_fluence: float
    pump_pulse_energy: float
    pump_spot_size: float | tuple[float, float]
    pump_spot_size_x: float
    pump_spot_size_y: float
    pump_profile: Incomplete
    pump_linewidth: float
    pump_duration: float
    pump_polarization: str | tuple[float, float]
    pump_polarization_theta: float
    pump_polarization_alpha: float


class _PROBEINFO(TypedDict, total=False):
    """TypedDict for attrs.

    see probe_info in xarray_extensions.py
    """

    probe_wavelength: float
    probe_energy: float | xr.DataArray
    probe_fluence: float
    probe_pulse_energy: float
    probe_spot_size: float | tuple[float, float]
    probe_spot_size_x: float
    probe_spot_size_y: float
    probe_profile: None
    probe_linewidth: float
    probe_duration: float
    probe_polarization: str | tuple[float, float]
    probe_polarization_theta: float
    probe_polarization_alpha: float


class _BEAMLINEINFO(TypedDict, total=False):
    """TypedDict for attrs.

    see beamline_info in xarray_extensions.py
    """

    hv: float | xr.DataArray
    linewidth: float
    photon_polarization: tuple[float, float]
    undulation_info: Incomplete
    repetition_rate: float
    beam_current: float
    entrance_slit: float | str
    exit_slit: float | str
    monochrometer_info: dict[str, float]


class LIGHTSOURCEINFO(_PROBEINFO, _PUMPINFO, _BEAMLINEINFO, total=False):
    polarization: float | tuple[float, float] | str
    photon_flux: float
    photocurrent: float
    probe: Incomplete
    probe_detail: Incomplete


class SAMPLEINFO(TypedDict, total=False):
    """TypedDict for attrs.

    see sample_info in xarray_extensions
    """

    id: int | str | None
    sample_name: str | None
    source: str | None
    reflectivity: float


class SCANINFO(TypedDict, total=False):
    time: str
    date: str
    spectrum_type: Literal["cut", "map", "hv_map", "ucut", "spem", "xps"]
    type: str | None
    experimenter: str | None
    sample: str | None
    pressure: float
    temperature: float | Literal["RT", "LT"]
    temperature_cryotip: float


class DAQINFO(TypedDict, total=False):
    """TypedDict for attrs.

    see daq_info in xarray_extensions.py
    """

    daq_type: str | None
    region: str | None
    region_name: str | None
    center_energy: float
    prebinning: dict[str, float]
    trapezoidal_correction_strategy: Incomplete
    dither_settings: Incomplete
    sweep_setting: Incomplete
    frames_per_slice: int
    frame_duration: float


class SPECTROMETER(ANALYZERINFO, COORDINATES, DAQINFO, total=False):
    rad_per_pixel: float
    dof: list[str]
    scan_dof: list[str]
    mstar: float
    dof_type: dict[str, list[str]]
    length: float


class EXPERIMENTINFO(
    SCANINFO,
    LIGHTSOURCEINFO,
    ANALYZERINFO,
    total=False,
):
    pass


class ARPESAttrs(SPECTROMETER, LIGHTSOURCEINFO, SAMPLEINFO, total=False):
    angle_unit: Literal["Degrees", "Radians", "deg", "rad"]
    energy_notation: Literal[
        "Binding",
        "Kinetic",
        "kinetic",
        "kinetic energy",
    ]


# TypedDict for Qt


class QSliderARGS(TypedDict, total=False):
    orientation: Orientation
    parent: QWidget | None


class QWidgetARGS(TypedDict, total=False):
    parent: QWidget | None
    f: WindowType


class QPushButtonARGS(TypedDict, total=False):
    icon: QIcon | QPixmap
    text: str
    parent: QWidget | None


#
# TypedDict for plotting
#
class MPLPlotKwargs(TypedDict, total=False):
    scalex: bool
    scaley: bool

    agg_filter: Callable[[NDArray[np.float_], int], tuple[NDArray[np.float_], int, int]]
    alpha: float | None
    animated: bool
    antialiased: bool
    aa: bool
    clip_box: BboxBase | None
    clip_on: bool
    # clip_path: Path | None color: ColorType
    c: ColorType
    dash_capstyle: CapStyleType
    dash_joinstyle: JoinStyleType
    dashes: Sequence[float | None]
    data: NDArray[np.float_]
    drawstyle: DrawStyleType
    figure: Figure
    fillstyle: FillStyleType
    gapcolor: ColorType | None
    gid: str
    in_layout: bool
    label: str
    linestyle: LineStyleType
    ls: LineStyleType
    linewidth: float
    lw: float
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
    mouseover: bool
    path_effects: list[AbstractPathEffect]
    picker: float | Callable[[Artist, Event], tuple[bool, dict]]
    pickradius: float
    rasterized: bool
    sketch_params: tuple[float, float, float]
    scale: float
    length: float
    randomness: float
    snap: bool | None
    solid_capstyle: CapStyleType
    solid_joinstyle: JoinStyleType
    url: str
    visible: bool
    xdata: NDArray[np.float_]
    ydata: NDArray[np.float_]
    zorder: float


class ColorbarParam(TypedDict, total=False):
    alpha: float
    orientation: None | Literal["vertical", "horizontal"]
    ticklocation: Literal["auto", "right", "top", "bottom"]
    drawedge: bool
    extend: Literal["neither", "both", "min", "max"]
    extendfrac: None | Literal["auto"] | float | tuple[float, float] | list[float]
    spacing: Literal["uniform", "proportional"]
    ticks: None | list[float]
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
    agg_filter: Callable[[NDArray[np.float_], int], tuple[NDArray[np.float_], int, int]]
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
    rotation: float | Literal["vertical", "horizontal"]
    rotation_mode: Literal[None, "default", "anchor"]
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
    ax: Axes
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
    antialiased: bool | list[bool]
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
    norm: Normalize | str | None
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
