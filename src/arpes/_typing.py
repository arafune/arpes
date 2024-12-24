"""Specialized type annotations for use in PyARPES.

In particular, `DataType` refers to either an xarray.DataArray or xarray.Dataset

`NormalizableDataType` referes to anything that can be tuned into datase,
such as by loading from the cache using an ID.
"""

from __future__ import annotations

import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Required,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    get_args,
)

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
    from pathlib import Path

    from _typeshed import Incomplete
    from matplotlib.artist import Artist
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
    from matplotlib.widgets import AxesWidget, Button, TextBox
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ANGLE",
    "EMISSION_ANGLE",
    "LEGENDLOCATION",
    "MOMENTUM",
    "AnalyzerInfo",
    "ConfigType",
    "CoordsOffset",
    "DataType",
    "NormalizableDataType",
    "Orientation",
    "ReduceMethod",
    "Spectrometer",
    "SpectrumType",
    "WorkSpaceType",
    "XrTypes",
    "flatten_literals",
]


DataType = TypeVar("DataType", xr.DataArray, xr.Dataset)
NormalizableDataType: TypeAlias = DataType | str | uuid.UUID

XrTypes: TypeAlias = xr.DataArray | xr.Dataset

ReduceMethod = Literal["sum", "mean"]

MOMENTUM = Literal["kp", "kx", "ky", "kz"]
EMISSION_ANGLE = Literal["phi", "psi"]
ANGLE = Literal["alpha", "beta", "chi", "theta"] | EMISSION_ANGLE
Orientation = Literal["horizontal", "vertical"]

HIGH_SYMMETRY_POINTS = Literal["G", "X", "Y", "M", "K", "S", "A1", "H", "C", "H1"]

SpectrumType = Literal["cut", "map", "hv_map", "ucut", "spem", "xps"]

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

CoordsOffset: TypeAlias = Literal[
    "alpha_offset",
    "beta_offset",
    "chi_offset",
    "phi_offset",
    "psi_offset",
    "theta_offset",
    "delay_offset",
    "eV_offset",
    "beta",
    "theta",
]


def flatten_literals(literal_type: type[Literal[str]] | Literal[str]) -> set[str]:
    """Recursively flattens a Literal type to extract all string values.

    Args:
        literal_type (type[Literal] | Literal): The Literal type to flatten.

    Returns:
        set[str]: A set of all string values in the Literal type.
    """
    args = get_args(literal_type)
    flattened = set()
    for arg in args:
        if hasattr(arg, "__args__"):
            flattened.update(flatten_literals(arg))
        else:
            flattened.add(arg)
    return flattened


class KspaceCoords(TypedDict, total=False):
    eV: NDArray[np.float64]
    kp: NDArray[np.float64]
    kx: NDArray[np.float64]
    ky: NDArray[np.float64]
    kz: NDArray[np.float64]


def is_dict_kspacecoords(
    a_dict: dict[Hashable, NDArray[np.float64]] | dict[str, NDArray[np.float64]],
) -> TypeGuard[KspaceCoords]:
    """Checks if a dictionary contains k-space coordinates.

    Args:
        a_dict (dict[Hashable, NDArray[np.float64]] | dict[str, NDArray[np.float64]]):
           The dictionary to check.

    Returns:
        TypeGuard[KspaceCoords]: True if the dictionary contains k-space coordinates,
        False otherwise.
    """
    if all(key in {"eV", "kp", "kx", "ky", "kz"} for key in a_dict):
        return all(isinstance(v, np.ndarray) for v in a_dict.values())
    return False


class _InteractiveConfigSettings(TypedDict, total=False):
    main_width: float
    marginal_width: float
    palette: str | Colormap


class ConfigSettings(TypedDict, total=False):
    """TypedDict for arpes.SETTINGS."""

    interactive: _InteractiveConfigSettings
    use_tex: bool


class WorkSpaceType(TypedDict, total=False):
    """TypedDict for arpes.CONFIG["WORKSPACE"]."""

    path: str | Path
    name: str


class CurrentContext(TypedDict, total=False):
    selected_components: list[float]  # in widget.py, selected_components is [0, 1] is default
    selected_indices: list[int]
    sum_data: Incomplete
    map_data: Incomplete
    selector: Incomplete
    integration_region: dict[Incomplete, Incomplete]
    original_data: XrTypes
    data: XrTypes
    widgets: list[dict[str, AxesWidget] | Button]
    points: list[Incomplete]
    rect_next: bool
    axis_button: Button
    axis_X_input: TextBox
    axis_Y_input: TextBox


class ConfigType(TypedDict, total=False):
    """TypedDict for arpes.CONFIG."""

    WORKSPACE: Required[WorkSpaceType]
    CURRENT_CONTEXT: CurrentContext | None  # see widgets.py
    ENABLE_LOGGING: Required[bool]
    LOGGING_STARTED: Required[bool]
    LOGGING_FILE: Required[str | Path | None]


#
# TypedDict for ARPES.attrs
#
class Coordinates(TypedDict, total=False):
    """TypedDict for attrs."""

    x: NDArray[np.float64] | float
    y: NDArray[np.float64] | float
    z: NDArray[np.float64] | float
    alpha: NDArray[np.float64] | float
    beta: NDArray[np.float64] | float
    chi: NDArray[np.float64] | float
    theta: NDArray[np.float64] | float
    psi: NDArray[np.float64] | float
    phi: NDArray[np.float64] | float


class AnalyzerInfo(TypedDict, total=False):
    """TypedDict for attrs.

    see analyzer_info in xarray_extensions.py (around line# 1490)
    """

    analyzer: str
    analyzer_name: str
    lens_mode: str | None
    lens_mode_name: str | None
    acquisition_mode: str
    pass_energy: float
    slit_shape: str | None
    slit_width: float
    slit_number: str | int
    lens_table: None
    parallel_deflectors: bool
    perpendicular_deflectors: bool
    analyzer_type: str | None
    mcp_voltage: float
    work_function: float
    is_slit_vertical: bool
    analyzer_radius: str | float


class _PumpInfo(TypedDict, total=False):
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


class _ProbeInfo(TypedDict, total=False):
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


class _BeamLineInfo(TypedDict, total=False):
    """TypedDict for attrs.

    see beamline_info in xarray_extensions.py
    """

    hv: float | xr.DataArray
    linewidth: float
    photon_polarization: tuple[float, float]
    undulator_info: Incomplete
    repetition_rate: float
    beam_current: float
    entrance_slit: float | str
    exit_slit: float | str
    monochromator_info: dict[str, float]


class BeamLineSettings(TypedDict, total=False):
    exit_slit: float | str
    entrance_slit: float | str
    hv: float
    grating: str | None


class LightSourceInfo(_ProbeInfo, _PumpInfo, _BeamLineInfo, total=False):
    """TypedDict for beamline_info."""

    polarization: float | tuple[float, float] | str
    photon_flux: float
    photocurrent: float
    probe: Incomplete
    probe_detail: Incomplete


class SampleInfo(TypedDict, total=False):
    """TypedDict for attrs.

    see sample_info in xarray_extensions
    """

    id: int | str | None
    sample_name: str | None
    source: str | None
    reflectivity: float


class ScanInfo(TypedDict, total=False):
    time: str
    date: str
    spectrum_type: SpectrumType
    type: str | None
    experimenter: str | None
    sample: str | None
    pressure: float
    temperature: float | Literal["RT", "LT"]
    temperature_cryotip: float


class DAQInfo(TypedDict, total=False):
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
    sweep_settings: Incomplete
    frames_per_slice: int
    frame_duration: float


class Spectrometer(AnalyzerInfo, Coordinates, DAQInfo, total=False):
    type: str
    rad_per_pixel: float
    dof: list[str]
    scan_dof: list[str]
    mstar: float
    dof_type: dict[str, list[str]]
    length: float
    detect_radius: float | str


class ExperimentInfo(
    ScanInfo,
    LightSourceInfo,
    AnalyzerInfo,
    total=False,
):
    analyzer_detail: AnalyzerInfo


class ARPESAttrs(Spectrometer, LightSourceInfo, SampleInfo, total=False):
    angle_unit: Literal["Degrees", "Radians", "deg", "rad"]
    energy_notation: Literal[
        "Binding",
        "Final",
        "Kinetic",
        "kinetic",
        "kinetic energy",
    ]


#
# TypedDict for plotting
#


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
