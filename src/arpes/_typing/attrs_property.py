"""Specialized type annotations for use in PyARPES.

In particular, `DataType` refers to either an xarray.DataArray or xarray.Dataset

`NormalizableDataType` referes to anything that can be tuned into datase,
such as by loading from the cache using an ID.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    TypeAlias,
    TypedDict,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import xarray as xr
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from .base import SpectrumType


class AnalyzerInfo(TypedDict, total=False):
    """TypedDict for attrs.

    see analyzer_info in xarray_extensions.py (around line# 1490)
    """

    analyzer: str
    analyzer_name: str
    lens_mode: str | None
    lens_mode_name: str | None
    acquisition_mode: str | None
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
    entrance_slit: float | str | None
    exit_slit: float | str | None
    monochromator_info: dict[str, float]


class BeamLineSettings(TypedDict, total=False):
    exit_slit: float | str | None
    entrance_slit: float | str | None
    hv: float | xr.DataArray
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
    time: str | None
    date: str | None
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


class KspaceCoords(TypedDict, total=False):
    eV: NDArray[np.float64]
    kp: NDArray[np.float64]
    kx: NDArray[np.float64]
    ky: NDArray[np.float64]
    kz: NDArray[np.float64]


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


class ScanDesc(TypedDict, total=False):
    """TypedDict based class for scan_desc."""

    file: str | Path
    location: str
    path: str | Path
    note: dict[str, str | float]  # used as attrs basically.
    id: int | str
