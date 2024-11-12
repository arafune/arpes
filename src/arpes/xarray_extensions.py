"""Establishes the PyARPES data model by extending the `xarray` types.

This is another core part of PyARPES. It provides a lot of extensions to
what comes out of the box in xarray. Some of these are useful generics,
generally on the .T extension, others collect and manipulate metadata,
interface with plotting routines, provide functional programming utilities,
etc.

If `f` is an ARPES spectrum, then `f.S` should provide a nice representation of your data
in a Jupyter cell. This is a complement to the text based approach that merely printing `f`
offers. Note, as of PyARPES v3.x.y, the xarray version has been bumped and this representation
is no longer necessary as one is provided upstream.

The main accessors are .S, .G, and .F.

The `.S` accessor:
    The `.S` accessor contains functionality related to spectroscopy. Utilities
    which only make sense in this context should be placed here, while more generic
    tools should be placed elsewhere.

The `.G.` accessor:
    This a general purpose collection of tools which exists to provide conveniences over
    what already exists in the xarray data model. As an example, there are various tools
    for simultaneous iteration of data and coordinates here, as well as for vectorized
    application of functions to data or coordinates.

The `.F` accessor:
    This is an accessor which contains tools related to interpreting curve fitting
    results (assumed the return of broadcast_model).
    In particular there are utilities for vectorized extraction of parameters,
    for plotting several curve fits, or for selecting "worst" or "best" fits according
    to some measure.
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import warnings
from collections import OrderedDict
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    Unpack,
)

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from more_itertools import always_reversible
from scipy import ndimage as ndi
from skimage import feature
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

import arpes
import arpes.constants
import arpes.utilities.math
from arpes.constants import TWO_DIMENSION

from ._typing import HighSymmetryPoints, MPLPlotKwargs
from .analysis import param_getter, param_stderr_getter, rebin
from .models.band import MultifitBand
from .plotting.dispersion import (
    LabeledFermiSurfaceParam,
    fancy_dispersion,
    hv_reference_scan,
    labeled_fermi_surface,
    reference_scan_fermi_surface,
    scan_var_reference_plot,
)
from .plotting.fermi_edge import fermi_edge_reference
from .plotting.holoviews import fit_inspection, profile_view
from .plotting.movie import plot_movie
from .plotting.parameter import plot_parameter
from .plotting.spatial import reference_scan_spatial
from .plotting.utils import fancy_labels, remove_colorbars
from .utilities import apply_dataarray
from .utilities.region import DesignatedRegions, normalize_region
from .utilities.xarray import unwrap_xarray_item

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Generator,
        Hashable,
        Iterator,
        Mapping,
        Sequence,
    )

    import lmfit
    from _typeshed import Incomplete
    from holoviews import AdjointLayout
    from matplotlib import animation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import DTypeLike, NDArray

    from ._typing import (
        ANGLE,
        HIGH_SYMMETRY_POINTS,
        AnalyzerInfo,
        BeamLineSettings,
        DAQInfo,
        DataType,
        ExperimentInfo,
        LightSourceInfo,
        PColorMeshKwargs,
        ProfileViewParam,
        SampleInfo,
        ScanInfo,
        XrTypes,
    )
    from .provenance import Provenance

    IncompleteMPL: TypeAlias = Incomplete

__all__ = ["ARPESDataArrayAccessor", "ARPESDatasetAccessor", "ARPESFitToolsAccessor"]

EnergyNotation = Literal["Binding", "Kinetic"]

ANGLE_VARS = ("alpha", "beta", "chi", "psi", "phi", "theta")

DEFAULT_RADII: dict[str, float] = {
    "kp": 0.02,
    "kx": 0.02,
    "ky": 0.02,
    "kz": 0.05,
    "phi": 0.02,
    "beta": 0.02,
    "theta": 0.02,
    "psi": 0.02,
    "eV": 0.05,
    "delay": 0.2,
    "T": 2,
    "temperature": 2,
}

UNSPESIFIED = 0.1

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

T = TypeVar("T")


class ARPESAngleProperty:
    """Class for Angle relateed property.

    This class should not be called directly.

    Attributes:
        _obj (XrTypes): ARPES data

    """

    _obj: XrTypes

    @property
    def angle_unit(self) -> Literal["Degrees", "Radians"]:
        return self._obj.attrs.get("angle_unit", "Radians")

    @angle_unit.setter
    def angle_unit(self, angle_unit: Literal["Degrees", "Radians"]) -> None:
        """Set "angle unit".

        Angle unit should be "Degrees" or "Radians"

        Args:
            angle_unit: Literal["Degrees", "Radians"]
        """
        assert angle_unit in {
            "Degrees",
            "Radians",
        }, "Angle unit should be 'Degrees' or 'Radians'"
        self._obj.attrs["angle_unit"] = angle_unit

        if isinstance(self._obj, xr.Dataset):
            for data_var in self._obj.data_vars.values():
                if "eV" in data_var.dims:
                    data_var.attrs["angle_unit"] = angle_unit

    def swap_angle_unit(self) -> None:
        """Swap angle unit (radians <-> degrees).

        Change the value of angle related objects/variables in attrs and coords
        """
        if self.angle_unit == "Radians" or self.angle_unit.startswith("rad"):
            self.radian_to_degree()
        elif self.angle_unit == "Degrees" or self.angle_unit.startswith("deg"):
            self.degree_to_radian()
        else:
            msg = 'The angle_unit must be "Radians" or "Degrees"'
            raise TypeError(msg)

    def radian_to_degree(self) -> None:
        """Swap angle unit in from Radians to Degrees."""
        self.angle_unit = "Degrees"
        for angle in ANGLE_VARS:
            if angle in self._obj.attrs:
                self._obj.attrs[angle] = np.rad2deg(self._obj.attrs.get(angle, np.nan))
            if angle + "_offset" in self._obj.attrs:
                self._obj.attrs[angle + "_offset"] = np.rad2deg(
                    self._obj.attrs.get(angle + "_offset", np.nan),
                )
            if angle in self._obj.coords:
                self._obj.coords[angle] = np.rad2deg(self._obj.coords[angle])

    def degree_to_radian(self) -> None:
        """Swap angle unit in from Degrees and Radians."""
        self.angle_unit = "Radians"
        for angle in ANGLE_VARS:
            if angle in self._obj.attrs:
                self._obj.attrs[angle] = np.deg2rad(self._obj.attrs.get(angle, np.nan))
            if angle + "_offset" in self._obj.attrs:
                self._obj.attrs[angle + "_offset"] = np.deg2rad(
                    self._obj.attrs.get(angle + "_offset", np.nan),
                )
            if angle in self._obj.coords:
                self._obj.coords[angle] = np.deg2rad(self._obj.coords[angle])

    def lookup_coord(self, name: str) -> xr.DataArray | float:
        """Return the coordinates, if not return np.nan."""
        if name in self._obj.coords:
            return unwrap_xarray_item(self._obj.coords[name])
        self._obj.coords[name] = np.nan
        return np.nan

    @property
    def sample_angles(
        self,
    ) -> tuple[
        xr.DataArray | float,
        xr.DataArray | float,
        xr.DataArray | float,
        xr.DataArray | float,
        xr.DataArray | float,
        xr.DataArray | float,
    ]:
        r"""The angle (:math:`\beta,\,\theta,\,\chi,\,\phi,\,\psi,\,\alpha`) values.

        Returns: tuple[xr.DataArray | float, ...]
            beta, theta, chi, phi, psi, alpha
        """
        return (
            # manipulator
            self.lookup_coord("beta"),
            self.lookup_coord("theta"),
            self.lookup_coord("chi"),
            # analyzer
            self.lookup_coord("phi"),
            self.lookup_coord("psi"),
            self.lookup_coord("alpha"),
        )


class ARPESPhysicalProperty:
    """Class for ARPES physical properties.

    This class should not be called directly.

    Attributes:
        _obj (XrTypes): ARPES data
    """

    _obj: XrTypes

    @property
    def work_function(self) -> float:
        """The work function of the sample, if present in metadata.

        Otherwise, uses something approximate.

        Note:
            This "work_function" should *NOT* be used for k-conversion!
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if "sample_workfunction" in self._obj.attrs:
            return self._obj.attrs["sample_workfunction"]
        return 4.3

    @property
    def analyzer_work_function(self) -> float:
        """The work function of the analyzer, if present in metadata.

        otherwise, use appropriate value.

        Note:
            Use this value for k-conversion.
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if "workfunction" in self._obj.attrs:
            return self._obj.attrs["workfunction"]
        return 4.401

    @property
    def inner_potential(self) -> float:
        """The inner potential, if present in metadata. Otherwise, 10 eV is assumed."""
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if "inner_potential" in self._obj.attrs:
            return self._obj.attrs["inner_potential"]
        return 10

    @property
    def sherman_function(self) -> float:
        """Sherman function from attributes.

        Returns: float
            Sharman function

        Raises: ValueError
            When no Sherman function related value is found.

        Todo:
            Test, Consider if it should be in "S"
        """
        for option in ["sherman", "sherman_function", "SHERMAN"]:
            if option in self._obj.attrs:
                return self._obj.attrs[option]
        msg = "No Sherman function could be found on the data. Is this a spin dataset?"
        raise ValueError(msg)

    @property
    def hv(self) -> float | xr.DataArray:
        """The photon energy for excitation.

        Returns: float | xr.DataArray
            Photon energy in eV unit.  (for hv_map type, xr.DataArray is returned.)
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        try:
            return float(self._obj.coords["hv"])
        except TypeError:
            return self._obj.coords["hv"]

    @property
    def temp(self) -> float | Literal["RT", "LT"]:
        """The temperature at which an experiment was performed."""
        prefered_attrs = [
            "TA",
            "ta",
            "t_a",
            "T_A",
            "T_1",
            "t_1",
            "t1",
            "T1",
            "temp",
            "temp_sample",
            "temperature",
            "temp_cryotip",
            "temperature_sensor_b",
            "temperature_sensor_a",
            "temperature_cryotip",
        ]
        for attr in prefered_attrs:
            if attr in self._obj.attrs:
                return self._obj.attrs[attr]
        msg = "Could not read temperature off any standard attr"
        logger.debug(msg, stacklevel=2)
        return np.nan

    @property
    def experimental_conditions(
        self,
    ) -> ExperimentInfo:
        """The experimental condition: hv, polarization, temperature.

        Use this property in plotting/annotations.py/conditions
        """
        return {
            "hv": self.hv,
            "polarization": self.polarization,
            "temperature": self.temp,
        }

    @property
    def polarization(self) -> float | str | tuple[float, float]:
        """The light polarization information.

        Todo:
            Test
        """
        if "epu_pol" in self._obj.attrs:
            # merlin: TODO normalize these
            # check and complete
            try:
                return {
                    0: "p",
                    1: "rc",
                    2: "s",
                }.get(int(self._obj.attrs["epu_pol"]), np.nan)
            except ValueError:
                return self._obj.attrs["epu_pol"]
        if "pol" in self._obj.attrs:
            return self._obj.attrs["pol"]
        return np.nan

    @property
    def sample_pos(self) -> tuple[float, float, float]:
        """The sample position, x, y, and z."""
        return (
            float(self._obj.attrs["x"]),
            float(self._obj.attrs["y"]),
            float(self._obj.attrs["z"]),
        )

    @property
    def probe_polarization(self) -> tuple[float, float]:
        """The probe polarization of the UV/x-ray source."""
        return (
            self._obj.attrs.get("probe_polarization_theta", np.nan),
            self._obj.attrs.get("probe_polarization_alpha", np.nan),
        )

    @property
    def pump_polarization(self) -> tuple[float, float]:
        """The pump polarization for Tr-ARPES experiments."""
        return (
            self._obj.attrs.get("pump_polarization_theta", np.nan),
            self._obj.attrs.get("pump_polarization_alpha", np.nan),
        )

    @property
    def energy_notation(self) -> EnergyNotation:
        """The energy notation ("Binding" energy or "Kinetic" energy).

        Note:
            The "Kinetic" energy refers to the Fermi level, not the vacuum level.
        """
        if "energy_notation" in self._obj.attrs:
            if self._obj.attrs["energy_notation"] in {
                "Kinetic",
                "kinetic",
                "kinetic energy",
                "Kinetic energy",
            }:
                self._obj.attrs["energy_notation"] = "Kinetic"
                return "Kinetic"
            return "Binding"
        self._obj.attrs["energy_notation"] = self._obj.attrs.get("energy_notation", "Binding")
        return "Binding"

    def switch_energy_notation(self, nonlinear_order: int = 1) -> None:
        """Switch the energy notation between binding and kinetic.

        Args:
            nonlinear_order (int): order of the nonliniarity, default to 1
        """
        if self._obj.coords["hv"].ndim == 0:
            if self.energy_notation == "Binding":
                self._obj.coords["eV"] = (
                    self._obj.coords["eV"] + nonlinear_order * self._obj.coords["hv"]
                )
                self._obj.attrs["energy_notation"] = "Kinetic"
            elif self.energy_notation == "Kinetic":
                self._obj.coords["eV"] = (
                    self._obj.coords["eV"] - nonlinear_order * self._obj.coords["hv"]
                )
                self._obj.attrs["energy_notation"] = "Binding"
        else:
            msg = "Not impremented yet."
            raise RuntimeError(msg)


class ARPESInfoProperty(ARPESPhysicalProperty):
    """Class for Information Property.

    This class should not be called directly.

    Attributes:
        _obj (XrTypes): ARPES data
    """

    _obj: XrTypes

    @property
    def scan_name(self) -> str:
        """The scan name.

        Returns: (str)
            If "scan" or "file" is set in attrs, return the file name.
            If they are not set, return "id" if "id" is set.
        """
        for option in ["scan", "file"]:
            if option in self._obj.attrs:
                return Path(self._obj.attrs[option]).name

        id_code: str | int | None = self._obj.attrs.get("id")

        return f"ID: {id_code}" if id_code is not None else "No ID"

    @property
    def label(self) -> str:
        return str(self._obj.attrs.get("description", self.scan_name))

    @property
    def endstation(self) -> str:
        """Alias for the location attribute used to load the data.

        Returns:
            The name of loader/location which was used to load data.
        """
        return str(self._obj.attrs["location"])

    @property
    def sample_info(self) -> SampleInfo:
        """Sample info property.

        Returns (SampleInfo):
        """
        sample_info: SampleInfo = {
            "id": self._obj.attrs.get("sample_id"),
            "sample_name": self._obj.attrs.get("sample_name"),
            "source": self._obj.attrs.get("sample_source"),
            "reflectivity": self._obj.attrs.get("sample_reflectivity", np.nan),
        }
        return sample_info

    @property
    def scan_info(self) -> ScanInfo:
        """Scan information, measurement data/time, scan type, and sample name, etc."""
        scan_info: ScanInfo = {
            "time": self._obj.attrs.get("time", None),
            "date": self._obj.attrs.get("date", None),
            "type": self.scan_type,
            "spectrum_type": self.spectrum_type,
            "experimenter": self._obj.attrs.get("experimenter"),
            "sample": self._obj.attrs.get("sample_name"),
        }
        return scan_info

    @property
    def experiment_info(self) -> ExperimentInfo:
        """Experiment information property, such as temperature, pressure, etc."""
        experiment_info: ExperimentInfo = {
            "temperature": self.temp,
            "temperature_cryotip": self._obj.attrs.get("temperature_cryotip", np.nan),
            "pressure": self._obj.attrs.get("pressure", np.nan),
            "polarization": self.probe_polarization,
            "photon_flux": self._obj.attrs.get("photon_flux", np.nan),
            "photocurrent": self._obj.attrs.get("photocurrent", np.nan),
            "probe": self._obj.attrs.get("probe"),
            "probe_detail": self._obj.attrs.get("probe_detail"),
            "analyzer_detail": self.analyzer_detail,
        }
        return experiment_info

    @property
    def pump_info(self) -> LightSourceInfo:
        """Pump pulse information property."""
        pump_info: LightSourceInfo = {
            "pump_wavelength": self._obj.attrs.get("pump_wavelength", np.nan),
            "pump_energy": self._obj.attrs.get("pump_energy", np.nan),
            "pump_fluence": self._obj.attrs.get("pump_fluence", np.nan),
            "pump_pulse_energy": self._obj.attrs.get("pump_pulse_energy", np.nan),
            "pump_spot_size": (
                self._obj.attrs.get("pump_spot_size_x", np.nan),
                self._obj.attrs.get("pump_spot_size_y", np.nan),
            ),
            "pump_profile": self._obj.attrs.get("pump_profile"),
            "pump_linewidth": self._obj.attrs.get("pump_linewidth", np.nan),
            "pump_duration": self._obj.attrs.get("pump_duration", np.nan),
            "pump_polarization": self.pump_polarization,
        }
        return pump_info

    @property
    def probe_info(self) -> LightSourceInfo:
        """Probe pulse information property.

        Returns (LightSourceInfo):
        """
        probe_info: LightSourceInfo = {
            "probe_wavelength": self._obj.attrs.get("probe_wavelength", np.nan),
            "probe_energy": self.hv,
            "probe_fluence": self._obj.attrs.get("probe_fluence", np.nan),
            "probe_pulse_energy": self._obj.attrs.get("probe_pulse_energy", np.nan),
            "probe_spot_size": (
                self._obj.attrs.get("probe_spot_size_x", np.nan),
                self._obj.attrs.get("probe_spot_size_y", np.nan),
            ),
            "probe_profile": self._obj.attrs.get("probe_profile"),
            "probe_linewidth": self._obj.attrs.get("probe_linewidth", np.nan),
            "probe_duration": self._obj.attrs.get("probe_duration", np.nan),
            "probe_polarization": self.probe_polarization,
        }
        return probe_info

    @property
    def laser_info(self) -> LightSourceInfo:
        """Laser information property, both pump and probe properties."""
        return {
            **self.probe_info,
            **self.pump_info,
            "repetition_rate": self._obj.attrs.get("repetition_rate", np.nan),
        }

    @property
    def analyzer_info(self) -> AnalyzerInfo:
        """General information about the photoelectron analyzer used."""
        analyzer_info: AnalyzerInfo = {
            "lens_mode": self._obj.attrs.get("lens_mode"),
            "lens_mode_name": self._obj.attrs.get("lens_mode_name"),
            "acquisition_mode": self._obj.attrs.get("acquisition_mode", None),
            "pass_energy": self._obj.attrs.get("pass_energy", np.nan),
            "slit_shape": self._obj.attrs.get("slit_shape", None),
            "slit_width": self._obj.attrs.get("slit_width", np.nan),
            "slit_number": self._obj.attrs.get("slit_number", np.nan),
            "lens_table": self._obj.attrs.get("lens_table"),
            "analyzer_type": self._obj.attrs.get("analyzer_type"),
            "mcp_voltage": self._obj.attrs.get("mcp_voltage", np.nan),
            "work_function": self._obj.S.analyzer_work_function,
        }
        return analyzer_info

    @property
    def daq_info(self) -> DAQInfo:
        """General information about the acquisition settings for an ARPES experiment."""
        daq_info: DAQInfo = {
            "daq_type": self._obj.attrs.get("daq_type"),
            "region": self._obj.attrs.get("daq_region"),
            "region_name": self._obj.attrs.get("daq_region_name"),
            "center_energy": self._obj.attrs.get("daq_center_energy", np.nan),
            "prebinning": self.prebinning,
            "trapezoidal_correction_strategy": self._obj.attrs.get(
                "trapezoidal_correction_strategy",
            ),
            "dither_settings": self._obj.attrs.get("dither_settings"),
            "sweep_settings": self.sweep_settings,
            "frames_per_slice": self._obj.attrs.get("frames_per_slice", np.nan),
            "frame_duration": self._obj.attrs.get("frame_duration", np.nan),
        }
        return daq_info

    @property
    def beamline_info(self) -> LightSourceInfo:
        """Information about the beamline or light source used for a measurement."""
        beamline_info: LightSourceInfo = {
            "hv": self.hv,
            "linewidth": self._obj.attrs.get("probe_linewidth", np.nan),
            "photon_polarization": self.probe_polarization,
            "undulator_info": self.undulator_info,
            "repetition_rate": self._obj.attrs.get("repetition_rate", np.nan),
            "beam_current": self._obj.attrs.get("beam_current", np.nan),
            "entrance_slit": self._obj.attrs.get("entrance_slit", None),
            "exit_slit": self._obj.attrs.get("exit_slit", None),
            "monochromator_info": self.monochromator_info,
        }
        return beamline_info

    @property
    def sweep_settings(self) -> dict[str, xr.DataArray | NDArray[np.float64] | float | None]:
        """For datasets acquired with swept acquisition settings, provides those settings."""
        return {
            "high_energy": self._obj.attrs.get("sweep_high_energy"),
            "low_energy": self._obj.attrs.get("sweep_low_energy"),
            "n_sweeps": self._obj.attrs.get("n_sweeps"),
            "step": self._obj.attrs.get("sweep_step"),
        }

    @property
    def prebinning(self) -> dict[str, Any]:
        """Information about the prebinning performed during scan acquisition."""
        prebinning = {}
        for d in self._obj.indexes:
            if f"{d}_prebinning" in self._obj.attrs:
                prebinning[d] = self._obj.attrs[f"{d}_prebinning"]

        return prebinning  # type: ignore [return-value]  # because I (RA) don't know the format of FITS.

    @property
    def monochromator_info(self) -> dict[str, float]:
        """Details about the monochromator used on the UV/x-ray source."""
        return {
            "grating_lines_per_mm": self._obj.attrs.get("grating_lines_per_mm", np.nan),
        }

    @property
    def undulator_info(self) -> dict[str, str | float | None]:
        """Details about the undulator for data performed at an undulator source."""
        return {
            "gap": self._obj.attrs.get("undulator_gap"),
            "z": self._obj.attrs.get("undulator_z"),
            "harmonic": self._obj.attrs.get("undulator_harmonic"),
            "polarization": self._obj.attrs.get("undulator_polarization"),
            "type": self._obj.attrs.get("undulator_type"),
        }

    @property
    def analyzer_detail(self) -> AnalyzerInfo:
        """Details about the analyzer, its capabilities, and metadata."""
        return {
            "analyzer_name": self._obj.attrs.get(
                "analyzer_name",
                self._obj.attrs.get("analyzer", ""),
            ),
            "parallel_deflectors": self._obj.attrs.get("parallel_deflectors", False),
            "perpendicular_deflectors": self._obj.attrs.get("perpendicular_deflectors", False),
            "analyzer_type": self._obj.attrs.get("analyzer_type", ""),
            "analyzer_radius": self._obj.attrs.get("analyzer_radius", np.nan),
        }

    @property
    def scan_type(self) -> str | None:
        """Scan type (DAQ type)."""
        scan_type = self._obj.attrs.get("daq_type")
        if scan_type:
            return scan_type
        return None

    @property
    def spectrum_type(self) -> Literal["cut", "map", "hv_map", "ucut", "spem", "xps"]:
        """Spectrum type (cut, map, hv_map, ucut, spem and xps)."""
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if self._obj.attrs.get("spectrum_type"):
            return self._obj.attrs["spectrum_type"]
        dim_types = {
            ("eV",): "xps",
            ("eV", "phi"): "cut",
            # this should check whether the other angular axis perpendicular to scan axis?
            ("eV", "phi", "beta"): "map",
            ("eV", "phi", "theta"): "map",
            ("eV", "hv", "phi"): "hv_map",
            # kspace
            ("eV", "kp"): "cut",
            ("eV", "kx", "ky"): "map",
            ("eV", "kp", "kz"): "hv_map",
        }
        dims: tuple = tuple(sorted(str(dim) for dim in self._obj.dims))
        if dims in dim_types:
            dim_type = dim_types.get(dims)
        else:
            msg = "Cannot determine spectrum type"
            raise TypeError(msg)

        def _dim_type_check(
            dim_type: str | None,
        ) -> TypeGuard[Literal["cut", "map", "hv_map", "ucut", "spem", "xps"]]:
            return dim_type in {"cut", "map", "hv_map", "ucut", "spem", "xps"}

        if _dim_type_check(dim_type):
            return dim_type
        msg = "Dimension type may be incorrect"
        raise TypeError(msg)


class ARPESOffsetProperty(ARPESAngleProperty):
    """Class for offset value propertiy.

    This class should not be called directly.

    Attributes:
        _obj (XrTypes): ARPES data
    """

    _obj: XrTypes

    def symmetry_points(
        self,
    ) -> dict[HIGH_SYMMETRY_POINTS, dict[str, float]]:
        """Return the dict object about symmetry point such as G-point in the ARPES data.

        The original version was something complicated, but the coding seemed to be in
        process and the purpose was unclear, so it was streamlined considerably.


        Returns (dict[HIGH_SYMMETRY_POINTS, dict[str, float]]):
            Dict object representing the symmpetry points in the ARPES data.

        Raises:
            RuntimeError: When the label of high symmetry_points in arr.attrs[symmetry_points] is
                not in HighSymmetryPoints declared in _typing.py

        Examples:
            symmetry_points = {"G": {"phi": 0.405}}
        """
        symmetry_points: dict[str, dict[str, float]] = {}
        our_symmetry_points = self._obj.attrs.get("symmetry_points", {})

        symmetry_points.update(our_symmetry_points)

        def is_key_high_sym_points(
            symmetry_points: dict[str, dict[str, float]],
        ) -> TypeGuard[dict[HIGH_SYMMETRY_POINTS, dict[str, float]]]:
            return all(key in HighSymmetryPoints for key in symmetry_points)

        if is_key_high_sym_points(symmetry_points):
            return symmetry_points
        msg = "Check the label of High symmetry points.\n"
        msg += f"The allowable labels are: f{HighSymmetryPoints}\n"
        msg += "If you really need the new label, "
        msg += "modify HighSymmetryPoints in _typing.py (and pull-request)."
        raise RuntimeError(msg)

    @property
    def logical_offsets(self) -> dict[str, float | xr.DataArray]:
        """The logical offsets of the sample position.

        Returns:
            dict object of long_[x, y, z] + physical_long_[x, y, z]

        Todo:
            Tests
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if "long_x" not in self._obj.coords:
            msg = "Logical offsets can currently only be accessed for hierarchical"
            msg += " motor systems like nanoARPES."
            raise ValueError(
                msg,
            )
        return {
            "x": self._obj.coords["long_x"] - self._obj.coords["physical_long_x"],
            "y": self._obj.coords["long_y"] - self._obj.coords["physical_long_y"],
            "z": self._obj.coords["long_z"] - self._obj.coords["physical_long_z"],
        }

    @property
    def offsets(self) -> dict[str, float]:
        """The offset values."""
        return {
            str(coord): self.lookup_offset(str(coord))
            for coord in self._obj.coords
            if f"{coord}_offset" in self._obj.attrs
        }

    def lookup_offset_coord(self, name: str) -> xr.DataArray | float:
        """Return the offset coordinate."""
        return self.lookup_coord(name) - self.lookup_offset(name)

    def lookup_offset(self, attr_name: str) -> float:
        """Return the offset value."""
        symmetry_points = self.symmetry_points()
        assert isinstance(symmetry_points, dict)
        if "G" in symmetry_points:
            gamma_point = symmetry_points["G"]  # {"phi": 0.405}  (cut)
            if attr_name in gamma_point:
                return gamma_point[attr_name]

        offset_name = attr_name + "_offset"
        if offset_name in self._obj.attrs:
            return self._obj.attrs[offset_name]

        return self._obj.attrs.get("data_preparation", {}).get(offset_name, 0)

    @property
    def beta_offset(self) -> float:
        r"""Offset of :math:`\beta` angle."""
        return self.lookup_offset("beta")

    @property
    def psi_offset(self) -> float:
        r"""Offset of :math:`\psi` angle."""
        return self.lookup_offset("psi")

    @property
    def theta_offset(self) -> float:
        r"""Offset of :math:`\theta` angle."""
        return self.lookup_offset("theta")

    @property
    def phi_offset(self) -> float:
        r"""Offset of :math:`\phi` angle."""
        return self.lookup_offset("phi")

    @property
    def chi_offset(self) -> float:
        r"""Offset of :math:`\chi` angle."""
        return self.lookup_offset("chi")

    @contextlib.contextmanager
    def with_rotation_offset(self, offset: float) -> Generator:
        """Temporarily rotates the chi_offset by `offset`.

        Args:
            offset (float): offset value about chi.

        Todo:
            Test
        """
        old_chi_offset = self.offsets.get("chi", 0)
        self.apply_offsets({"chi": old_chi_offset + offset})
        yield old_chi_offset + offset
        self.apply_offsets({"chi": old_chi_offset})

    def apply_offsets(self, offsets: dict[ANGLE, float]) -> None:
        assert isinstance(self._obj, xr.Dataset | xr.DataArray)
        for k, v in offsets.items():
            self._obj.attrs[f"{k}_offset"] = v

    @property
    def iter_own_symmetry_points(self) -> Iterator[tuple[HIGH_SYMMETRY_POINTS, dict[str, float]]]:
        sym_points = self.symmetry_points()
        yield from sym_points.items()

    @property
    def is_slit_vertical(self) -> bool:
        """Infers whether the scan is taken on an analyzer with vertical slit.

        Caveat emptor: this assumes that the alpha coordinate is not some intermediate value.

        Returns:
            True if the alpha value is consistent with a vertical slit analyzer. False otherwise.
        """
        angle_tolerance = 1.0
        if self.angle_unit.startswith("Deg") or self.angle_unit.startswith("deg"):
            return float(np.abs(self.lookup_offset_coord("alpha") - 90.0)) < angle_tolerance
        return float(np.abs(self.lookup_offset_coord("alpha") - np.pi / 2)) < float(
            np.deg2rad(
                angle_tolerance,
            ),
        )


class ARPESProvenanceProperty:
    _obj: XrTypes

    def short_history(self, key: str = "by") -> list:
        """Return the short version of history.

        Args:
            key (str): key str in recored dict of self.history.  (default: "by")
        """
        return [h["record"][key] if isinstance(h, dict) else h for h in self.history]  # type: ignore[literal-required]

    @property
    def is_differentiated(self) -> bool:
        """Return True if the spectrum is differentiated data.

        Returns: bool

        Todo:
            Test
        """
        short_history = self.short_history()
        if "dn_along_axis" in short_history:
            return True
        if any(by_keyword.startswith("curvature") for by_keyword in short_history):
            return True
        return any(by_keyword.startswith("minimum_gradient") for by_keyword in short_history)

    @property
    def history(self) -> list[Provenance]:
        provenance_recorded = self._obj.attrs.get("provenance", None)

        def unlayer(
            prov: Provenance | None | str,
        ) -> tuple[list[Provenance], Provenance | str | None]:
            if prov is None:
                return [], None  # tuple[list[Incomplete] | None]
            if isinstance(prov, str):
                warnings.warn("provenance should be dict type object.", stacklevel=2)
                return [prov], None
            first_layer: Provenance = copy.copy(prov)

            rest = first_layer.pop("parents_provenance", None)
            if isinstance(rest, list):
                warnings.warn(
                    "Encountered multiple parents in history extraction, "
                    "throwing away all but the first.",
                    stacklevel=2,
                )
                rest = rest[0] if rest else None

            return [first_layer], rest

        def _unwrap_provenance(prov: Provenance | None) -> list[Provenance]:
            if prov is None:
                return []

            first, rest = unlayer(
                prov,
            )

            return first + _unwrap_provenance(rest)

        return _unwrap_provenance(provenance_recorded)

    @property
    def parent_id(self) -> int | str | None:
        if not self.history:
            return None
        assert self.history is not None
        for a_history in reversed(self.history):
            if "parent_id" in a_history:
                return a_history["parent_id"]
        return None


class ARPESPropertyBase(ARPESInfoProperty, ARPESOffsetProperty, ARPESProvenanceProperty):
    _obj: XrTypes

    @property
    def is_spatial(self) -> bool:
        """Infers whether a given scan has real-space dimensions (SPEM or u/nARPES).

        Returns:
            True if the data is explicltly a "ucut" or "spem" or contains "X", "Y", or "Z"
            dimensions. False otherwise.
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if self.spectrum_type in {"ucut", "spem"}:
            return True

        return any(d in {"X", "Y", "Z"} for d in self._obj.dims)

    @property
    def is_kspace(self) -> bool:
        """Infers whether the scan is k-space converted or not.

        Because of the way this is defined, it will return
        true for XPS spectra, which I suppose is true but trivially.

        Returns:
            True if the data is k-space converted. False otherwise.
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        return not any(d in {"phi", "theta", "beta", "angle"} for d in self._obj.dims)

    @property
    def reference_settings(self) -> dict[str, Any]:
        settings = self.spectrometer_settings or {}

        settings.update(
            {
                "hv": self.hv,
            },
        )

        return settings

    @property
    def beamline_settings(self) -> BeamLineSettings:
        settings: BeamLineSettings = {}
        settings["entrance_slit"] = self._obj.attrs.get("entrance_slit", np.nan)
        settings["exit_slit"] = self._obj.attrs.get("exit_slit", np.nan)
        settings["hv"] = self._obj.attrs.get(
            "exit_slit",
            self._obj.attrs.get("photon_energy", np.nan),
        )
        settings["grating"] = self._obj.attrs.get("grating", None)

        return settings

    @property
    def spectrometer_settings(self) -> dict[str, Any]:
        find_keys = {
            "lens_mode": {
                "lens_mode",
            },
            "pass_energy": {
                "pass_energy",
            },
            "scan_mode": {
                "scan_mode",
            },
            "scan_region": {
                "scan_region",
            },
            "slit": {
                "slit",
                "slit_plate",
            },
        }
        settings = {}
        for key, options in find_keys.items():
            for option in options:
                if option in self._obj.attrs:
                    settings[key] = self._obj.attrs[option]
                    break

        if isinstance(settings.get("slit"), float):
            settings["slit"] = int(round(settings["slit"]))

        return settings

    @property
    def full_coords(
        self,
    ) -> xr.Coordinates:
        """Return the coordinate.

        Returns: xr.Coordinates
            Coordinates data.
        """
        full_coords: xr.Coordinates

        full_coords = xr.Coordinates(dict(zip(["x", "y", "z"], self.sample_pos, strict=True)))
        full_coords.update(
            dict(
                zip(
                    ["beta", "theta", "chi", "phi", "psi", "alpha"],
                    self.sample_angles,
                    strict=True,
                ),
            ),
        )
        full_coords.update(
            {
                "hv": self.hv,
            },
        )
        full_coords.update(self._obj.coords)
        return full_coords


class ARPESProperty(ARPESPropertyBase):
    _obj: XrTypes

    @staticmethod
    def dict_to_html(d: Mapping[str, float | str]) -> str:
        """Returnn html format of dict object.

        Args:
            d: dict object

        Returns:
            html representation of dict object

        Todo:
            Test
        """
        return """
        <table>
          <thead>
            <tr>
              <th>Key</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """.format(
            rows="".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in d.items()]),
        )

    @staticmethod
    def _repr_html_full_coords(
        coords: xr.Coordinates,
    ) -> str:
        significant_coords = {}
        for k, v in coords.items():
            if v is None:
                continue
            if np.any(np.isnan(v)):
                continue
            significant_coords[k] = v

        def coordinate_dataarray_to_flat_rep(value: xr.DataArray) -> str | float:
            if not isinstance(value, xr.DataArray | DataArrayCoordinates | DatasetCoordinates):
                return value
            if len(value.dims) == 0:
                tmp = "<span>{var:.5g}</span>"
                return tmp.format(var=value.values)
            tmp = "<span>{min:.3g}<strong> to </strong>{max:.3g}"
            tmp += "<strong> by </strong>{delta:.3g}</span>"
            return tmp.format(
                min=value.min().item(),
                max=value.max().item(),
                delta=value.values[1] - value.values[0],
            )

        return ARPESProperty.dict_to_html(
            {str(k): coordinate_dataarray_to_flat_rep(v) for k, v in significant_coords.items()},
        )

    def _repr_html_spectrometer_info(self) -> str:
        ordered_settings = OrderedDict(self.spectrometer_settings)

        return ARPESProperty.dict_to_html(ordered_settings)

    @staticmethod
    def _repr_html_experimental_conditions(conditions: ExperimentInfo) -> str:
        """Return the experimental conditions with html format.

        Args:
            conditions (ExperimentInfo): self.confitions is usually used.

        Returns (str):
            html representation of the experimental conditions.
        """

        def _experimentalinfo_to_dict(conditions: ExperimentInfo) -> dict[str, str]:
            transformed_dict: dict[str, str] = {}
            for k, v in conditions.items():
                if k == "polarrization":
                    assert isinstance(v, (float | str))
                    transformed_dict[k] = {
                        "p": "Linear Horizontal",
                        "s": "Linear Vertical",
                        "rc": "Right Circular",
                        "lc": "Left Circular",
                        "s-p": "Linear Dichroism",
                        "p-s": "Linear Dichroism",
                        "rc-lc": "Circular Dichroism",
                        "lc-rc": "Circular Dichroism",
                    }.get(str(v), str(v))
                if k == "temp":
                    if isinstance(v, float) and not np.isnan(v):
                        transformed_dict[k] = f"{v} Kelvin"
                    elif isinstance(v, str):
                        transformed_dict[k] = v
                if k == "hv":
                    if isinstance(v, xr.DataArray):
                        max_hv: float = v.max().item()
                        min_hv: float = v.min().item()
                        transformed_dict[k] = (
                            f"<strong> from </strong> {min_hv} <strong>  to </strong> {max_hv} eV"
                        )
                    elif isinstance(v, float) and not np.isnan(v):
                        transformed_dict[k] = f"{v} eV"
            return transformed_dict

        transformed_dict = _experimentalinfo_to_dict(conditions)
        return ARPESProperty.dict_to_html(transformed_dict)

    def _repr_html_(self) -> str:
        """Return html representation of ARPES data.

        Returns:
            html representation.
        """
        skip_data_vars = {
            "time",
        }

        if isinstance(self._obj, xr.Dataset):
            to_plot = [str(k) for k in self._obj.data_vars if k not in skip_data_vars]
            to_plot = [str(k) for k in to_plot if 1 <= len(self._obj[k].dims) < 3]  # noqa: PLR2004
            to_plot = to_plot[:5]

            if to_plot:
                _, ax = plt.subplots(
                    nrows=1,
                    ncols=len(to_plot),
                    figsize=(len(to_plot) * 3, 3),
                )
                if len(to_plot) == 1:
                    ax = [ax]

                for i, plot_var in enumerate(to_plot):
                    spectrum = self._obj[plot_var]
                    spectrum.S.transpose_to_front("eV").plot(ax=ax[i])
                    fancy_labels(ax[i])
                    ax[i].set_title(plot_var.replace("_", " "))

                remove_colorbars()

        elif 1 <= len(self._obj.dims) < 3:  # noqa: PLR2004
            _, ax = plt.subplots(1, 1, figsize=(4, 3))
            spectrum = self._obj
            spectrum.S.transpose_to_front("eV").plot(ax=ax)
            fancy_labels(ax, data=self._obj)
            ax.set_title("")

            remove_colorbars()
        wrapper_style = 'style="display: flex; flex-direction: row;"'

        if "id" in self._obj.attrs:
            name = "ID: " + str(self._obj.attrs["id"])[:9] + "..."
        else:
            name = "No name"

        warning = ""

        if len(self._obj.attrs) < 10:  # noqa: PLR2004
            warning = ':  <span style="color: red;">Few Attributes, Data Is Summed?</span>'

        return f"""
        <header><strong>{name}{warning}</strong></header>
        <div {wrapper_style}>
        <details open>
            <summary>Experimental Conditions</summary>
            {self._repr_html_experimental_conditions(self.experimental_conditions)}
        </details>
        <details open>
            <summary>Full Coordinates</summary>
            {self._repr_html_full_coords(self.full_coords)}
        </details>
        <details open>
            <summary>Spectrometer</summary>
            {self._repr_html_spectrometer_info()}
        </details>
        </div>
        """


class ARPESAccessorBase(ARPESProperty):
    """Base class for the xarray extensions in PyARPES."""

    def __init__(self, xarray_obj: XrTypes) -> None:
        self._obj = xarray_obj

    def find(self, name: str) -> list[str]:
        """Return the property names containing the "name".

        Args:
            name (str): string to find.

        Returns: list[str]
            Property list
        """
        return [n for n in dir(self) if name in n]

    def transpose_to_front(self, dim: str) -> XrTypes:
        """Transpose the dimensions (to front).

        Args:
            dim: dimension to front

        Returns: (XrTypes)
            Transposed ARPES data

        Tooo:
            Test
        """
        dims = list(self._obj.dims)
        assert dim in dims
        dims.remove(dim)
        return self._obj.transpose(*([dim, *dims]))

    def transpose_to_back(self, dim: str) -> XrTypes:
        """Transpose the dimensions (to back).

        Args:
            dim: dimension to back

        Returns: (XrTypes)
            Transposed ARPES data.

        Tooo:
            Test
        """
        dims = list(self._obj.dims)
        assert dim in dims
        dims.remove(dim)
        return self._obj.transpose(*([*dims, dim]))

    @staticmethod
    def _radius(
        points: dict[Hashable, xr.DataArray] | dict[Hashable, float],
        radius: float | dict[Hashable, float] | None,
        **kwargs: float,
    ) -> dict[Hashable, float]:
        """Helper function. Generate radius dict.

        When radius is dict form, nothing has been done, essentially.

        Args:
            points (dict[Hashable, float]): Selection point
            radius (dict[Hashable, float] | float | None): radius
            kwargs (float): additional radii parameters by keyword with `_r` postfix.

        Returns: dict[Hashable, float]
            radius for selection.
        """
        if isinstance(radius, float):
            radius = {str(d): radius for d in points}
        else:
            collectted_terms = {f"{k}_r" for k in points}.intersection(set(kwargs.keys()))
            if collectted_terms:
                radius = {
                    d: kwargs.get(f"{d}_r", DEFAULT_RADII.get(str(d), UNSPESIFIED)) for d in points
                }
            elif radius is None:
                radius = {d: DEFAULT_RADII.get(str(d), UNSPESIFIED) for d in points}
        assert isinstance(radius, dict)
        return {d: radius.get(str(d), DEFAULT_RADII.get(str(d), UNSPESIFIED)) for d in points}

    def sum_other(
        self,
        dim_or_dims: list[str],
        *,
        keep_attrs: bool = False,
    ) -> XrTypes:
        assert isinstance(dim_or_dims, list)

        return self._obj.sum(
            [d for d in self._obj.dims if d not in dim_or_dims],
            keep_attrs=keep_attrs,
        )

    def mean_other(
        self,
        dim_or_dims: list[str] | str,
        *,
        keep_attrs: bool = False,
    ) -> XrTypes:
        assert isinstance(dim_or_dims, list)

        return self._obj.mean(
            [d for d in self._obj.dims if d not in dim_or_dims],
            keep_attrs=keep_attrs,
        )

    def fat_sel(
        self,
        widths: dict[str, Any] | None = None,
        **kwargs: float,
    ) -> XrTypes:
        """Allows integrating a selection over a small region.

        The produced dataset will be normalized by dividing by the number
        of slices integrated over.

        This can be used to produce temporary datasets that have reduced
        uncorrelated noise.

        Args:
            widths: Override the widths for the slices. Reasonable defaults are used otherwise.
                    Defaults to None.
            kwargs: slice dict. The width can also be specified by like "eV_wdith=0.1".

        Returns:
            The data after selection.
        """
        logger.debug(f"widths: {widths}")
        logger.debug(f"kwargs: {kwargs}")
        if widths is None:
            widths = {}
        assert isinstance(widths, dict)
        default_widths = DEFAULT_RADII

        if self._obj.S.angle_unit == "Degrees":
            default_widths["phi"] = 1.0
            default_widths["beta"] = 1.0
            default_widths["theta"] = 1.0
            default_widths["psi"] = 1.0
        extra_kwargs: dict[str, Incomplete] = {
            k: v for k, v in kwargs.items() if k not in self._obj.dims
        }
        slice_kwargs = {k: v for k, v in kwargs.items() if k in self._obj.dims}
        slice_widths: dict[str, float] = {
            k: widths.get(k, extra_kwargs.get(k + "_width", default_widths.get(k)))
            for k in slice_kwargs
        }
        slices = {
            k: slice(v - slice_widths[k] / 2, v + slice_widths[k] / 2)
            for k, v in slice_kwargs.items()
        }
        sliced = self._obj.sel(slices)
        thickness = np.prod([len(sliced.coords[k]) for k in slice_kwargs])
        normalized = sliced.sum(slices.keys(), keep_attrs=True, min_count=1) / thickness
        for k, v in slices.items():
            normalized.coords[k] = (v.start + v.stop) / 2
        normalized.attrs.update(self._obj.attrs.copy())
        return normalized


class ARPESDataArrayAccessorBase(ARPESAccessorBase):
    _obj: xr.DataArray

    class _SliceAlongPathKwags(TypedDict, total=False):
        axis_name: str
        resolution: float
        n_points: int
        extend_to_edge: bool

    @property
    def is_subtracted(self) -> bool:
        """Infers whether a given data is subtracted.

        Returns (bool):
            Return True if the data is subtracted.
        """
        assert isinstance(self._obj, xr.DataArray)
        if self._obj.attrs.get("subtracted"):
            return True

        threshold_is_5_percent = 0.05
        if (((self._obj < 0) * 1).mean() > threshold_is_5_percent).item():
            self._obj.attrs["subtracted"] = True
            return True
        return False

    def select_around_data(
        self,
        points: dict[Hashable, xr.DataArray] | xr.Dataset,
        radius: dict[Hashable, float] | float | None = None,  # radius={"phi": 0.005}
        *,
        mode: Literal["sum", "mean"] = "sum",
        **kwargs: Incomplete,
    ) -> xr.DataArray:
        """Performs a binned selection around a point or points.

        Can be used to perform a selection along one axis as a function of another, integrating a
        region in the other dimensions.

        Example:
            As an example, suppose we have a dataset with dimensions ('eV', 'kp', 'T',)
            and we also by fitting determined the Fermi momentum as a function of T, kp_F('T'),
            stored in the dataset kFs. Then we could select momentum integrated EDCs in a small
            window around the fermi momentum for each temperature by using

            >>> edcs = full_data.S.select_around_data(points={'kp': kFs}, radius={'kp': 0.04})

            The resulting data will be EDCs for each T, in a region of radius 0.04 inverse angstroms
            around the Fermi momentum.

        Args:
            points: The set of points where the selection should be performed.
            radius: The radius of the selection in each coordinate. If dimensions are omitted, a
                    standard sized selection will be made as a compromise.
            mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"
            kwargs: Can be used to pass radii parameters by keyword with `_r` postfix.

        Returns:
            The binned selection around the desired point or points.
        """
        assert mode in {"sum", "mean"}, "mode parameter should be either sum or mean."
        assert isinstance(points, dict | xr.Dataset)
        radius = radius or {}
        if isinstance(points, xr.Dataset):
            points = {k: points[k].item() for k in points.data_vars}
        assert isinstance(points, dict)
        radius = self._radius(points, radius, **kwargs)
        logger.debug(f"radius: {radius}")

        assert isinstance(radius, dict)
        logger.debug(f"iter(points.values()): {iter(points.values())}")

        along_dims = next(iter(points.values())).dims
        selected_dims = list(points.keys())

        new_dim_order = [d for d in self._obj.dims if d not in along_dims] + list(along_dims)

        data_for = self._obj.transpose(*new_dim_order)
        new_data = data_for.sum(selected_dims, keep_attrs=True)

        stride: dict[Hashable, float] = self._obj.G.stride(generic_dim_names=False)
        for coord in data_for.G.iter_coords(along_dims):
            value = data_for.sel(coord, method="nearest")
            nearest_sel_params: dict[Hashable, xr.DataArray] = {}
            for dim, v in radius.items():
                if v < stride[dim]:
                    nearest_sel_params[dim] = points[dim].sel(coord)
            radius = {dim: v for dim, v in radius.items() if dim not in nearest_sel_params}
            selection_slices = {
                dim: slice(
                    points[dim].sel(coord) - radius[dim],
                    points[dim].sel(coord) + radius[dim],
                )
                for dim in points
                if dim in radius
            }
            selected = value.sel(selection_slices)
            if nearest_sel_params:
                selected = selected.sel(nearest_sel_params, method="nearest")
            for d in nearest_sel_params:
                del selected.coords[d]
            if mode == "sum":
                new_data.loc[coord] = selected.sum(list(radius.keys())).values
            elif mode == "mean":
                new_data.loc[coord] = selected.mean(list(radius.keys())).values
        return new_data

    def select_around(
        self,
        point: dict[Hashable, float],
        radius: dict[Hashable, float] | float,
        *,
        mode: Literal["sum", "mean"] = "sum",
        **kwargs: float,
    ) -> xr.DataArray:
        """Selects and integrates a region around a one dimensional point.

        This method is useful to do a small region integration, especially around
        point on a path of a k-point of interest. See also the companion method
        `select_around_data`.

        If radii are not set, or provided through kwargs as 'eV_r' or 'phi_r' for instance,
        then we will try to use reasonable default values; buyer beware.

        Args:
            point: The point where the selection should be performed.
            radius: The radius of the selection in each coordinate. If dimensions are omitted, a
                    standard sized selection will be made as a compromise.
            safe: If true, infills radii with default values. Defaults to `True`.
            mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"
            **kwargs: Can be used to pass radii parameters by keyword with `_r` postfix.

        Returns:
            The binned selection around the desired point.
        """
        assert mode in {"sum", "mean"}, "mode parameter should be either sum or mean."
        assert isinstance(point, dict | xr.Dataset)
        radius = self._radius(point, radius, **kwargs)
        stride = self._obj.G.stride(generic_dim_names=False)
        nearest_sel_params: dict[Hashable, float] = {}
        for dim, v in radius.items():
            if v < stride[dim]:
                nearest_sel_params[dim] = point[dim]
        radius = {dim: v for dim, v in radius.items() if dim not in nearest_sel_params}
        selection_slices = {
            dim: slice(point[dim] - radius[dim], point[dim] + radius[dim])
            for dim in point
            if dim in radius
        }
        selected = self._obj.sel(selection_slices)
        if nearest_sel_params:
            selected = selected.sel(nearest_sel_params, method="nearest")
        for d in nearest_sel_params:
            del selected.coords[d]
        if mode == "sum":
            return selected.sum(list(radius.keys()))
        return selected.mean(list(radius.keys()))

    def find_spectrum_energy_edges(
        self,
        *,
        indices: bool = False,
    ) -> NDArray[np.float64] | NDArray[np.int_]:
        """Return energy position corresponding to the (1D) spectrum edge.

        Spectrum edge is infection point of the peak.

        Args:
            indices (bool): if True, return the pixel (index) number.

        Returns: NDArray[np.float64]
            Energy position
        """
        assert isinstance(
            self._obj,
            xr.DataArray,
        )  # if self._obj is xr.Dataset, values is  function
        energy_marginal = self._obj.sum([d for d in self._obj.dims if d != "eV"])

        embed_size = 20
        embedded: NDArray[np.float64] = np.ndarray(shape=[embed_size, energy_marginal.sizes["eV"]])
        embedded[:] = energy_marginal.values
        embedded = ndi.gaussian_filter(embedded, embed_size / 3)

        edges = feature.canny(
            embedded,
            sigma=embed_size / 5,
            use_quantiles=True,
            low_threshold=0.1,
        )
        edges = np.where(edges[int(embed_size / 2)] == 1)[0]
        if indices:
            return edges

        delta = self._obj.G.stride(generic_dim_names=False)
        return edges * delta["eV"] + self._obj.coords["eV"].values[0]

    def find_spectrum_angular_edges_full(
        self,
        *,
        indices: bool = False,
        energy_division: float = 0.05,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], xr.DataArray]:
        """[TODO:summary].

        Args:
            indices: [TODO:description]
            energy_division: [TODO:description]

        Returns:
            [TODO:description]

        Todo:
            Test
        """
        # as a first pass, we need to find the bottom of the spectrum, we will use this
        # to select the active region and then to rebin into course steps in energy from 0
        # down to this region
        # we will then find the appropriate edge for each slice, and do a fit to the edge locations
        energy_edge = self.find_spectrum_energy_edges()
        low_edge: np.float64 = np.min(energy_edge) + energy_division
        high_edge: np.float64 = np.max(energy_edge) - energy_division

        if high_edge - low_edge < 3 * energy_division:
            # Doesn't look like the automatic inference of the energy edge was valid
            high_edge = self._obj.coords["eV"].max().item()
            low_edge = self._obj.coords["eV"].min().item()

        angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
        energy_cut = self._obj.sel(eV=slice(low_edge, high_edge)).S.sum_other(["eV", angular_dim])

        n_cuts = int(np.ceil((high_edge - low_edge) / energy_division))
        new_shape = {"eV": n_cuts}
        new_shape[angular_dim] = energy_cut.sizes[angular_dim]
        logger.debug(f"new_shape: {new_shape}")
        rebinned = rebin(energy_cut, shape=new_shape)

        embed_size = 20
        embedded: NDArray[np.float64] = np.empty(
            shape=[embed_size, rebinned.sizes[angular_dim]],
        )
        low_edges = []
        high_edges = []
        for e_cut_index in range(rebinned.sizes["eV"]):
            e_slice = rebinned.isel(eV=e_cut_index)
            embedded[:] = e_slice.values
            embedded = ndi.gaussian_filter(embedded, embed_size / 1.5)  # < = Why 1.5

            edges: NDArray[np.bool_] = feature.canny(
                image=embedded,
                sigma=4,
                use_quantiles=False,
                low_threshold=0.7,
                high_threshold=1.5,
            )
            edges = np.where(edges[int(embed_size / 2)] == 1)[0]
            low_edges.append(np.min(edges))
            high_edges.append(np.max(edges))

        if indices:
            return np.array(low_edges), np.array(high_edges), rebinned.coords["eV"]

        delta = self._obj.G.stride(generic_dim_names=False)

        return (
            np.array(low_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0],
            np.array(high_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0],
            rebinned.coords["eV"],
        )

    def zero_spectrometer_edges(
        self,
        cut_margin: int = 0,
        interp_range: float | None = None,
        low: Sequence[float] | NDArray[np.float64] | None = None,
        high: Sequence[float] | NDArray[np.float64] | None = None,
    ) -> xr.DataArray:
        """[TODO:summary].

        Args:
            cut_margin: [TODO:description]
            interp_range: [TODO:description]
            low: [TODO:description]
            high: [TODO:description]

        Returns:
            [TODO:description]

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.DataArray)
        if low is not None:
            assert high is not None
            assert len(low) == len(high) == TWO_DIMENSION

            low_edges = low
            high_edges = high

        (
            low_edges,
            high_edges,
            rebinned_eV_coord,
        ) = self.find_spectrum_angular_edges_full(indices=True)

        angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
        if not cut_margin:
            if "pixel" in self._obj.dims:
                cut_margin = 50
            else:
                cut_margin = int(0.08 / self._obj.G.stride(generic_dim_names=False)[angular_dim])
        elif isinstance(cut_margin, float):
            assert angular_dim == "phi"
            cut_margin = int(
                cut_margin / self._obj.G.stride(generic_dim_names=False)[angular_dim],
            )

        if interp_range is not None:
            low_edge = xr.DataArray(low_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
            high_edge = xr.DataArray(high_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
            low_edge = low_edge.sel(eV=interp_range)
            high_edge = high_edge.sel(eV=interp_range)
        other_dims = list(self._obj.dims)
        other_dims.remove("eV")
        other_dims.remove(angular_dim)
        copied = self._obj.copy(deep=True).transpose(*(["eV", angular_dim, *other_dims]))

        low_edges += cut_margin
        high_edges -= cut_margin

        for i, energy in enumerate(copied.coords["eV"].values):
            index = np.searchsorted(rebinned_eV_coord, energy)
            other = index + 1
            if other >= len(rebinned_eV_coord):
                other = len(rebinned_eV_coord) - 1
                index = len(rebinned_eV_coord) - 2

            low_index = int(np.interp(energy, rebinned_eV_coord, low_edges))
            high_index = int(np.interp(energy, rebinned_eV_coord, high_edges))
            copied.values[i, 0:low_index] = 0
            copied.values[i, high_index:-1] = 0

        return copied

    def find_spectrum_angular_edges(
        self,
        *,
        angle_name: str = "phi",
        indices: bool = False,
    ) -> NDArray[np.float64] | NDArray[np.int_]:
        """Return angle position corresponding to the (1D) spectrum edge.

        Args:
            angle_name (str): angle name to find the edge
            indices (bool):  if True, return the index not the angle value.

        Returns: NDArray[np.float64] | NDArray[np.int64]
            Angle position
        """
        angular_dim: str = "pixel" if "pixel" in self._obj.dims else angle_name
        assert isinstance(self._obj, xr.DataArray)
        phi_marginal = self._obj.sum(
            [d for d in self._obj.dims if d != angular_dim],
        )

        embed_size = 20
        embedded: NDArray[np.float64] = np.ndarray(
            shape=[embed_size, phi_marginal.sizes[angular_dim]],
        )
        embedded[:] = phi_marginal.values
        embedded = ndi.gaussian_filter(embedded, embed_size / 3)

        # try to avoid dependency conflict with numpy v0.16

        edges = feature.canny(
            image=embedded,
            sigma=embed_size / 5,
            use_quantiles=True,
            low_threshold=0.2,
        )
        edges = np.where(edges[int(embed_size / 2)] == 1)[0]
        if indices:
            return edges

        delta = self._obj.G.stride(generic_dim_names=False)
        return edges * delta[angular_dim] + self._obj.coords[angular_dim].values[0]

    def wide_angle_selector(self, *, include_margin: bool = True) -> slice:
        """[TODO:summary].

        Args:
            include_margin: [TODO:description]

        Returns:
            [TODO:description]

        Todo:
            Test/Consider to remove
        """
        edges = self.find_spectrum_angular_edges()
        low_edge, high_edge = np.min(edges), np.max(edges)

        # go and build in a small margin
        if include_margin:
            if "pixels" in self._obj.dims:
                low_edge += 50
                high_edge -= 50
            else:
                low_edge += 0.05
                high_edge -= 0.05

        return slice(low_edge, high_edge)

    def meso_effective_selector(self) -> slice:
        """[TODO:summary].

        Returns:
            [TODO:description]

        Todo:
            Test/Consider to remove
        """
        energy_edge = self.find_spectrum_energy_edges()
        return slice(np.max(energy_edge) - 0.3, np.max(energy_edge) - 0.1)

    def region_sel(
        self,
        *regions: Literal["copper_prior", "wide_angular", "narrow_angular"]
        | dict[str, DesignatedRegions],
    ) -> XrTypes:
        """[TODO:summary].

        Args:
            regions: [TODO:description]

        Returns:
            [TODO:description]

        Raises:
            NotImplementedError: [TODO:description]

        Todo:
            Test
        """

        def process_region_selector(
            selector: slice | DesignatedRegions,
            dimension_name: str,
        ) -> slice | Callable[..., slice]:
            if isinstance(selector, slice):
                return selector

            options = {
                "eV": (
                    DesignatedRegions.ABOVE_EF,
                    DesignatedRegions.BELOW_EF,
                    DesignatedRegions.EF_NARROW,
                    DesignatedRegions.MESO_EF,
                    DesignatedRegions.MESO_EFFECTIVE_EF,
                    DesignatedRegions.ABOVE_EFFECTIVE_EF,
                    DesignatedRegions.BELOW_EFFECTIVE_EF,
                    DesignatedRegions.EFFECTIVE_EF_NARROW,
                ),
                "phi": (
                    DesignatedRegions.NARROW_ANGLE,
                    DesignatedRegions.WIDE_ANGLE,
                    DesignatedRegions.TRIM_EMPTY,
                ),
            }

            options_for_dim = options.get(dimension_name, list(DesignatedRegions))
            assert selector in options_for_dim

            # now we need to resolve out the region
            resolution_methods = {
                DesignatedRegions.ABOVE_EF: slice(0, None),
                DesignatedRegions.BELOW_EF: slice(None, 0),
                DesignatedRegions.EF_NARROW: slice(-0.1, 0.1),
                DesignatedRegions.MESO_EF: slice(-0.3, -0.1),
                DesignatedRegions.MESO_EFFECTIVE_EF: self.meso_effective_selector,
                # Implement me
                # DesignatedRegions.TRIM_EMPTY: ,
                DesignatedRegions.WIDE_ANGLE: self.wide_angle_selector,
                # DesignatedRegions.NARROW_ANGLE: self.narrow_angle_selector,
            }
            resolution_method = resolution_methods[selector]
            if isinstance(resolution_method, slice):
                return resolution_method
            if callable(resolution_method):
                return resolution_method()

            msg = "Unable to determine resolution method."
            raise NotImplementedError(msg)

        obj = self._obj

        def unpack_dim(dim_name: str) -> str:
            if dim_name == "angular":
                return "pixel" if "pixel" in obj.dims else "phi"

            return dim_name

        for region in regions:
            # remove missing dimensions from selection for permissiveness
            # and to transparent composing of regions
            obj = obj.sel(
                {
                    k: process_region_selector(v, k)
                    for k, v in {
                        unpack_dim(k): v for k, v in normalize_region(region).items()
                    }.items()
                    if k in obj.dims
                },
            )

        return obj


@xr.register_dataarray_accessor("S")
class ARPESDataArrayAccessor(ARPESDataArrayAccessorBase):
    """Spectrum related accessor for `xr.DataArray`."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """Initialize."""
        self._obj: xr.DataArray = xarray_obj
        assert isinstance(self._obj, xr.DataArray)

    def cut_nan_coords(self: Self) -> xr.DataArray:
        """Selects data where coordinates are not `nan`.

        Returns (xr.DataArray):
            The subset of the data where coordinates are not `nan`.

        Todo:
            Test
        """
        slices = {}
        assert isinstance(self._obj, xr.DataArray)
        for cname, cvalue in self._obj.coords.items():
            try:
                end_ind = np.where(np.isnan(cvalue.values))[0][0]
                end_ind = None if end_ind == -1 else end_ind
                slices[cname] = slice(None, end_ind)
            except IndexError:
                pass
        return self._obj.isel(slices)

    def corrected_angle_by(
        self,
        angle_for_correction: Literal[
            "alpha_offset",
            "beta_offset",
            "chi_offset",
            "phi_offset",
            "psi_offset",
            "theta_offset",
            "beta",
            "theta",
        ],
    ) -> xr.DataArray:
        """Return xr.DataArray corrected angle by "angle_for_correction".

        if "angle_for_correction" is like "'angle'_offset", the 'angle' corrected by the
        'angle'_offset value. if "angle_for_correction" is "beta" or "theta", the angle "phi" or
        "psi" is shifted.

        Args:
            angle_for_correction(str): should be one of "alpha_offset", "beta_offset",
                                       "chi_offset", "phi_offset", "psi_offset", "theta_offset",
                                       "beta", "theta"

        Returns:
            xr.DataArray

        Todo:
            Test
        """
        assert angle_for_correction in {
            "alpha_offset",
            "beta_offset",
            "chi_offset",
            "phi_offset",
            "psi_offset",
            "theta_offset",
            "beta",
            "theta",
        }
        assert isinstance(self._obj, xr.DataArray)
        assert angle_for_correction in self._obj.attrs
        arr: xr.DataArray = self._obj.copy(deep=True)
        arr.S.correct_angle_by(angle_for_correction)
        return arr

    def correct_angle_by(
        self,
        angle_for_correction: Literal[
            "alpha_offset",
            "beta_offset",
            "chi_offset",
            "phi_offset",
            "psi_offset",
            "theta_offset",
            "beta",
            "theta",
        ],
    ) -> None:
        """Angle correction in place.

        if "angle_for_correction" is like "'angle'_offset", the 'angle' corrected by the
        'angle'_offset value. if "angle_for_correction" is "beta" or "theta", the angle "phi" or
        "psi" is shifted.

        Args:
            angle_for_correction (str): should be one of "alpha_offset", "beta_offset",
                                        "chi_offset", "phi_offset", "psi_offset", "theta_offset",
                                        "beta", "theta"

        Todo:
            Test
        """
        assert angle_for_correction in {
            "alpha_offset",
            "beta_offset",
            "chi_offset",
            "phi_offset",
            "psi_offset",
            "theta_offset",
            "beta",
            "theta",
        }
        assert angle_for_correction in self._obj.attrs
        if "_offset" in angle_for_correction:
            angle = angle_for_correction.split("_")[0]
            if angle in self._obj.coords:
                self._obj.coords[angle] = (
                    self._obj.coords[angle] - self._obj.attrs[angle_for_correction]
                )
            if angle in self._obj.attrs:
                self._obj.attrs[angle] = (
                    self._obj.attrs[angle] - self._obj.attrs[angle_for_correction]
                )
            self._obj.attrs[angle_for_correction] = 0
            return
        if angle_for_correction == "beta":
            if self._obj.S.is_slit_vertical:
                self._obj.coords["phi"] = (
                    self._obj.coords["phi"] + self._obj.attrs[angle_for_correction]
                )
            else:
                self._obj.coords["psi"] = (
                    self._obj.coords["psi"] + self._obj.attrs[angle_for_correction]
                )
        if angle_for_correction == "theta":
            if self._obj.S.is_slit_vertical:
                self._obj.coords["psi"] = (
                    self._obj.coords["psi"] + self._obj.attrs[angle_for_correction]
                )
            else:
                self._obj.coords["phi"] = (
                    self._obj.coords["phi"] + self._obj.attrs[angle_for_correction]
                )
        self._obj.coords[angle_for_correction] = 0
        self._obj.attrs[angle_for_correction] = 0
        return

    # --- Mehhods about plotting
    # --- TODO : [RA] Consider refactoring/removing
    def plot(
        self: Self,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> None:
        """Utility delegate to `xr.DataArray.plot` which rasterizes`.

        Args:
            rasterized (bool): if True, rasterized (Not vector) drawing
            args: Pass to xr.DataArray.plot
            kwargs: Pass to xr.DataArray.plot
        """
        if len(self._obj.dims) == TWO_DIMENSION:
            kwargs.setdefault("rasterized", True)
        with plt.rc_context(rc={"text.usetex": False}):
            self._obj.plot(*args, **kwargs)

    def show(self, **kwargs: Unpack[ProfileViewParam]) -> AdjointLayout:
        """Show holoviews based plot."""
        return profile_view(self._obj, **kwargs)

    def fs_plot(
        self: Self,
        pattern: str = "{}.png",
        **kwargs: Unpack[LabeledFermiSurfaceParam],
    ) -> Path | tuple[Figure | None, Axes]:
        """Provides a reference plot of the approximate Fermi surface."""
        assert isinstance(self._obj, xr.DataArray)
        out = kwargs.get("out")
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{self.label}_fs")
            kwargs["out"] = out
        return labeled_fermi_surface(self._obj, **kwargs)

    def fermi_edge_reference_plot(
        self: Self,
        pattern: str = "{}.png",
        out: str | Path = "",
        **kwargs: Unpack[MPLPlotKwargs],
    ) -> Path | Axes:
        """Provides a reference plot for a Fermi edge reference.

        Args:
            pattern ([TODO:type]): [TODO:description]
            out (str | Path): Path name for output figure.
            kwargs: pass to plotting.fermi_edge.fermi_edge_reference

        Returns:
            [TODO:description]
        """
        assert isinstance(self._obj, xr.DataArray)
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{self.label}_fermi_edge_reference")
        return fermi_edge_reference(self._obj, out=out, **kwargs)

    def _referenced_scans_for_spatial_plot(
        self: Self,
        *,
        use_id: bool = True,
        pattern: str = "{}.png",
        out: str | Path = "",
    ) -> Path | tuple[Figure, NDArray[np.object_]]:
        """[TODO:summary].

        A Helper function.

        Args:
            use_id (bool): [TODO:description]
            pattern (str): [TODO:description]
            out (str|bool): if str, Path for output figure. if True,
                the file name is automatically set. If False/"", no output is given.
        """
        label = self._obj.attrs["id"] if use_id else self.label
        if isinstance(out, bool) and out is True:
            out = pattern.format(f"{label}_reference_scan_fs")
        elif isinstance(out, bool) and out is False:
            out = ""

        return reference_scan_spatial(self._obj, out=out)

    def _referenced_scans_for_map_plot(
        self: Self,
        pattern: str = "{}.png",
        *,
        use_id: bool = True,
        **kwargs: Unpack[LabeledFermiSurfaceParam],
    ) -> Path | Axes:
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{label}_reference_scan_fs")
            kwargs["out"] = out

        return reference_scan_fermi_surface(self._obj, **kwargs)

    class HvRefScanParam(LabeledFermiSurfaceParam):
        """Parameter for hf_ref_scan."""

        e_cut: float
        bkg_subtraction: float

    def _referenced_scans_for_hv_map_plot(
        self: Self,
        pattern: str = "{}.png",
        *,
        use_id: bool = True,
        **kwargs: Unpack[HvRefScanParam],
    ) -> Path | Axes:
        out = kwargs.get("out")
        label = self._obj.attrs["id"] if use_id else self.label
        if out is not None and isinstance(out, bool):
            out = pattern.format(f"{label}_hv_reference_scan")
            out = f"{label}_hv_reference_scan.png"
            kwargs["out"] = out

        return hv_reference_scan(self._obj, **kwargs)

    def _simple_spectrum_reference_plot(
        self: Self,
        *,
        use_id: bool = True,
        pattern: str = "{}.png",
        out: str | Path = "",
        **kwargs: Unpack[PColorMeshKwargs],
    ) -> Axes | Path:
        assert isinstance(self._obj, xr.DataArray)
        label = self._obj.attrs["id"] if use_id else self.label
        if isinstance(out, bool):
            out = pattern.format(f"{label}_spectrum_reference")

        return fancy_dispersion(self._obj, out=out, **kwargs)

    def reference_plot(
        self,
        **kwargs: Incomplete,
    ) -> Axes | Path | tuple[Figure, NDArray[np.object_]]:
        """Generates a reference plot for this piece of data according to its spectrum type.

        Args:
            kwargs: pass to referenced_scans_for_**

        Raises:
            NotImplementedError: If there is no standard approach for plotting this data.

        Returns:
            The axes which were used for plotting.
        """
        if self.spectrum_type == "map":
            return self._referenced_scans_for_map_plot(**kwargs)
        if self.spectrum_type == "hv_map":
            return self._referenced_scans_for_hv_map_plot(**kwargs)
        if self.spectrum_type == "cut":
            return self._simple_spectrum_reference_plot(**kwargs)
        if self.spectrum_type in {"ucut", "spem"}:
            return self._referenced_scans_for_spatial_plot(**kwargs)
        raise NotImplementedError


NORMALIZED_DIM_NAMES = ["x", "y", "z", "w"]


class GenericAccessorBase:
    _obj: XrTypes

    def round_coordinates(
        self,
        coords: dict[str, list[float] | NDArray[np.float64]],
        *,
        as_indices: bool = False,
    ) -> dict:
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        data = self._obj
        rounded = {
            k: v.item() for k, v in data.sel(coords, method="nearest").coords.items() if k in coords
        }

        if as_indices:
            rounded = {k: data.coords[k].index(v) for k, v in rounded.items()}

        return rounded

    def apply_over(
        self,
        fn: Callable,
        *,
        copy: bool = True,
        **selections: Incomplete,
    ) -> XrTypes:
        """[TODO:summary].

        Args:
            fn: [TODO:description]
            copy: [TODO:description]
            selections: [TODO:description]

        Returns:
            [TODO:description]

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        data = self._obj

        if copy:
            data = data.copy(deep=True)

        try:
            transformed = fn(data.sel(**selections))
        except TypeError:
            transformed = fn(data.sel(**selections).values)

        if isinstance(transformed, xr.DataArray):
            transformed = transformed.values

        data.loc[selections] = transformed
        return data

    def coordinatize(self, as_coordinate_name: str | None = None) -> XrTypes:
        """Copies data into a coordinate's data, with an optional renaming.

        If you think of an array as a function c => f(c) from coordinates to values at
        those coordinates, this function replaces f by the identity to give c => c

        Remarkably, `coordinatize` is a word.

        For the most part, this is only useful when converting coordinate values into
        k-space "forward".

        Args:
            as_coordinate_name: A new coordinate name for the only dimension. Defaults to None.

        Returns:
            An array which consists of the mapping c => c.

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        assert len(self._obj.dims) == 1

        dim = self._obj.dims[0]
        if as_coordinate_name is None:
            as_coordinate_name = str(dim)

        o = self._obj.rename({dim: as_coordinate_name})
        o.coords[as_coordinate_name] = o.values

        return o

    def enumerate_iter_coords(
        self,
        dim_names: Sequence[Hashable] = (),
    ) -> Iterator[tuple[tuple[int, ...], dict[Hashable, float]]]:
        """Return an iterator for pixel and physical coordinates.

        Aargs:
            dir_names (Sequence[Hashable]): Dimension names for iterateion.

        Returns:
            Iteratoring the data like:
            ((0, 0), {'phi': -0.2178031280148764, 'eV': 9.0})
            which shows the relationship between pixel position and physical (like "eV" and "phi").
        """
        assert isinstance(self._obj, xr.DataArray | xr.Dataset)
        if not dim_names:
            dim_names = tuple(self._obj.dims)
        if isinstance(dim_names, str):
            dim_names = [dim_names]
        coords_list = [self._obj.coords[d].values for d in dim_names]
        for indices in itertools.product(*[range(len(c)) for c in coords_list]):
            cut_coords = [cs[index] for cs, index in zip(coords_list, indices, strict=True)]
            yield indices, dict(zip(self._obj.dims, cut_coords, strict=True))

    def iter_coords(
        self,
        dim_names: Sequence[Hashable] = (),
        *,
        reverse: bool = False,
    ) -> Iterator[dict[Hashable, float]]:
        """Iterator for cooridinates along the axis.

        Args:
            dim_names (Sequence[Hashable]): Dimensions for iteration.
            reverse: return the "reversivle" iterator.

        Returns:
            Iterator of the physical position like ("eV" and "phi")
            {'phi': -0.2178031280148764, 'eV': 9.002}
        """
        if not dim_names:
            dim_names = tuple(self._obj.dims)
        if isinstance(dim_names, str):
            dim_names = [dim_names]
        the_iterator: Iterator = itertools.product(*[self._obj.coords[d].values for d in dim_names])
        the_iterator = always_reversible(the_iterator) if reverse else the_iterator
        for ts in the_iterator:
            yield dict(zip(dim_names, ts, strict=True))

    def range(
        self,
        *,
        generic_dim_names: bool = True,
    ) -> dict[Hashable, tuple[float, float]]:
        """Return the maximum/minimum value in each dimension.

        Args:
            generic_dim_names (bool): if True, use Generic dimension name, such as 'x', is used.

        Returns: (dict[str, tuple[float, float]])
            The range of each dimension.
        """
        indexed_coords = [self._obj.coords[d] for d in self._obj.dims]
        indexed_ranges = [(coord.min().item(), coord.max().item()) for coord in indexed_coords]

        dim_names: list[str] | tuple[Hashable, ...] = tuple(self._obj.dims)
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[: len(dim_names)]

        return dict(zip(dim_names, indexed_ranges, strict=True))

    def stride(
        self,
        *args: str | list[str] | tuple[str, ...],
        generic_dim_names: bool = True,
    ) -> dict[Hashable, float] | list[float] | float:
        """Return the stride in each dimension.

        Note that the stride defined in this method is just a difference between first two values.
        In most case, this treatment does not cause a problem.  However, when the data has been
        concatenated, this assumption may not be not valid.

        Args:
            args: The dimension to return.  ["eV", "phi"] or "eV", "phi"
            generic_dim_names (bool): if True, use Generic dimension name, such as 'x', is used.

        Returns:
            The stride of each dimension
        """
        indexed_coords: list[xr.DataArray] = [self._obj.coords[d] for d in self._obj.dims]
        indexed_strides: list[float] = [
            coord.values[1] - coord.values[0] for coord in indexed_coords
        ]

        dim_names: list[str] | tuple[Hashable, ...] = tuple(self._obj.dims)
        if generic_dim_names:
            dim_names = NORMALIZED_DIM_NAMES[: len(dim_names)]

        result: dict[Hashable, float] = dict(zip(dim_names, indexed_strides, strict=True))
        if args:
            if isinstance(args[0], str):
                return (
                    result[args[0]]
                    if len(args) == 1
                    else [result[str(selected_names)] for selected_names in args]
                )
            return [result[selected_names] for selected_names in args[0]]
        return result

    def filter_coord(
        self,
        coordinate_name: str,
        sieve: Callable[[Any, XrTypes], bool],
    ) -> XrTypes:
        """Filters a dataset along a coordinate.

        Sieve should be a function which accepts a coordinate value and the slice
        of the data along that dimension.

        Internally, the predicate function `sieve` is applied to the coordinate and slice to
        generate a mask. The mask is used to select from the data after iteration.

        An improvement here would support filtering over several coordinates.

        Args:
            coordinate_name: The coordinate which should be filtered.
            sieve: A predicate to be applied to the coordinate and data at that coordinate.

        Returns:
            A subset of the data composed of the slices which make the `sieve` predicate `True`.

        Todo:
            Test
        """
        mask = np.array(
            [
                i
                for i, c in enumerate(self._obj.coords[coordinate_name])
                if sieve(c, self._obj.isel({coordinate_name: i}))
            ],
        )
        return self._obj.isel({coordinate_name: mask})


@xr.register_dataset_accessor("G")
class GenericDatasetAccessor(GenericAccessorBase):
    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialization hook for xarray.Dataset.

        This should never need to be called directly.

        Args:
            xarray_obj: The parent object which this is an accessor for
        """
        self._obj = xarray_obj
        assert isinstance(self._obj, xr.Dataset)

    def filter_vars(
        self,
        f: Callable[[Hashable, xr.DataArray], bool],
    ) -> xr.Dataset:
        """[TODO:summary].

        Args:
            f: [TODO:description]

        Returns:
            [TODO:description]

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.Dataset)  # ._obj.data_vars
        return xr.Dataset(
            data_vars={k: v for k, v in self._obj.data_vars.items() if f(k, v)},
            attrs=self._obj.attrs,
        )

    def shift_coords(
        self,
        dims: tuple[str, ...],
        shift: NDArray[np.float64] | float,
    ) -> xr.Dataset:
        """[TODO:summary].

        Args:
            dims: [TODO:description]
            shift: [TODO:description]

        Returns:
            [TODO:description]

        Raises:
            RuntimeError: [TODO:description]

        Todo:
            Test
        """
        if not isinstance(shift, np.ndarray):
            shift = np.ones((len(dims),)) * shift

        def transform(data: NDArray[np.float64]) -> NDArray[np.float64]:
            new_shift: NDArray[np.float64] = shift
            for _ in range(len(dims)):
                new_shift = np.expand_dims(new_shift, axis=0)

            return data + new_shift

        return self.transform_coords(dims, transform)

    def scale_coords(
        self,
        dims: tuple[str, ...],
        scale: float | NDArray[np.float64],
    ) -> xr.Dataset:
        """[TODO:summary].

        Args:
            dims: [TODO:description]
            scale: [TODO:description]

        Returns:
            [TODO:description]

        Todo:
            Test
        """
        if not isinstance(scale, np.ndarray):
            n_dims = len(dims)
            scale = np.identity(n_dims) * scale
        elif len(scale.shape) == 1:
            scale = np.diag(scale)

        return self.transform_coords(dims, scale)

    def transform_coords(
        self,
        dims: Collection[str],
        transform: NDArray[np.float64] | Callable,
    ) -> xr.Dataset:
        """Transforms the given coordinate values according to an arbitrary function.

        The transformation should either be a function from a len(dims) x size of raveled coordinate
        array to len(dims) x size of raveled_coordinate array or a linear transformation as a matrix
        which is multiplied into such an array.

        Params:
            dims: A list or tuple of dimensions that should be transformed
            transform: The transformation to apply, can either be a function, or a matrix

        Returns:
            An identical valued array over new coordinates.

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.Dataset)
        as_ndarray = np.stack([self._obj.data_vars[d].values for d in dims], axis=-1)

        if isinstance(transform, np.ndarray):
            transformed = np.dot(as_ndarray, transform)
        else:
            transformed = transform(as_ndarray)

        copied = self._obj.copy(deep=True)

        for d, arr in zip(dims, np.split(transformed, transformed.shape[-1], axis=-1), strict=True):
            copied.data_vars[d].values = np.squeeze(arr, axis=-1)

        return copied


@xr.register_dataarray_accessor("G")
class GenericDataArrayAccessor(GenericAccessorBase):
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj: xr.DataArray = xarray_obj
        assert isinstance(self._obj, xr.DataArray)

    def argmax_coords(self) -> dict[Hashable, float]:
        """Return dict representing the position for maximum value."""
        data: xr.DataArray = self._obj
        raveled = data.argmax(None)
        assert isinstance(raveled, xr.DataArray)
        idx = raveled.item()
        flat_indices = np.unravel_index(idx, data.values.shape)
        return {d: data.coords[d][flat_indices[i]].item() for i, d in enumerate(data.dims)}

    def ravel(self) -> Mapping[Hashable, xr.DataArray | NDArray[np.float64]]:
        """Converts to a flat representation where the coordinate values are also present.

        Extremely valuable for plotting a dataset with coordinates, X, Y and values Z(X,Y)
        on a scatter plot in 3D.

        By default the data is listed under the key 'data'.

        Returns:
            A dictionary mapping between coordinate names and their coordinate arrays.
            Additionally, there is a key "data" which maps to the `.values` attribute of the array.
        """
        assert isinstance(self._obj, xr.DataArray)

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        raveled_coordinates = dict(
            zip(
                dims,
                [cs.ravel() for cs in np.meshgrid(*coords_as_list)],
                strict=True,
            ),
        )
        assert "data" not in raveled_coordinates
        raveled_coordinates["data"] = self._obj.values.ravel()

        return raveled_coordinates

    def meshgrid(
        self,
        *,
        as_dataset: bool = False,
    ) -> dict[Hashable, NDArray[np.float64]] | xr.Dataset:
        assert isinstance(self._obj, xr.DataArray)  # ._obj.values is used.

        dims = self._obj.dims
        coords_as_list = [self._obj.coords[d].values for d in dims]
        meshed_coordinates = dict(zip(dims, list(np.meshgrid(*coords_as_list)), strict=True))
        assert "data" not in meshed_coordinates
        meshed_coordinates["data"] = self._obj.values

        if as_dataset:
            # this could use a bit of cleaning up
            faked = ["x", "y", "z", "w"]
            return xr.Dataset(
                {
                    k: (faked[: len(v.shape)], v)
                    for k, v in meshed_coordinates.items()
                    if k != "data"
                },
            )

        return meshed_coordinates

    def to_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Converts a (1D) `xr.DataArray` into two plain ``ndarray`` s of their coordinate and data.

        Useful for rapidly converting into a format than can be `plt.scatter` ed
        or similar.

        Example:
            We can use this to quickly scatter a 1D dataset where one axis is the coordinate value.

            >>> plt.scatter(*data.G.as_arrays(), marker='s')  # doctest: +SKIP

        Returns:
            A tuple of the coordinate array (first index) and the data array (second index)

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.DataArray)
        assert len(self._obj.dims) == 1

        return (self._obj.coords[self._obj.dims[0]].values, self._obj.values)

    def clean_outliers(self, clip: float = 0.5) -> xr.DataArray:
        """[TODO:summary].

        Args:
            clip: [TODO:description]

        Returns:
            [TODO:description]

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.DataArray)
        low, high = np.percentile(self._obj.values, [clip, 100 - clip])
        copied = self._obj.copy(deep=True)
        copied.values[copied.values < low] = low
        copied.values[copied.values > high] = high
        return copied

    def as_movie(
        self,
        time_dim: str = "delay",
        pattern: str = "{}.png",
        *,
        out: str | bool = "",
        **kwargs: Unpack[PColorMeshKwargs],
    ) -> Path | animation.FuncAnimation:
        """[TODO:summary].

        Args:
            time_dim: [TODO:description]
            pattern: [TODO:description]
            out: [TODO:description]
            kwargs: [TODO:description]

        Returns:
            [TODO:description]

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.DataArray)

        if isinstance(out, bool) and out is True:
            out = pattern.format(f"{self._obj.S.label}_animation")
        assert isinstance(out, str)
        return plot_movie(self._obj, time_dim, out=out, **kwargs)

    def map_axes(
        self,
        axes: list[str] | str,
        fn: Callable[[XrTypes, dict[str, float]], DataType],
        dtype: DTypeLike = None,
    ) -> xr.DataArray:
        """[TODO:summary].

        Args:
            axes ([TODO:type]): [TODO:description]
            fn: [TODO:description]
            dtype: [TODO:description]

        Raises:
            TypeError: [TODO:description]

        Todo:
            Test
        """
        obj = self._obj.copy(deep=True)

        if dtype is not None:
            obj.values = np.ndarray(shape=obj.values.shape, dtype=dtype)

        type_assigned = False
        for coord in self.iter_coords(axes):
            value = self._obj.sel(coord, method="nearest")
            new_value = fn(value, coord)

            if dtype is None:
                if not type_assigned:
                    obj.values = np.ndarray(shape=obj.values.shape, dtype=new_value.data.dtype)
                    type_assigned = True

                obj.loc[coord] = new_value.values
            else:
                obj.loc[coord] = new_value

        return obj

    def transform(
        self,
        axes: str | list[str],
        transform_fn: Callable,
        dtype: DTypeLike = None,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> xr.DataArray:
        """Applies a vectorized operation across a subset of array axes.

        Transform has similar semantics to matrix multiplication, the dimensions of the
        output can grow or shrink depending on whether the transformation is size preserving,
        grows the data, shinks the data, or leaves in place.

        Examples:
            As an example, let us suppose we have a function which takes the mean and
            variance of the data:

                [dimension], coordinate_value -> [{'mean', 'variance'}]

            And a dataset with dimensions [X, Y]. Then calling transform
            maps to a dataset with the same dimension X but where Y has been replaced by
            the length 2 label {'mean', 'variance'}. The full dimensions in this case are
            ['X', {'mean', 'variance'}].

            >>> data.G.transform('X', f).dims  # doctest: +SKIP
            ["X", "mean", "variance"]

        Please note that the transformed axes always remain in the data because they
        are iterated over and cannot therefore be modified.

        The transform function `transform_fn` must accept the coordinate of the
        marginal at the currently iterated point.

        Args:
            axes: Dimension/axis or set of dimensions to iterate over
            transform_fn: Transformation function that takes a DataArray into a new DataArray
            dtype: An optional type hint for the transformed data. Defaults to None.
            args: args to pass into transform_fn
            kwargs: kwargs to pass into transform_fn

        Raises:
            TypeError: When the underlying object is an `xr.Dataset` instead of an `xr.DataArray`.
                This is due to a constraint related to type inference with a single passed dtype.


        Returns:
            The data consisting of applying `transform_fn` across the specified axes.

        Todo:
            Test
        """
        dest = None
        for coord in self._obj.G.iter_coords(axes):
            value = self._obj.sel(coord, method="nearest")
            new_value = transform_fn(value, coord, *args, **kwargs)

            if dest is None:
                new_value = transform_fn(value, coord, *args, **kwargs)

                original_dims = [d for d in self._obj.dims if d not in value.dims]
                original_shape = [self._obj.shape[self._obj.dims.index(d)] for d in original_dims]
                original_coords = {k: v for k, v in self._obj.coords.items() if k not in value.dims}

                full_shape = original_shape + list(new_value.shape)

                new_coords = original_coords
                new_coords.update(
                    {k: v for k, v in new_value.coords.items() if k not in original_coords},
                )
                new_dims = original_dims + list(new_value.dims)
                dest = xr.DataArray(
                    data=np.zeros(full_shape, dtype=dtype or new_value.data.dtype),
                    coords=new_coords,
                    dims=new_dims,
                )

            dest.loc[coord] = new_value
        assert isinstance(dest, xr.DataArray)
        return dest

    def map(
        self,
        fn: Callable[[NDArray[np.float64], Any], NDArray[np.float64]],
        **kwargs: Incomplete,
    ) -> xr.DataArray:
        """[TODO:summary].

        Args:
            fn (Callable): Function applying to xarray.values
            kwargs: [TODO:description]

        Returns:
            [TODO:description]
        """
        return apply_dataarray(self._obj, np.vectorize(fn, **kwargs))

    def shift_by(
        self,
        other: xr.DataArray | NDArray[np.float64],
        shift_axis: str = "",
        by_axis: str = "",
        *,
        zero_nans: bool = True,
        shift_coords: bool = False,
    ) -> xr.DataArray:
        """Data shift along the axis.

        For now we only support shifting by a one dimensional array

        Args:
            other (xr.DataArray | NDArray): [TODO:description]
                 we only support shifting by a one dimensional array
            shift_axis (str): [TODO:description]
            by_axis (str): The dimension name of `other`.  When `other` is xr.DataArray, this value
                 is ignored.
            zero_nans (bool): if True, fill 0 for np.nan
            shift_coords (bool): [TODO:description]

        Returns (xr.DataArray):
            Shifted xr.DataArray

        Todo:
            Test
        """
        assert shift_axis, "shift_by must take shift_axis argument."
        data = self._obj.copy(deep=True)
        mean_shift: np.float64 | float = 0.0
        if isinstance(other, xr.DataArray):
            assert other.ndim == 1
            by_axis = str(other.dims[0])
            assert len(other.coords[by_axis]) == len(data.coords[by_axis])
            if shift_coords:
                mean_shift = np.mean(other.values)
                other -= mean_shift
            shift_amount = -other.values / data.G.stride(generic_dim_names=False)[shift_axis]
        else:
            assert isinstance(other, np.ndarray)
            assert other.ndim == 1
            if not by_axis:
                if data.ndim == TWO_DIMENSION:
                    by_axis = str(set(data.dims).difference(shift_axis).pop())
                else:
                    msg = "When np.ndarray is used for shift_by, and the data is not 2D,"
                    msg += '"by_axis" is required.'
                    raise TypeError(msg)
            assert other.shape[0] == len(data.coords[by_axis])
            if shift_coords:
                mean_shift = np.mean(other)
                other -= mean_shift
            shift_amount = -other / data.G.stride(generic_dim_names=False)[shift_axis]

        shifted_data: NDArray[np.float64] = arpes.utilities.math.shift_by(
            arr=data.values,
            value=shift_amount,
            axis=data.dims.index(shift_axis),
            by_axis=data.dims.index(by_axis),
            order=1,
        )
        if zero_nans:
            shifted_data[np.isnan(shifted_data)] = 0
        built_data = data.G.with_values(shifted_data, keep_attrs=True)
        if shift_coords:
            built_data = built_data.assign_coords(
                {shift_axis: data.coords[shift_axis] + mean_shift},
            )
        return built_data

    def drop_nan(self) -> xr.DataArray:
        """[TODO:summary]..

        Returns:
            [TODO:description]

        Todo:
            Test
        """
        assert len(self._obj.dims) == 1

        mask = np.logical_not(np.isnan(self._obj.values))
        return self._obj.isel({self._obj.dims[0]: mask})

    def with_values(
        self,
        new_values: NDArray[np.float64],
        *,
        keep_attrs: bool = True,
    ) -> xr.DataArray:
        """Copy with new array values.

        Easy way of creating a DataArray that has the same shape as the calling object but data
        populated from the array `new_values`.

        Notes: This method is applicable only for xr.DataArray.  (Not xr.Dataset)

        Args:
            new_values: The new values which should be used for the data.
            keep_attrs (bool): If True, attributes are also copied.

        Returns:
            A copy of the data with new values but identical dimensions, coordinates, and attrs.

        ToDo: Test
        """
        assert isinstance(self._obj, xr.DataArray)
        if keep_attrs:
            return xr.DataArray(
                data=new_values.reshape(self._obj.values.shape),
                coords=self._obj.coords,
                dims=self._obj.dims,
                attrs=self._obj.attrs,
            )
        return xr.DataArray(
            data=new_values.reshape(self._obj.values.shape),
            coords=self._obj.coords,
            dims=self._obj.dims,
        )


@xr.register_dataset_accessor("F")
class ARPESDatasetFitToolAccessor:
    _obj: xr.Dataset

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    def show(self, **kwargs: Unpack[ProfileViewParam]) -> AdjointLayout:
        """[TODO:summary].

        Todo:
            Need Revision (It does not work, currently)/Consider removing.
        """
        return fit_inspection(self._obj, **kwargs)

    @property
    def fit_dimensions(self) -> list[str]:
        """Returns the dimensions which were broadcasted across, as opposed to fit across.

        This is a sibling property to `broadcast_dimensions`.

        Returns:
            The list of the dimensions which were **not** used in any individual fit.
            For example, a broadcast of MDCs across energy on a dataset with dimensions
            `["eV", "kp"]` would produce `["eV"]`.
        """
        assert isinstance(self._obj, xr.Dataset)
        return list(set(self._obj.data.dims).difference(self._obj.results.dims))


@xr.register_dataarray_accessor("F")
class ARPESFitToolsAccessor:
    """Utilities related to examining curve fits."""

    _obj: xr.DataArray

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """Initialization hook for xarray.

        This should never need to be called directly.

        Args:
            xarray_obj: The parent object which this is an accessor for
        """
        self._obj = xarray_obj

    class _PlotParamKwargs(MPLPlotKwargs, total=False):
        ax: Axes | None
        shift: float
        x_shift: float
        two_sigma: bool
        figsize: tuple[float, float]

    def plot_param(self, param_name: str, **kwargs: Unpack[_PlotParamKwargs]) -> None:
        """Creates a scatter plot of a parameter from a multidimensional curve fit.

        Args:
            param_name: The name of the parameter which should be plotted
            kwargs: Passed to plotting routines to provide user control
                figsize=, color=, markersize=,
        """
        plot_parameter(self._obj, param_name, **kwargs)

    def param_as_dataset(self, param_name: str) -> xr.Dataset:
        """Maps from `lmfit.ModelResult` to a Dict parameter summary.

        Args:
            param_name: The parameter which should be summarized.

        Returns:
            A dataset consisting of two arrays: "value" and "error"
            which are the fit value and standard error on the parameter
            requested.
        """
        return xr.Dataset(
            {
                "value": self.p(param_name),
                "error": self.s(param_name),
            },
        )

    def best_fits(self) -> xr.DataArray:
        """Orders the fits into a raveled array by the MSE error.

        Todo:
            Test
        """
        return self.order_stacked_fits(ascending=True)

    def worst_fits(self) -> xr.DataArray:
        """Orders the fits into a raveled array by the MSE error.

        Todo:
            Test
        """
        return self.order_stacked_fits(ascending=False)

    def mean_square_error(self) -> xr.DataArray:
        """Calculates the mean square error of the fit across fit axes.

        Producing a scalar metric of the error for all model result instances in
        the collection.
        """
        assert isinstance(self._obj, xr.DataArray)

        def safe_error(model_result_instance: lmfit.model.ModelResult | None) -> float:
            if model_result_instance is None:
                return np.nan
            assert isinstance(model_result_instance.residual, np.ndarray)
            return (model_result_instance.residual**2).mean()

        return self._obj.G.map(safe_error)

    def order_stacked_fits(self, *, ascending: bool = False) -> xr.DataArray:
        """Produces an ordered collection of `lmfit.ModelResult` instances.

        For multidimensional broadcasts, the broadcasted dimensions will be
        stacked for ordering to produce a 1D array of the results.

        Args:
            ascending: Whether the results should be ordered according to ascending
              mean squared error (best fits first) or descending error (worst fits first).

        Returns:
            An xr.DataArray instance with stacked axes whose values are the ordered models.

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.DataArray)
        stacked = self._obj.stack(dim={"by_error": self._obj.dims})

        error = stacked.F.mean_square_error()

        if not ascending:
            error = -error

        indices = np.argsort(error.values)
        return stacked[indices]

    def p(self, param_name: str) -> xr.DataArray:
        """Collects the value of a parameter from curve fitting.

        Across an array of fits, walks parameters to collect the value
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the value found by the fitting routine.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        assert isinstance(self._obj, xr.DataArray)
        return self._obj.G.map(param_getter(param_name), otypes=[float])

    def s(self, param_name: str) -> xr.DataArray:
        """Collects the standard deviation of a parameter from fitting.

        Across an array of fits, walks parameters to collect the standard error
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the floating point value for the fits.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.
        """
        assert isinstance(self._obj, xr.DataArray)
        return self._obj.G.map(param_stderr_getter(param_name), otypes=[float])

    @property
    def bands(self) -> dict[str, MultifitBand]:
        """Collects bands after a multiband fit.

        Returns:
            The collected bands.
        """
        warnings.warn(
            "This method will be deprecated.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        band_names = self.band_names

        return {label: MultifitBand(label=label, data=self._obj) for label in band_names}

    @property
    def band_names(self) -> set[str]:
        """Collects the names of the bands from a multiband fit.

        Heuristically, a band is defined as a dispersive peak so we look for
        prefixes corresponding to parameter names which contain `"center"`.

        Returns:
            The collected prefix names for the bands.

            For instance, if the param name `"a_center"`, the return value
            would contain `"a_"`.
        """
        collected_band_names: set[str] = set()
        assert isinstance(self._obj, xr.DataArray)
        for item in self._obj.values.ravel():
            if item is None:
                continue
            band_names = [k[:-6] for k in item.params if "center" in k]
            collected_band_names = collected_band_names.union(set(band_names))
        return collected_band_names

    @property
    def parameter_names(self) -> set[str]:
        """Collects the parameter names for a multidimensional fit.

        Assumes that the model used is the same for all ``lmfit.ModelResult`` s
        so that we can merely extract the parameter names from a single non-null
        result.

        Returns:
            A set of all the parameter names used in a curve fit.

        Todo:
            Test
        """
        collected_parameter_names: set[str] = set()
        assert isinstance(self._obj, xr.DataArray)
        for item in self._obj.values.ravel():
            if item is None:
                continue

            param_names = list(item.params.keys())
            collected_parameter_names = collected_parameter_names.union(set(param_names))

        return collected_parameter_names


@xr.register_dataset_accessor("S")
class ARPESDatasetAccessor(ARPESAccessorBase):
    """Spectrum related accessor for `xr.Dataset`."""

    def __getattr__(self, item: str) -> dict:
        """Forward attribute access to the spectrum, if necessary.

        Args:
            item: Attribute name

        Returns:
            The attribute after lookup on the default spectrum
        """
        return getattr(self._obj.S.spectrum.S, item)

    @property
    def is_spatial(self) -> bool:
        """Predicate indicating whether the dataset is a spatial scanning dataset.

        Returns:
            True if the dataset has dimensions indicating it is a spatial scan.
            False otherwise
        """
        assert isinstance(self.spectrum, xr.DataArray | xr.Dataset)

        return self.spectrum.S.is_spatial

    @property
    def spectrum(self) -> xr.DataArray:
        """Isolates a single spectrum from a dataset.

        This is a convenience method which is typically used in startup for
        tools and analysis routines which need to operate on a single
        piece of data.
        Historically, the handling of Dataset and Dataarray was a mess in previous pyarpes.
        Most of the current pyarpes methods/function are sufficient to treat DataArray as the main
        object. (The few exceptions are broadcast_model, whose return value is a Dataset, which is
        reasonable.) For backward compatibility, the return of load_data is still a Dataset,
        so in many cases, using this property for a DataArray will provide a more robust analysing
        environment in many cases.

        In practice, we filter data variables by whether they contain "spectrum"
        in the name before selecting the one with the largest pixel volume.
        This is a heuristic which tries to guarantee we select ARPES data
        above XPS data, if they were collected together.

        Returns:
            A spectrum found in the dataset, if one can be isolated.

            In the case that several candidates are found, a single spectrum
            is selected among the candidates.

            Attributes from the parent dataset are assigned onto the selected
            array as a convenience.

        ToDo: Need test
        """
        if "spectrum" in self._obj.data_vars:
            return self._obj.spectrum
        if "raw" in self._obj.data_vars:
            return self._obj.raw
        if "__xarray_dataarray_variable__" in self._obj.data_vars:
            return self._obj.__xarray_dataarray_variable__
        candidates = self.spectra
        if candidates:
            spectrum = candidates[0]
            best_volume = np.prod(spectrum.shape)
            for c in candidates[1:]:
                volume = np.prod(c.shape)
                if volume > best_volume:
                    spectrum = c
                    best_volume = volume
        else:
            msg = "No spectrum found"
            raise RuntimeError(msg)
        return spectrum

    @property
    def spectra(self) -> list[xr.DataArray]:
        """Collects the variables which are likely spectra.

        Returns:
            The subset of the data_vars which have dimensions indicating
            that they are spectra.
        """
        return [dv for dv in self._obj.data_vars.values() if "eV" in dv.dims]

    @property
    def spectrum_type(self) -> Literal["cut", "map", "hv_map", "ucut", "spem", "xps"]:
        """Gives a heuristic estimate of what kind of data is contained by the spectrum.

        Returns:
            The kind of data, coarsely
        """
        return self.spectrum.S.spectrum_type

    @property
    def degrees_of_freedom(self) -> set[Hashable]:
        """The collection of all degrees of freedom.

        Equivalently, dimensions on a piece of data.

        Returns:
            All degrees of freedom as a set.
        """
        return set(self.spectrum.dims)

    @property
    def spectrum_degrees_of_freedom(self) -> set[Hashable]:
        """Collects the spectrometer degrees of freedom.

        Spectrometer degrees of freedom are any which would be collected by an ARToF
        and their momentum equivalents.

        Returns:
            The collection of spectrum degrees of freedom.
        """
        return self.degrees_of_freedom.intersection({"eV", "phi", "pixel", "kx", "kp", "ky"})

    @property
    def scan_degrees_of_freedom(self) -> set[Hashable]:
        """Collects the scan degrees of freedom.

        Scan degrees of freedom are all of the degrees of freedom which are not recorded
        by the spectrometer but are "scanned over". This includes spatial axes,
        temperature, etc.

        Returns:
            The collection of scan degrees of freedom represented in the array.
        """
        return self.degrees_of_freedom.difference(self.spectrum_degrees_of_freedom)

    def reference_plot(self: Self, **kwargs: IncompleteMPL) -> None:
        """Creates reference plots for a dataset.

        A bit of a misnomer because this actually makes many plots. For full datasets,
        the relevant components are:

        #. Temperature as function of scan DOF
        #. Photocurrent as a function of scan DOF
        #. Photocurrent normalized + unnormalized figures, in particular

            #. The reference plots for the photocurrent normalized spectrum
            #. The normalized total cycle intensity over scan DoF, i.e. cycle vs scan DOF integrated
                over E, phi

            #. For delay scans

                #. Fermi location as a function of scan DoF, integrated over phi
                #. Subtraction scans

        #. For spatial scans

            #. energy/angle integrated spatial maps with subsequent measurements indicated
            #. energy/angle integrated FS spatial maps with subsequent measurements indicated

        Args:
            kwargs: Passed to plotting routines to provide user control
        """
        self._obj.sum(self.scan_degrees_of_freedom)
        kwargs.get("out")
        # <== CHECK ME  the above two lines were:

        # make figures for temperature, photocurrent, delay
        make_figures_for = ["T", "IG_nA", "current", "photocurrent"]
        name_normalization = {
            "T": "T",
            "IG_nA": "photocurrent",
            "current": "photocurrent",
        }

        for figure_item in make_figures_for:
            if figure_item not in self._obj.data_vars:
                continue
            name = name_normalization.get(figure_item, figure_item)
            data_var: xr.DataArray = self._obj[figure_item]
            out = f"{self.label}_{name}_spec_integrated_reference.png"
            scan_var_reference_plot(data_var, title=f"Reference {name}", out=out)

        # may also want to make reference figures summing over cycle, or summing over beta

        # make photocurrent normalized figures
        normalized = self._obj / self._obj.IG_nA
        normalized.S.make_spectrum_reference_plots(prefix="norm_PC_", out=True)

        self.make_spectrum_reference_plots(out=True)

    def make_spectrum_reference_plots(
        self,
        prefix: str = "",
        **kwargs: Incomplete,
    ) -> None:
        """Creates photocurrent normalized + unnormalized figures.

        Creates:

        #. The reference plots for the photocurrent normalized spectrum
        #. The normalized total cycle intensity over scan DoF,
           i.e. cycle vs scan DOF integrated over E, phi
        #. For delay scans

            #. Fermi location as a function of scan DoF, integrated over phi
            #. Subtraction scans

        Args:
            prefix: A prefix inserted into filenames to make them unique.
            kwargs: Passed to plotting routines to provide user control over plotting
                    behavior
        """
        self.spectrum.S.reference_plot(pattern=prefix + "{}.png", **kwargs)

        if self.is_spatial:
            pass
            # <== CHECK ME: original is  referenced = self.referenced_scans
        if "cycle" in self._obj.coords:
            integrated_over_scan = self._obj.sum(self.spectrum_degrees_of_freedom)
            integrated_over_scan.S.spectrum.S.reference_plot(
                pattern=prefix + "sum_spec_DoF_{}.png",
                **kwargs,
            )

        if "delay" in self._obj.coords:
            dims = self.spectrum_degrees_of_freedom
            dims.remove("eV")
            angle_integrated = self._obj.sum(dims)

            # subtraction scan
            self.spectrum.S.subtraction_reference_plots(pattern=prefix + "{}.png", **kwargs)
            angle_integrated.S.fermi_edge_reference_plots(pattern=prefix + "{}.png", **kwargs)

    def switch_energy_notation(self, nonlinear_order: int = 1) -> None:
        """Switch the energy notation between binding and kinetic.

        Args:
            nonlinear_order (int): order of the nonliniarity, default to 1
        """
        super().switch_energy_notation(nonlinear_order=nonlinear_order)
        for data in self._obj.data_vars.values():
            if data.S.energy_notation == "Binding":
                data.attrs["energy_notation"] = "Kinetic"
            else:
                data.attrs["energy_notation"] = "Binding"

    def radian_to_degree(self) -> None:
        """Swap angle unit in from Radians to Degrees."""
        super().radian_to_degree()
        self.angle_unit = "Degrees"
        for data in self._obj.data_vars.values():
            data.S.radian_to_degree()
            data.S.angle_unit = "Radians"

    def degree_to_radian(self) -> None:
        """Swap angle unit in from Degrees and Radians."""
        super().degree_to_radian()
        self.angle_unit = "Radians"
        for data in self._obj.data_vars.values():
            data.S.degree_to_radian()
            data.S.angle_unit = "Degrees"

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialization hook for xarray.

        This should never need to be called directly.

        Args:
            xarray_obj: The parent object which this is an accessor for
        """
        self._obj: xr.Dataset
        super().__init__(xarray_obj)
