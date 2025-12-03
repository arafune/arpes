"""Classes for ARPES property."""

from __future__ import annotations

import contextlib
from logging import DEBUG, INFO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypeGuard,
    get_args,
)

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates

from arpes._typing.base import (
    ANGLE,
    HIGH_SYMMETRY_POINTS,
    SpectrumType,
)
from arpes.debug import setup_logger
from arpes.plotting.utils import fancy_labels, remove_colorbars
from arpes.utilities.xarray import unwrap_xarray_item
from arpes.xarray_extensions._helper import unwrap_provenance

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Iterator,
        Mapping,
    )

    from numpy.typing import NDArray

    from arpes._typing.attrs_property import (
        AnalyzerInfo,
        BeamLineSettings,
        DAQInfo,
        ExperimentInfo,
        LightSourceInfo,
        SampleInfo,
        ScanInfo,
    )
    from arpes._typing.base import XrTypes
    from arpes.provenance import Provenance

EnergyNotation: TypeAlias = Literal["Binding", "Final"]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class ARPESAngleProperty:
    """Class for Angle related property.

    Attributes:
        _obj (XrTypes): ARPES data

    Note:
        This class should not be called directly.
    """

    _obj: XrTypes

    @property
    def angle_unit(self) -> Literal["Degrees", "Radians"]:
        """Return Angle unit ("Degrees" or "Radians")."""
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

    Attributes:
        _obj (XrTypes): ARPES data

    Note:
        This class should not be called directly.
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
        """The energy notation ("Binding" energy or "Final" state energy)."""
        notation = self._obj.attrs.get("energy_notation", "Binding").lower()
        final_notations = {"kinetic", "kinetic energy", "final", "final stat energy"}
        if notation in final_notations:
            self._obj.attrs["energy_notation"] = "Final"
            return "Final"

        self._obj.attrs["energy_notation"] = "Binding"
        return "Binding"

    def switch_energy_notation(self, nonlinear_order: int = 1) -> None:
        """Switch the energy notation between binding and kinetic.

        Args:
            nonlinear_order (int): order of the nonliniarity, default to 1
        """
        if self._obj.coords["hv"].ndim != 0:
            msg = "Not implemented yet."
            raise RuntimeError(msg)

        energy_notation = self.energy_notation
        shift = nonlinear_order * self._obj.coords["hv"]

        if energy_notation == "Binding":
            self._obj.coords["eV"] = self._obj.coords["eV"] + shift
            self._obj.attrs["energy_notation"] = "Final"
        elif energy_notation == "Final":
            self._obj.coords["eV"] = self._obj.coords["eV"] - shift
            self._obj.attrs["energy_notation"] = "Binding"


class ARPESInfoProperty(ARPESPhysicalProperty):
    """Class for Information Property.

    Attributes:
        _obj (XrTypes): ARPES data

    Note:
        This class should not be called directly.
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
        """Return "description" in attrs or scan_name."""
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
    def spectrum_type(self) -> SpectrumType:
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
        dims: tuple[str, ...] = tuple(sorted(str(dim) for dim in self._obj.dims))
        if dims in dim_types:
            dim_type = dim_types.get(dims)
        else:
            msg = "Cannot determine spectrum type"
            raise TypeError(msg)

        def _dim_type_check(
            dim_type: str | None,
        ) -> TypeGuard[SpectrumType]:
            return dim_type in get_args(SpectrumType)

        if _dim_type_check(dim_type):
            return dim_type
        msg = "Dimension type may be incorrect"
        raise TypeError(msg)


class ARPESOffsetProperty(ARPESAngleProperty):
    """Class for offset value property.

    Attributes:
        _obj (XrTypes): ARPES data

    Note:
        This class should not be called directly.
    """

    _obj: XrTypes

    def symmetry_points(
        self,
    ) -> dict[HIGH_SYMMETRY_POINTS, dict[str, float]]:
        """Return the dict object about symmetry point such as G-point in the ARPES data.

        The original version was something complicated, but the coding seemed to be in
        process and the purpose was unclear, so it was streamlined considerably.


        Returns (dict[HIGH_SYMMETRY_POINTS, dict[str, float]]):
            Dict object representing the symmetry points in the ARPES data.

        Raises:
            RuntimeError: When the label of high symmetry_points in arr.attrs[symmetry_points] is
                not in HIGH_SYMMETRY_POINTS declared in _typing.py

        Examples:
            symmetry_points = {"G": {"phi": 0.405}}
        """
        symmetry_points: dict[str, dict[str, float]] = {}
        our_symmetry_points = self._obj.attrs.get("symmetry_points", {})

        symmetry_points.update(our_symmetry_points)

        def is_key_high_sym_points(
            symmetry_points: dict[str, dict[str, float]],
        ) -> TypeGuard[dict[HIGH_SYMMETRY_POINTS, dict[str, float]]]:
            return all(key in get_args(HIGH_SYMMETRY_POINTS) for key in symmetry_points)

        if is_key_high_sym_points(symmetry_points):
            return symmetry_points
        msg = "Check the label of High symmetry points.\n"
        msg += f"The allowable labels are: f{get_args(HIGH_SYMMETRY_POINTS)}\n"
        msg += "If you really need the new label, "
        msg += "modify HIGH_SYMMETRY_POINTS in _typing.py (and pull-request)."
        raise RuntimeError(msg)

    @property
    def logical_offsets(self) -> dict[str, float | xr.DataArray]:  # pragma: no cover
        """The logical offsets of the sample position.

        Returns:
            dict object of long_[x, y, z] + physical_long_[x, y, z]

        Todo:
            Consering if this is really suitable way?
                * While this variable used just in MAESTRO.py which I haven't used, to keep
                  consistensy with other plugins the following change seems to be reasonable.

                    * coords["long_x"] should be coosrds["x"] ?
                    * coords["physical_long_x"] seems to be just x_offset.

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
        """Applies and records angular offsets to the xarray object's attributes.

        This method iterates through a dictionary of angle types and their
        corresponding offset values, storing each offset in the `_obj.attrs`
        dictionary. The attribute key is formatted as "{angle_type}_offset".

        These offsets are typically used to correct or define the zero-point
        for various angular dimensions (e.g., k-parallel, polar angle theta).

        Parameters:
            offsets (dict[ANGLE, float]): A dictionary where keys are `ANGLE`
                enum members (or similar type representing angle dimensions,
                e.g., "k_parallel", "theta") and values are the float offsets
                to be applied.

        Returns:
            None: This method modifies the `_obj.attrs` in-place and does not
                return any value.

        Raises:
            AssertionError: If the internal `_obj` is not an instance of
                `xarray.Dataset` or `xarray.DataArray`.

        Examples:
            Assuming `ds` is an `xr.Dataset` and `ANGLE.K_PARALLEL` is defined:

            >>> ds_accessor = YourAccessorClass(ds)
            >>> ds_accessor.apply_offsets({ANGLE.K_PARALLEL: 0.05, ANGLE.THETA: -0.1})
            >>> ds.attrs['k_parallel_offset']
            0.05
            >>> ds.attrs['theta_offset']
            -0.1
        """
        assert isinstance(self._obj, xr.Dataset | xr.DataArray)
        for k, v in offsets.items():
            self._obj.attrs[f"{k}_offset"] = v

    @property
    def iter_own_symmetry_points(self) -> Iterator[tuple[HIGH_SYMMETRY_POINTS, dict[str, float]]]:
        """An iterator property that yields high-symmetry points and their coordinates.

        This property provides a convenient way to iterate over the high-symmetry
        points associated with the current dataset's Brillouin zone or band structure,
        along with their corresponding coordinate dictionaries. It relies on
        the `symmetry_points()` method (which is assumed to be defined elsewhere
        within this class or a related one) to retrieve the mapping of symmetry
        point names to their coordinates.

        Yields:
            Iterator[tuple[HIGH_SYMMETRY_POINTS, dict[str, float]]]: An iterator
            where each item is a tuple containing:
            - A `HIGH_SYMMETRY_POINTS` enum member (or equivalent identifier)
              representing the name of the symmetry point.
            - A dictionary mapping dimension names (strings) to their float
              coordinate values at that symmetry point.

        Examples:
            Assuming `ds_accessor` has symmetry points defined:

            >>> # Assume ds_accessor.symmetry_points() returns:
            >>> # {HIGH_SYMMETRY_POINTS.GAMMA: {'kx': 0.0, 'ky': 0.0},
            >>> #  HIGH_SYMMETRY_POINTS.X_POINT: {'kx': 1.0, 'ky': 0.0}}
            >>> for point_name, coords in ds_accessor.iter_own_symmetry_points:
            ...     print(f"Symmetry Point: {point_name}, Coordinates: {coords}")

            Symmetry Point: Gamma, Coordinates: {'kx': 0.0, 'ky': 0.0}

            Symmetry Point: X_Point, Coordinates: {'kx': 1.0, 'ky': 0.0}
        """
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
    """Class for Provenance related property."""

    _obj: XrTypes

    def short_history(self, key: str = "by") -> list:
        """Return the short version of history.

        Args:
            key (str): key str in recorded dict of self.history.  (default: "by")
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
        """Retrieves the complete processing history (provenance) of the xarray object.

        This property extracts nested provenance records stored in the
        `_obj.attrs["provenance"]` attribute. It unwraps these records
        from the most recent operation back to the original data, forming
        a chronological list of processing steps.

        The provenance information is expected to be stored in a nested dictionary
        structure where each step has a `parents_provenance` key pointing to
        the previous step(s). This method flattens that hierarchical structure
        into a linear list.

        Returns:
            list[Provenance]: A list of dictionaries, where each dictionary
            represents a processing step (Provenance record) in the history of
            the dataset, ordered from the most recent to the oldest. An empty
            list is returned if no provenance is recorded or if it's invalid.

        Warns:
            UserWarning:
                - If the `provenance` attribute is found to be a string type,
                  indicating an older or malformed provenance record.
                - If multiple parents are encountered in a `parents_provenance` list,
                  as only the first parent will be considered and others ignored.

        Examples:
            Assuming `ds` is an `xr.Dataset` with provenance recorded:

            >>> # Example setup for a Dataset with nested provenance
            >>> from arpes_analyzer.history import Provenance # Hypothetical import
            >>> ds = xr.Dataset()
            >>> ds.attrs['provenance'] = {
            ...     'step_name': 'filter',
            ...     'params': {'kernel_size': 3},
            ...     'parents_provenance': {
            ...         'step_name': 'normalize',
            ...         'params': {'method': 'max'},
            ...         'parents_provenance': {
            ...             'step_name': 'load_data',
            ...             'params': {'file': 'data.h5'}
            ...         }
            ...     }
            ... }
            >>> accessor = YourARPESAccessor(ds)
            >>> history_list = accessor.history
            >>> for entry in history_list:
            ...     print(entry['step_name'])
            filter
            normalize
            load_data

            >>> # Example with no provenance
            >>> ds_no_prov = xr.Dataset()
            >>> accessor_no_prov = YourARPESAccessor(ds_no_prov)
            >>> accessor_no_prov.history
            []

            >>> # Example with string provenance (warns)
            >>> ds_str_prov = xr.Dataset()
            >>> ds_str_prov.attrs['provenance'] = "Old string provenance record."
            >>> accessor_str_prov = YourARPESAccessor(ds_str_prov)
            >>> # This will print a warning and return ['Old string provenance record.']
            >>> accessor_str_prov.history
            ['Old string provenance record.']
        """
        provenance_recorded = self._obj.attrs.get("provenance", None)

        return unwrap_provenance(provenance_recorded)

    @property
    def parent_id(self) -> int | str | None:
        """Return id object of the parent object."""
        if not self.history:
            return None
        assert self.history is not None
        for a_history in reversed(self.history):
            if "parent_id" in a_history:
                return a_history["parent_id"]
        return None


class ARPESPropertyBase(ARPESInfoProperty, ARPESOffsetProperty, ARPESProvenanceProperty):
    """Base class for ARPES Property."""

    _obj: XrTypes

    @property
    def is_spatial(self) -> bool:
        """Infers whether a given scan has real-space dimensions (SPEM or u/nARPES).

        Returns:
            True if the data is explicitly a "ucut" or "spem" or contains "X", "Y", or "Z"
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
        """Return Reference setting."""
        settings = self.spectrometer_settings or {}

        settings.update(
            {
                "hv": self.hv,
            },
        )

        return settings

    @property
    def beamline_settings(self) -> BeamLineSettings:
        """Return beam line setting."""
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
        """Return spectrometer setting."""
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
            settings["slit"] = round(settings["slit"])

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
    """Class for ARPES property."""

    _obj: XrTypes

    @staticmethod
    def dict_to_html(d: Mapping[str, float | str]) -> str:
        """Return html format of dict object.

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
        return ARPESProperty.dict_to_html(self.spectrometer_settings)

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
                    spectrum.transpose("eV", ...).plot(ax=ax[i])  # type: ignore[reportCallIssue]
                    fancy_labels(ax[i])
                    ax[i].set_title(plot_var.replace("_", " "))

                remove_colorbars()

        elif 1 <= len(self._obj.dims) < 3:  # noqa: PLR2004
            _, ax = plt.subplots(1, 1, figsize=(4, 3))
            spectrum = self._obj
            spectrum.transpose("eV", ...).plot(ax=ax)  # type: ignore[reportCallIssue]
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
