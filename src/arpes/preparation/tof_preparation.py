"""Data prep routines for time-of-flight data."""

from __future__ import annotations  # noqa: I001

import math

import numpy as np
import xarray as xr

from arpes.provenance import update_provenance

from .axis_preparation import transform_dataarray_axis
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = [
    "build_KE_coords_to_time_pixel_coords",
    "build_KE_coords_to_time_coords",
    "process_DLD",
    "process_SToF",
]


@update_provenance("Convert ToF data from timing signal to kinetic energy")
def convert_to_kinetic_energy(
    dataarray: xr.DataArray,
    kinetic_energy_axis: NDArray[np.float_],
) -> xr.DataArray:
    """Convert the ToF timing information into an energy histogram.

    The core of these routines come from the Igor procedures in
    ``LoadTOF7_3.51.ipf``.

    To explain in a bit more detail what actually happens, this
    function essentially

    1. Determines the time to energy conversion factor
    2. Calculates the requested energy binning range and allocates arrays
    3. Rebins a time spectrum into an energy spectrum, preserving the
       spectral weight, this requires a modicum of care around splitting
       counts at the edges of the new bins.
    """
    # This should be simplified
    # Removed factors of ten and substituted mstar = 0.5
    c = (
        (0.5)
        * (9.11e6)
        * dataarray.S.spectrometer["mstar"]
        * (dataarray.S.spectrometer["length"] ** 2)
        / 1.6
    )

    new_dim_order = list(dataarray.dims)
    new_dim_order.remove("time")
    new_dim_order = ["time", *new_dim_order]
    dataarray = dataarray.transpose(*new_dim_order)
    new_dim_order[0] = "eV"

    timing = dataarray.coords["time"].values
    assert timing[1] > timing[0]
    t_min, t_max = np.min(timing), np.max(timing)

    # Prep arrays
    d_energy = kinetic_energy_axis[1] - kinetic_energy_axis[0]
    time_index = 0  # after transpose
    new_shape = list(dataarray.data.shape)
    new_shape[time_index] = len(kinetic_energy_axis)

    new_data = np.zeros(tuple(new_shape))

    def energy_to_time(conv: float, energy: float) -> float:
        return math.sqrt(conv / energy)

    # Rebin data
    old_data = dataarray.data
    for i, E in enumerate(kinetic_energy_axis):
        t_L = energy_to_time(c, E + d_energy / 2)
        t_S = energy_to_time(c, E - d_energy / 2)

        # clamp
        t_L = t_L if t_L <= t_max else t_max
        t_S = t_S if t_S > t_min else t_min

        # with some math we could back calculate the index, but we don't need to
        t_L_idx = np.searchsorted(timing, t_L)
        t_S_idx = np.searchsorted(timing, t_S)

        new_data[i] = (
            np.sum(old_data[t_L_idx:t_S_idx], axis=0)
            + ((timing[t_L_idx] - t_L) * old_data[t_L_idx])
            + ((t_S - timing[t_S_idx - 1]) * old_data[t_S_idx - 1]) / d_energy
        )

    new_coords = dict(dataarray.coords)
    del new_coords["time"]
    new_coords["eV"] = kinetic_energy_axis

    # Put provenance here

    return xr.DataArray(
        new_data,
        coords=new_coords,
        dims=new_dim_order,
        attrs=dataarray.attrs,
        name=dataarray.name,
    )


def build_KE_coords_to_time_pixel_coords(
    dataset: xr.Dataset,
    interpolation_axis: NDArray[np.float_],
) -> Callable[..., tuple[xr.DataArray]]:
    """Constructs a coordinate conversion function from kinetic energy to time pixels."""
    conv = (
        dataset.S.spectrometer["mstar"]
        * (9.11e6)
        * 0.5
        * (dataset.S.spectrometer["length"] ** 2)
        / 1.6
    )
    time_res = 0.17  # this is only approximate

    def KE_coords_to_time_pixel_coords(
        coords: xr.DataArray,
        axis: str = "",
    ) -> tuple[xr.DataArray, ...]:
        """Like ``KE_coords_to_time_coords`` but it converts to the raw timing pixels.

        These typically come off of a DLD.
        This is instead to the unitful values that we receive from the Spin-ToF DAQ.

        Args:
            coords: tuple of coordinates
            axis: Which axis to use for the kinetic energy

        Returns:
            New tuple of converted coordinates
        """
        kinetic_energy_pixel = coords[axis]
        kinetic_energy = interpolation_axis[kinetic_energy_pixel]
        real_timing = math.sqrt(conv / kinetic_energy)
        pixel_timing = (real_timing - dataset.attrs["timing_offset"]) / time_res
        coords_list = list(coords)
        coords_list[axis] = pixel_timing

        return tuple(coords_list)

    return KE_coords_to_time_pixel_coords


def build_KE_coords_to_time_coords(
    dataset: xr.Dataset,
    interpolation_axis: NDArray[np.float_],
) -> Callable[..., tuple[xr.DataArray]]:
    """Constructs a coordinate conversion function from kinetic energy to time coords.

    Geometric transform assumes pixel -> pixel transformations so we need to get
    the index associated to the appropriate timing value
    """
    conv = (
        dataset.S.spectrometer["mstar"]
        * (9.11e6)
        * 0.5
        * (dataset.S.spectrometer["length"] ** 2)
        / 1.6
    )
    timing = dataset.coords["time"]
    photon_offset = dataset.attrs["laser_t0"] + dataset.S.spectrometer["length"] * (10 / 3)
    low_offset = np.min(timing)
    d_timing = timing[1] - timing[0]

    def KE_coords_to_time_coords(
        coords: xr.DataArray,
        axis: str = "",
    ) -> tuple[xr.DataArray]:
        """Used to convert the timing coordinates off of the spin-ToF to kinetic energy coordinates.

        As the name suggests, because of how scipy.ndimage interpolates, we require the inverse
        coordinate transform.

        All coordinates except for the energy coordinate are left untouched.

        We do the same logic done implicitly in timeProcessX in order to get the part of the data
        that has time coordinate less than the nominal t0. This is necessary because the recorded
        times are the time between electron events and laser pulses, rather than the other way
        around.

        Args:
            coords: tuple of coordinates
            axis: Energy axis name

        Return:
            new tuple of converted coordinates
        """
        kinetic_energy = interpolation_axis[coords[axis]]
        real_timing = math.sqrt(conv / kinetic_energy)
        real_timing = photon_offset - real_timing
        coords_list = list(coords)
        coords_list[axis] = len(timing) - (real_timing - low_offset) / d_timing
        return tuple(coords_list)

    return KE_coords_to_time_coords


@update_provenance("Convert ToF data from timing signal to kinetic energy ALT")
def convert_SToF_to_energy(dataset: xr.Dataset) -> xr.Dataset:
    """Achieves the same computation as timeProcessX and t2energyProcessX in LoadTOF_3.51.ipf.

    Args:
        dataset: input data

    Returns:
        The energy converted data.
    """
    e_min, e_max = 0.1, 10.0

    # TODO: we can better infer a reasonable gridding here
    spacing = dataset.attrs.get("dE", 0.005)
    ke_axis = np.linspace(e_min, e_max, int((e_max - e_min) / spacing))

    drs = {k: spectrum for k, spectrum in dataset.data_vars.items() if "time" in spectrum.dims}

    new_dataarrays = [convert_to_kinetic_energy(dr, ke_axis) for dr in drs]

    for v in new_dataarrays:
        dataset[str(v.name).replace("t_", "")] = v

    return dataset


@update_provenance("Preliminary data processing Spin-ToF")
def process_SToF(dataset: xr.Dataset) -> xr.Dataset:
    """Converts spin-ToF data to energy units.

    This isn't the best unit conversion function because it doesn't properly
    take into account the Jacobian of the coordinate conversion. This can
    be fixed by multiplying each channel by the appropriate amount, but it might still
    be best to use the alternative method.
    """
    e_min = dataset.attrs.get("E_min", 1)
    e_max = dataset.attrs.get("E_max", 10)
    de = dataset.attrs.get("dE", 0.01)
    ke_axis = np.linspace(e_min, e_max, (e_max - e_min) / de)

    dataset = transform_dataarray_axis(
        build_KE_coords_to_time_coords(dataset, ke_axis),
        ("time", "eV"),
        ke_axis,
        dataset,
        lambda x: x,
    )

    dataset = dataset.rename({"t_up": "up", "t_down": "down"})

    if "up" in dataset.data_vars:
        # apply the sherman function corrections
        sherman = dataset.attrs.get("sherman", 0.2)
        polarization = 1 / sherman * (dataset.up - dataset.down) / (dataset.up + dataset.down)
        new_up = (dataset.up + dataset.down) * (1 + polarization)
        new_down = (dataset.up + dataset.down) * (1 - polarization)
        dataset = dataset.assign(up=new_up, down=new_down)

    return dataset


@update_provenance("Preliminary data processing prototype DLD")
def process_DLD(dataset: xr.Dataset) -> xr.Dataset:
    """Converts delay line data to kinetic energy coordinates."""
    e_min = 1
    ke_axis = np.linspace(
        e_min,
        dataset.attrs["E_max"],
        (dataset.attrs["E_max"] - e_min) / dataset.attrs["dE"],
    )
    return transform_dataarray_axis(
        build_KE_coords_to_time_pixel_coords(dataset, ke_axis),
        ("t_pixels", "kinetic"),
        ke_axis,
        dataset,
        lambda: "kinetic_spectrum",
    )