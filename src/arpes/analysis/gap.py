"""Utilities for gap fitting in ARPES, contains tools to normalize by Fermi-Dirac occupation."""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from arpes.constants import K_BOLTZMANN_EV_KELVIN
from arpes.debug import setup_logger
from arpes.fits.fit_models import AffineBroadenedFD
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from lmfit.model import ModelResult
    from xarray.core.common import DataWithCoords

    from arpes._typing import DataType

__all__ = ("determine_broadened_fermi_distribution", "normalize_by_fermi_dirac", "symmetrize")


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def determine_broadened_fermi_distribution(
    reference_data: DataWithCoords,
    *,
    fixed_temperature: bool = True,
) -> ModelResult:
    """Determine the parameters for broadening by temperature and instrumental resolution.

    As a general rule, we first try to estimate the instrumental broadening and linewidth broadening
    according to calibrations provided for the beamline + instrument, as a starting point.

    We also calculate the thermal broadening to expect, and fit an edge location. Then we use a
    Gaussian convolved Fermi-Dirac distribution against an affine density of states near the Fermi
    level, with a constant offset background above the Fermi level as a simple but effective model
    when away from lineshapes.

    These parameters can be used to bootstrap a fit to actual data or used directly in
    ``normalize_by_fermi_dirac``.

    Args:
        reference_data: The data we want to estimate from.
        fixed_temperature: Whether we should force the temperature to the recorded value.

    Return:
        The estimated fit result for the Fermi distribution.
    """
    params = {}

    if fixed_temperature:
        params["width"] = {
            "value": reference_data.S.temp * K_BOLTZMANN_EV_KELVIN,
            "vary": False,
        }

    reference_data_array = (
        reference_data
        if isinstance(reference_data, xr.DataArray)
        else normalize_to_spectrum(reference_data)
    )

    sum_dims = list(reference_data_array.dims)
    sum_dims.remove("eV")

    return AffineBroadenedFD().guess_fit(reference_data_array.sum(sum_dims), params=params)


@update_provenance("Normalize By Fermi Dirac")
def normalize_by_fermi_dirac(  # noqa: PLR0913
    data: DataType,
    reference_data: DataType | None = None,
    broadening: float = 0,
    temperature_axis: str = "",
    temp_offset: float = 0,
    *,
    plot: bool = False,
    **kwargs: bool,
) -> xr.DataArray:
    """Normalizes data by Fermi level.

    Data normalization according to a Fermi level reference on separate data or using the same
    source spectrum.

    To do this, a linear density of states is multiplied against a resolution broadened Fermi-Dirac
    distribution (`arpes.fits.fit_models.AffineBroadenedFD`). We then set the density of states to
    1 and evaluate this model to obtain a reference that the desired spectrum is normalized by.

    Args:
        data: Data to be normalized.
        reference_data: A reference spectrum, typically a metal
            reference. If not provided the integrated data is used.
            Beware: this is inappropriate if your data is gapped.
        plot: A debug flag, allowing you to view the normalization
            spectrum and relevant curve-fits.
        broadening: Detector broadening.
        temperature_axis: Temperature coordinate, used to adjust the
            quality of the reference for temperature dependent data.
        temp_offset: Temperature calibration in the case of low
            temperature data. Useful if the temperature at the sample is
            known to be hotter than the value recorded off of a diode.
        **kwargs: pass to determine_broadened_fermi_distribution (Thus, fixed_temperature)

    Returns:
        Data after normalization by the Fermi occupation factor.
    """
    reference_data = data if reference_data is None else reference_data
    broadening_fit = determine_broadened_fermi_distribution(reference_data, **kwargs)
    broadening = broadening_fit.params["conv_width"].value if broadening is None else broadening

    if plot:
        msg = f"Gaussian broadening is: {broadening_fit.params['conv_width'].value * 1000} meV"
        msg += " (Gaussian sigma)"
        logger.info(msg)
        msg = f"Fermi edge location is: {broadening_fit.params['center'].value * 1000} meV"
        msg += " (fit chemical potential)"
        logger.info(msg)
        msg = f"Fermi width is: {broadening_fit.params['width'].value * 1000} meV"
        msg += " (fit fermi width)"
        logger.info(msg)

        broadening_fit.plot()

    offset = broadening_fit.params["offset"].value
    without_offset = broadening_fit.eval(offset=0)

    cut_index = -np.argmax(without_offset[::-1] > 0.1 * offset)
    cut_energy = reference_data.coords["eV"].values[cut_index]

    if (not temperature_axis) and "temp" in data.dims:
        temperature_axis = "temp"

    transpose_order: list[str] = [str(dim) for dim in data.dims]
    transpose_order.remove("eV")

    if temperature_axis:
        transpose_order.remove(temperature_axis)
        transpose_order = [*transpose_order, temperature_axis]

    transpose_order = [*transpose_order, "eV"]

    without_background = (data - data.sel(eV=slice(cut_energy, None)).mean("eV")).transpose(
        *transpose_order,
    )
    # <== NEED TO CHECK (What it the type of without_background ?)

    without_background_arr = normalize_to_spectrum(without_background)
    assert isinstance(without_background_arr, xr.DataArray)
    if temperature_axis:
        divided = without_background_arr.G.map_axes(
            temperature_axis,
            lambda x, coord: x
            / broadening_fit.eval(
                x=x.coords["eV"].values,
                lin_bkg=0,
                const_bkg=1,
                offset=0,
                conv_width=broadening,
                width=(coord[temperature_axis] + temp_offset) * K_BOLTZMANN_EV_KELVIN,
            ),
        )
    else:
        divided = without_background_arr / broadening_fit.eval(
            x=data.coords["eV"].values,
            conv_width=broadening,
            lin_bkg=0,
            const_bkg=1,
            offset=0,
        )

    divided.coords["eV"].values = (
        divided.coords["eV"].values - broadening_fit.params["center"].value
    )
    return divided


def _shift_energy_interpolate(
    data: xr.DataArray,
    shift: xr.DataArray | None = None,
) -> xr.DataArray:
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    data_arr = data.transpose("eV", ...)

    new_data = data_arr.copy(deep=True)
    new_axis = new_data.coords["eV"]
    new_values = new_data.values * 0

    if shift is None:
        closest_to_zero = data_arr.coords["eV"].sel(eV=0, method="nearest")
        shift = -closest_to_zero

    assert isinstance(shift, xr.DataArray)
    stride = data_arr.G.stride("eV", generic_dim_names=False)

    if np.abs(shift) >= stride:
        n_strides = int(shift / stride)
        new_axis = new_axis + n_strides * stride

        shift = shift - stride * n_strides

    new_axis = new_axis + shift
    assert shift is not None

    weight = float(shift / stride)
    new_values = new_values + data_arr.values * (1 - weight)
    if shift > 0:
        new_values[1:] += data_arr.values[:-1] * weight
    if shift < 0:
        new_values[:-1] += data_arr.values[1:] * weight

    new_data.coords["eV"] = new_axis
    new_data.values = new_values

    return new_data


@update_provenance("Symmetrize")
def symmetrize(
    data: xr.DataArray,
    *,
    subpixel: bool = False,
    full_spectrum: bool = False,
) -> xr.DataArray:
    """Symmetrizes data across the chemical potential.

    This provides a crude tool by which
    gap analysis can be performed. In this implementation, subpixel accuracy is achieved by
    interpolating data.

    Args:
        data: Input array.
        subpixel: Enable subpixel correction
        full_spectrum: Returns data above and below the chemical
            potential. By default, only the bound part of the spectrum
            (below the chemical potential) is returned, because the
            other half is identical.

    Returns:
        The symmetrized data.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    data = data.transpose("eV", ...)

    if subpixel or full_spectrum:
        data = _shift_energy_interpolate(data)
    assert isinstance(data, xr.DataArray)

    above = data.sel(eV=slice(0, None))
    below = data.sel(eV=slice(None, 0)).copy(deep=True)

    length_eV_coords = len(above.coords["eV"])

    zeros = below.values * 0
    zeros[-length_eV_coords:] = above.values[::-1]

    below.values += zeros

    if full_spectrum:
        if not subpixel:
            warnings.warn("full spectrum symmetrization uses subpixel correction", stacklevel=2)

        full_data = below.copy(deep=True)

        new_above = full_data.copy(deep=True)[::-1]
        new_above.coords["eV"] = new_above.coords["eV"] * -1

        full_data = xr.concat([full_data, new_above[1:]], dim="eV")

        result = full_data
    else:
        result = below

    return result
