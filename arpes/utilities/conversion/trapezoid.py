"""Implements forward and reverse trapezoidal corrections."""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

import numba
import numpy as np
import xarray as xr

from arpes.trace import traceable
from arpes.utilities import normalize_to_spectrum

from .base import CoordinateConverter
from .core import convert_coordinates

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete
    from numpy.typing import NDArray
    from xarray.core.indexes import Indexes

__all__ = ["apply_trapezoidal_correction"]

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


@numba.njit(parallel=True)
def _phi_to_phi(
    energy: NDArray[np.float_],
    phi: NDArray[np.float_],
    phi_out: NDArray[np.float_],
    corner_angles: tuple[float, float, float, float],
) -> None:
    """Performs reverse coordinate interpolation using four angular waypoints.

    Args:
        energy: The binding energy in the corrected coordinate space
        phi: The angle in the corrected coordinate space
        phi_out: The array to populate with the measured phi angles
        corner_angles: (tuple[float, float, float, float]) the values for the edge of the
            hemisphere's range.  (l_fermi, l_volt, r_fermi, r_volt)
            l_fermi: The measured phi coordinate of the left edge of the hemisphere's range
                at the Fermi level
            l_volt: The measured phi coordinate of the left edge of the hemisphere's range
                at a binding energy of 1 eV (eV = -1.0)
            r_fermi: The measured phi coordinate of the right edge of the hemisphere's range
                at the Fermi level
            r_volt: The measured phi coordinate of the right edge of the hemisphere's range
                at a binding energy of 1 eV (eV = -1.0)
    """
    l_fermi, l_volt, r_fermi, r_volt = corner_angles
    for i in numba.prange(len(phi)):
        left_edge = l_fermi - energy[i] * (l_volt - l_fermi)
        right_edge = r_fermi - energy[i] * (r_volt - r_fermi)

        # These are the forward equations, we can just invert them below

        dac_da = (right_edge - left_edge) / (r_fermi - l_fermi)
        phi_out[i] = (phi[i] - l_fermi) * dac_da + left_edge


@numba.njit(parallel=True)
def _phi_to_phi_forward(
    energy: NDArray[np.float_],
    phi: NDArray[np.float_],
    phi_out: NDArray[np.float_],
    corner_angles: tuple[float, float, float, float],
) -> None:
    """The inverse transform to ``_phi_to_phi``. See that function for details."""
    l_fermi, l_volt, r_fermi, r_volt = corner_angles
    for i in numba.prange(len(phi)):
        left_edge = l_fermi - energy[i] * (l_volt - l_fermi)
        right_edge = r_fermi - energy[i] * (r_volt - r_fermi)

        # These are the forward equations
        c = (phi[i] - left_edge) / (right_edge - left_edge)
        phi_out[i] = l_fermi + c * (r_fermi - l_fermi)


class ConvertTrapezoidalCorrection(CoordinateConverter):
    """A converter for applying the trapezoidal correction to ARPES data."""

    def __init__(
        self,
        *args: Incomplete,
        corners: list[dict[str, float]],
        **kwargs: Incomplete,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.phi = None

        # we normalize the corners so that they are equivalent to four corners at the Fermi level
        # and one volt below.
        c1, c2, c3, c4 = sorted(corners, key=lambda x: x["phi"])
        c1, c2 = sorted([c1, c2], key=lambda x: x["eV"])
        c3, c4 = sorted([c3, c4], key=lambda x: x["eV"])

        # now, corners are in
        # (c1, c2, c3, c4) = (LL, UL, LR, UR) order

        left_per_volt = (c1["phi"] - c2["phi"]) / (c1["eV"] - c2["eV"])
        left_phi_fermi = c2["phi"] - c2["eV"] * left_per_volt
        left_phi_one_volt = left_phi_fermi - left_per_volt

        right_per_volt = (c3["phi"] - c4["phi"]) / (c3["eV"] - c4["eV"])
        right_phi_fermi = c3["phi"] - c4["eV"] * right_per_volt
        right_phi_one_volt = right_phi_fermi - right_per_volt

        self.corner_angles = (
            left_phi_fermi,
            left_phi_one_volt,
            right_phi_fermi,
            right_phi_one_volt,
        )

    def get_coordinates(self, *args: Incomplete, **kwargs: Incomplete) -> Indexes:  # TODO: rename !
        if args:
            logger.debug("ConvertTrapezoidalCorrection.get_coordinates: args is not used but set.")
        if kwargs:
            for k, v in kwargs.items():
                msg = f"ConvertTrapezoidalCorrection.get_coordinates: key({k}: value{v} is not used"
                msg += " but set."
                logger.debug(msg)

        return self.arr.indexes

    def conversion_for(self, dim: str) -> Callable[..., NDArray[np.float_]]:
        def with_identity(*args: Incomplete) -> NDArray[np.float_]:
            return self.identity_transform(dim, *args)

        return {
            "phi": self.phi_to_phi,
        }.get(dim, with_identity)

    def phi_to_phi(
        self,
        binding_energy: NDArray[np.float_],
        phi: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """[TODO:summary].

        Args:
            binding_energy: [TODO:description]
            phi: [TODO:description]
            args: [TODO:description]
            kwargs: [TODO:description]
        """
        if self.phi is not None:
            return self.phi
        self.phi = np.zeros_like(phi)
        _phi_to_phi(binding_energy, phi, self.phi, self.corner_angles)
        return self.phi

    def phi_to_phi_forward(
        self,
        binding_energy: NDArray[np.float_],
        phi: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        """[TODO:summary].

        Args:
            binding_energy: [TODO:description]
            phi: [TODO:description]
            args: [TODO:description]
            kwargs: [TODO:description]
        """
        phi_out = np.zeros_like(phi)
        _phi_to_phi_forward(binding_energy, phi, phi_out, self.corner_angles)
        return phi_out


@traceable
def apply_trapezoidal_correction(
    data: xr.DataArray,
    corners: list[dict[str, float]],
    trace: Callable | None = None,
) -> xr.DataArray:
    """Applies the trapezoidal correction to data in angular units by linearly interpolating slices.

    Shares some code with standard coordinate conversion, i.e. to momentum, because you can think of
    this as performing a coordinate conversion between two angular coordinate sets, the measured
    angles and the true angles.

    Args:
        data: The xarray instances to perform correction on
        corners: These don't actually have to be corners, but are waypoints of the conversion. Use
            points near the Fermi level and near the bottom of the spectrum just at the edge of
            recorded angular region.
        trace: A trace instance which can be used to enable execution tracing and debugging.
               Pass ``True`` to enable.


    Returns:
        The corrected data.
    """
    if isinstance(data, dict):
        warnings.warn(
            "Treating dict-like data as an attempt to forward convert a single coordinate.",
            stacklevel=2,
        )
        converter = ConvertTrapezoidalCorrection(None, [], corners=corners)
        result = dict(data)
        result["phi"] = converter.phi_to_phi_forward(
            np.array([data["eV"]]),
            np.array([data["phi"]]),
        )[0]
        return result

    if isinstance(data, xr.Dataset):
        msg = "Remember to use a DataArray not a Dataset, "
        msg += "attempting to extract spectrum and copy attributes."
        warnings.warn(msg, stacklevel=2)
        attrs = data.attrs.copy()
        data = normalize_to_spectrum(data)
        assert isinstance(data, xr.DataArray)
        data.attrs.update(attrs)

    original_coords = data.coords

    trace("Determining dimensions.") if trace else None
    if "phi" not in data.dims:
        msg = "The data must have a phi coordinate."
        raise ValueError(msg)
    trace("Replacing dummy coordinates with index-like ones.") if trace else None
    removed = [d for d in data.dims if d not in ["eV", "phi"]]
    data = data.transpose(*(["eV", "phi", *removed]))
    converted_dims = data.dims

    restore_index_like_coordinates = {r: data.coords[r].values for r in removed}
    new_index_like_coordinates = {r: np.arange(len(data.coords[r].values)) for r in removed}
    data = data.assign_coords(new_index_like_coordinates)

    converter = ConvertTrapezoidalCorrection(data, converted_dims, corners=corners)
    converted_coordinates = converter.get_coordinates()

    trace("Calling convert_coordinates") if trace else None
    result = convert_coordinates(
        data,
        converted_coordinates,
        {
            "dims": data.dims,
            "transforms": dict(
                zip(data.dims, [converter.conversion_for(d) for d in data.dims], strict=True),
            ),
        },
        trace=trace,
    )
    assert isinstance(result, xr.DataArray)
    trace("Reassigning index-like coordinates.") if trace else None
    result = result.assign_coords(restore_index_like_coordinates)
    result = result.assign_coords(
        {c: v for c, v in original_coords.items() if c not in result.coords},
    )
    return result.assign_attrs(data.attrs)
