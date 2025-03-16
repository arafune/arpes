"""Implements forward and reverse trapezoidal corrections.

There are two types of trapezoidal correction for ARPES data: one that results in a trapezoidal
shape and one that starts with a trapezoidal shape.

In the original version (<= v3.0), only the first one is considered.
The need for trapezoidal correction is not very common. However, there are cases where one may want
to apply trapezoidal correction to measured data. Additionally, while it may have been a local
requirement specific to their group, the process in the ConvertTrapezoidCorrection's __init__
method does not seem correct.

Since there have been significant changes in the specifications, caution is required if this
feature was used in a previous version.
"""

from __future__ import annotations

import operator
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, TypeGuard

import numba
import numpy as np
import xarray as xr

from arpes.debug import setup_logger

from .base import CoordinateConverter
from .core import convert_coordinates

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import MOMENTUM

__all__ = ["apply_trapezoidal_correction"]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@numba.njit(parallel=True)
def _phi_to_phi(
    energy: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_out: NDArray[np.float64],
    corner_angles: tuple[float, float, float, float],
) -> None:
    """Performs reverse coordinate interpolation using four angular waypoints.

    Transform from rectangle to trapezoid.

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
    energy: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_out: NDArray[np.float64],
    corner_angles: tuple[float, float, float, float],
) -> None:
    """The inverse transform to ``_phi_to_phi`` (See that function for details).

    Transform from trapezoid to rectangle
    """
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
        """[TODO:summary].

        Args:
            args: [TODO:description]
            corners: corner coordinates of the trapezoid.
            kwargs: [TODO:description]
        """
        super().__init__(*args, **kwargs)
        self.phi = None

        # we normalize the corners so that they are equivalent to four corners at the Fermi level
        # and one volt below.
        lower_left, upper_left, lower_right, upper_right = sorted(
            corners,
            key=operator.itemgetter("phi"),
        )
        lower_left, upper_left = sorted([lower_left, upper_left], key=operator.itemgetter("eV"))
        lower_right, upper_right = sorted([lower_right, upper_right], key=operator.itemgetter("eV"))

        left_per_volt = (lower_left["phi"] - upper_left["phi"]) / (
            lower_left["eV"] - upper_left["eV"]
        )
        left_phi_fermi = upper_left["phi"]
        left_phi_one_volt = left_phi_fermi - left_per_volt

        right_per_volt = (lower_right["phi"] - upper_right["phi"]) / (
            lower_right["eV"] - upper_right["eV"]
        )
        right_phi_fermi = lower_right["phi"]
        right_phi_one_volt = right_phi_fermi - right_per_volt

        self.corner_angles = (
            left_phi_fermi,
            left_phi_one_volt,
            right_phi_fermi,
            right_phi_one_volt,
        )

    def get_coordinates(
        self,
        resolution: dict[MOMENTUM, float] | None = None,
        bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    ) -> dict[Hashable, NDArray[np.float64]]:
        """[TODO:summary].

        [TODO:description]

        Args:
            resolution: [TODO:description]
            bounds: [TODO:description]

        Returns:
            [TODO:description]
        """
        del resolution
        del bounds
        return {k: v.values for k, v in self.arr.indexes.items()}

    def conversion_for(self, dim: Hashable) -> Callable[..., NDArray[np.float64]]:
        def _with_identity(*args: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.identity_transform(dim, *args)

        return {
            "phi": self.phi_to_phi,
        }.get(
            str(dim),
            _with_identity,
        )

    def phi_to_phi(
        self,
        binding_energy: NDArray[np.float64],
        phi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Converts the given phi values to a new phi representation based on binding energy.

        This method computes the new phi values based on the provided binding energy and phi values,
        and stores the result in `self.phi`. If `self.phi` is already set, it simply returns
        the existing value.

        Args:
            binding_energy (NDArray[np.float64]): The array of binding energy values.
            phi (NDArray[np.float64]): The array of phi values to be converted.

        Returns:
            NDArray[np.float64]: The transformed phi values.

        Raises:
            ValueError: If any required attributes are missing or invalid.
        """
        if self.phi is not None:
            return self.phi
        self.phi = np.zeros_like(phi)
        _phi_to_phi(binding_energy, phi, self.phi, self.corner_angles)
        return self.phi

    def phi_to_phi_forward(
        self,
        binding_energy: NDArray[np.float64],
        phi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Transforms phi values based on binding energy using a forward method.

        This method computes the new phi values based on the provided binding energy and phi values,
        applying a forward transformation. The result is stored in the `phi_out` array.

        Args:
            binding_energy (NDArray[np.float64]): The array of binding energy values.
            phi (NDArray[np.float64]): The array of phi values to be converted.

        Returns:
            NDArray[np.float64]: The transformed phi values after the forward transformation.
        """
        phi_out = np.zeros_like(phi)
        _phi_to_phi_forward(binding_energy, phi, phi_out, self.corner_angles)
        return phi_out


def apply_trapezoidal_correction(
    data: xr.DataArray,
    corners: list[dict[str, float] | float],
    from_trapezoid: bool = True,
) -> xr.DataArray:
    r"""Applies the trapezoidal correction in angular units by linearly interpolating slices.

    Shares some code with standard coordinate conversion, i.e. to momentum, because you can think of
    this as performing a coordinate conversion between two angular coordinate sets, the measured
    angles and the true angles.



           (UL)_____________ (UR)          (UL) +--------+   (UR)
        ↑      \           /                    |        |
        |       \         /       ↔             |        |
        eV       \_______/                 (LL) +--------+   (LR)
            (LL)          (LR)

                          ----------→ phi

    Args:
        data: The xarray instances to perform correction on
        corners: The coordinate of the trapezoid corners. (thus, len(corners)==4)  If it is dict,
            the key must be both "eV" and "phi", which is used in from_trapezoid=True.
            If it is list, the for corners (LL, UL, LR, UR), which is used in from_trapezoid=False
            (dict arg can be used in the case from_trapezoid=False).
        from_trapezoid: bool, if True, transpose to rectangle. in this case the corners are
            set as those of the trapezoid (left figure).  If False, trapspose *to* trapezoid. In
            this case, the corners indicate the points to which the maximum and minimum values
            of eV and phi in the original data are mapped, respectively.

    Returns:
        The corrected data.
    """
    assert isinstance(data, xr.DataArray)
    assert "phi" in data.coords, "The data must have a phi coordinate."
    assert len(corners) == len(("LL", "UL", "LR", "UR"))
    eV_max, eV_min = data.coords["eV"].max().item, data.coords["eV"].min().item
    if _is_all_floats(corners):
        trapezoid_corners = [
            {"eV": eV_min, "phi": corners[0]},
            {"eV": eV_max, "phi": corners[1]},
            {"eV": eV_min, "phi": corners[2]},
            {"eV": eV_max, "phi": corners[3]},
        ]
    elif _is_all_dicts(corners):
        trapezoid_corners = corners
    else:
        msg = "corners should be list of dict or list of float."
        raise TypeError(msg)
    logger.debug("Determining dimensions.")
    data = data.transpose("eV", "phi", ...)
    converted_dims = data.dims

    converter = ConvertTrapezoidalCorrection(
        arr=data,
        converted_dims=converted_dims,
        corners=trapezoid_corners,
    )
    converted_coordinates = converter.get_coordinates()

    logger.debug("Calling convert_coordinates")
    result = convert_coordinates(
        arr=data,
        target_coordinates=converted_coordinates,
        coordinate_transform={
            "dims": list(data.dims),
            "transforms": {str(dim): converter.conversion_for(dim) for dim in data.dims},
        },
    )
    assert isinstance(result, xr.DataArray)
    logger.debug("Reassigning index-like coordinates.")
    return result.assign_attrs(data.attrs)


def _is_all_floats(corners: List[float | int]) -> TypeGuard[list[float]]:
    return all(isinstance(corner, float) for corner in corners)


def _is_all_dicts(corners: List[Any]) -> TypeGuard[List[dict[str, float]]]:
    return all(isinstance(corner, dict) for corner in corners)
