"""Implements forward and reverse trapezoidal corrections.

There are two types of trapezoidal correction for ARPES data: one that results in a trapezoidal
shape and one that starts with a trapezoidal shape.

In the original version (<= v3.0), only the first one is considered.
The trapezoidal correction is so frequently needed. However, there are cases where one may want
to apply trapezoidal correction to measured data. Additionally, while it may have been a local
requirement specific to their group, the process in the original ConvertTrapezoidCorrection's
__init__ method does not seem correct.

Since there have been significant changes in the specifications, caution is required if this
feature was used in a previous version.
"""

from __future__ import annotations

import operator
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, TypeGuard, TypeVar

import numba
import numpy as np
import xarray as xr
from numba import typed, types

from arpes.debug import setup_logger
from arpes.utilities.conversion.base import CoordinateConverter
from arpes.utilities.conversion.core import convert_coordinates

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from _typeshed import Incomplete
    from numpy.typing import NDArray


__all__ = ["trapezoid"]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@numba.njit(parallel=True)
def _phi_to_phi(
    energy: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_out: NDArray[np.float64],
    corners: typed.typeddict.Dict[str, typed.typeddict.Dict[str, float]],
    rectangle_phis: list[float],
) -> None:
    """Performs reverse coordinate interpolation using four angular waypoints.

    Transform from rectangle to trapezoid.

    Args:
        energy: The binding energy in the corrected coordinate space.
        phi: The angle in the corrected coordinate space.
        phi_out: The array to populate with the measured phi angles.
        corners (dict[str, dict[str, float]]): The values for the edge of the trapezoid
            (the hemisphere's range).
        rectangle_phis (list[float, float]): the min and max value of the rectangle frame.

    Returns:
        None: The function modifies the phi_out array in-place.

    Notes:
        This function uses Numba's njit decorator to compile the function just-in-time,
        which can improve performance.
    """
    for i in numba.prange(len(phi)):
        slope_left_edge_ = (corners["upper_left"]["phi"] - corners["lower_left"]["phi"]) / (
            corners["upper_left"]["eV"] - corners["lower_left"]["eV"]
        )
        slope_right_edge_ = (corners["upper_right"]["phi"] - corners["lower_right"]["phi"]) / (
            corners["upper_right"]["eV"] - corners["lower_right"]["eV"]
        )
        left_edge = (
            slope_left_edge_ * (energy[i] - corners["upper_left"]["eV"])
            + corners["upper_left"]["phi"]
        )
        right_edge = (
            slope_right_edge_ * (energy[i] - corners["upper_right"]["eV"])
            + corners["upper_right"]["phi"]
        )

        dac_da = (right_edge - left_edge) / (max(rectangle_phis) - min(rectangle_phis))
        phi_out[i] = (phi[i] - min(rectangle_phis)) * dac_da + left_edge


@numba.njit(parallel=True)
def _phi_to_phi_forward(
    energy: NDArray[np.float64],
    phi: NDArray[np.float64],
    phi_out: NDArray[np.float64],
    corners: typed.typeddict.Dict[str, typed.typeddict.Dict[str, float]],
    rectangle_phis: list[float],
) -> None:
    """Transform from trapezoid to rectangle.

    This function performs the inverse transform of `_phi_to_phi`.
    It takes in the energy and phi values of a trapezoid and outputs the corresponding phi values of
    a rectangle.

    Args:
        energy : NDArray[np.float64]
            The energy values of the trapezoid.
        phi : NDArray[np.float64]
            The phi values of the trapezoid.
        phi_out : NDArray[np.float64]
            The output phi values of the rectangle.
        corners : dict[str, dict[str, float]]
            The corners of the trapezoid, each corner is a dictionary with 'eV' and 'phi' keys.
        rectangle_phis : list[float]
            The phi values of the rectangle.

    Returns: None
        The function modifies the phi_out array in-place.

    Notes:
        This function uses Numba's njit decorator to compile the function just-in-time, which can
        improve performance.
    """
    for i in numba.prange(len(phi)):
        slope_left_edge_ = (corners["upper_left"]["phi"] - corners["lower_left"]["phi"]) / (
            corners["upper_left"]["eV"] - corners["lower_left"]["eV"]
        )
        slope_right_edge_ = (corners["upper_right"]["phi"] - corners["lower_right"]["phi"]) / (
            corners["upper_right"]["eV"] - corners["lower_right"]["eV"]
        )
        left_edge = (
            slope_left_edge_ * (energy[i] - corners["upper_left"]["eV"])
            + corners["upper_left"]["phi"]
        )
        right_edge = (
            slope_right_edge_ * (energy[i] - corners["upper_right"]["eV"])
            + corners["upper_right"]["phi"]
        )

        # These are the forward equations
        c = (phi[i] - left_edge) / (right_edge - left_edge)
        phi_out[i] = min(rectangle_phis) + c * (max(rectangle_phis) - min(rectangle_phis))


class ConvertTrapezoidalCorrection(CoordinateConverter):
    """A converter for applying the trapezoidal correction to ARPES data."""

    def __init__(
        self,
        *args: Incomplete,
        corners: list[dict[str, float]],
        rectangle_phis: list[float],
        **kwargs: Incomplete,
    ) -> None:
        """Initializes a ConvertTrapezoidalCorrection instance.

        Args:
            *args: Variable length argument list.
            corners (list[dict[str, float]]): The corner coordinates of the trapezoid.
                Each dictionary should have the following keys: "phi", "eV".
            rectangle_phis (list[float]): the min and max phi value of the rectangle frame.
            **kwargs: Incompliete

        Note:
            The corners list should contain exactly 4 dictionaries, representing the 4 corners of
            the trapezoid.
            The rectangle_phis list should contain exactly 2 float values.
        """
        super().__init__(*args, **kwargs)
        self.phi = None
        self.corners: typed.typeddict.Dict[str, typed.typeddict.Dict[str, float]] = (
            _corners_typed_dict(corners)
        )

        self.rectangle_phis = rectangle_phis

    def get_coordinates(
        self,
        resolution: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> dict[Hashable, NDArray[np.float64]]:
        """Calculates the coordinates which should be used in correced data.

        Args:
            resolution(dict): Represents corrected resolution
            bounds(dict, optional): bounds of the momentum coordinates

        Returns: dict[str, NDArray[np.float]
            Object that is to be used the coordinates in the corrected dat.
        """
        resolution = resolution if resolution is not None else {}
        bounds = bounds if bounds is not None else {}

        coordinates = {k: v.values for k, v in self.arr.coords.items()}
        if "phi" in bounds:
            phi_low, phi_high = bounds["phi"]
        else:
            phi_low, phi_high = (
                self.arr.coords["phi"].min().item(),
                self.arr.coords["phi"].max().item(),
            )
        coordinates["phi"] = np.arange(
            phi_low,
            phi_high,
            resolution.get("phi", self.arr.G.stride("phi", generic_dim_names=False)),
        )
        logger.debug(f"coordinates: {coordinates}")
        return coordinates

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
            rectangle_phis (list[float]): max and min of the angle phi in the rectangle.

        Returns:
            NDArray[np.float64]: The transformed phi values.

        Raises:
            ValueError: If any required attributes are missing or invalid.
        """
        if self.phi is not None:
            return self.phi
        self.phi = np.zeros_like(phi)
        _phi_to_phi(
            energy=binding_energy,
            phi=phi,
            phi_out=self.phi,
            corners=self.corners,
            rectangle_phis=self.rectangle_phis,
        )
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
            rectangle_phis (list[float]): max and min of the angle phi in the rectangle.

        Returns:
            NDArray[np.float64]: The transformed phi values after the forward transformation.
        """
        phi_out = np.zeros_like(phi)
        logger.debug(f"type of self.corners in phi_to_phi_forward : {type(self.corners)}")
        _phi_to_phi_forward(
            energy=binding_energy,
            phi=phi,
            phi_out=phi_out,
            corners=self.corners,
            rectangle_phis=self.rectangle_phis,
        )
        return phi_out


def trapezoid(
    data: xr.DataArray,
    corners: list[dict[str, float] | float],
    rectangle_phis: list[float] | None = None,
    *,
    from_trapezoid: bool = True,
) -> xr.DataArray:
    r"""Applies the trapezoidal correction in angular units by linearly interpolating slices.

    This function shares some code with standard coordinate conversion routines, such as those
    used for momentum conversion, because it can be viewed as a coordinate conversion between two
    angular coordinate systems: the measured angles and the true angles.

    However, this procedure should be regarded as a correction rather than a conversion. Therefore,
    the convert_to_kspace procedure is not appropriate for use with the ConvertTrapezoidalCorrection
    class.

    This function is specifically designed for the ConvertTrapezoidalCorrection class. The most
    notable difference is that this function preserves the stride of phi (granularity of phi) in
    the corrected data.


           (UL)_____________ (UR)                 +--------+
        ↑      \           /                      |        |
        |       \         /        ⇄              |        |
        eV       \_______/               (L_Rect) +--------+  (R_Rect)
            (LL)          (LR)

                                ----------→ phi
    Args:
        data (xr.DataArray): The xarray instances to perform correction on.
        corners (list[dict [str, float] | float]): The coordinate of the trapezoid corners.
            If it is dict, the key must be both "eV" and "phi", which is used in
            from_trapezoid=True. If it is list, the for corners (LL, UL, LR, UR), which is used in
            from_trapezoid=False (dict arg can be used in the case from_trapezoid=False).
        rectangle_phis (list[float]): the phi value of the rectangle corners.
            (i.e. L_Rect and R_Rect). if not specified (None), use the
            arr.coords["phi"].min().item, and arr.coords["phi"].max().item. Defaults to None.
            As the coords of "eV" (and other coords excepting "phi"), does not change, specifying
            L_Rect and R_Rect is enough.
        from_trapezoid: bool, if True, transpose *to* rectangle. in this case the corners are
            set as those of the trapezoid (left figure).  If False, trapspose *from* rectangle. In
            this case, the corners indicate the points to which the maximum and minimum values
            of eV and phi in the original data are mapped, respectively. Defaults to True.

    Returns:
        xr.DataArray: The corrected data.
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

    if rectangle_phis is None and from_trapezoid:
        rectangle_phis = [trapezoid_corners[1]["phi"], trapezoid_corners[3]["phi"]]
    elif rectangle_phis is None and not from_trapezoid:
        rectangle_phis = [data.coords["phi"].min().item(), data.coords["phi"].max().item()]
    assert isinstance(rectangle_phis, list)

    logger.debug("Determining dimensions.")
    data = data.transpose("eV", "phi", ...)
    converted_dims = data.dims

    converter = ConvertTrapezoidalCorrection(
        arr=data,
        dim_order=converted_dims,
        corners=trapezoid_corners,
        rectangle_phis=rectangle_phis,
    )
    c: dict[str, dict[str, float]] = _corners(trapezoid_corners)

    if from_trapezoid:
        upper_width = c["upper_right"]["phi"] - c["upper_left"]["phi"]
        lower_width = c["lower_right"]["phi"] - c["lower_left"]["phi"]

        phi_resolution = (
            data.G.stride("phi", generic_dim_names=False)
            * (max(rectangle_phis) - min(rectangle_phis))
            / max(upper_width, lower_width)
        )
        phi_bounds = (min(rectangle_phis), max(rectangle_phis))
    else:
        phi_resolution = (
            data.G.stride("phi", generic_dim_names=False)
            * min(
                c["lower_right"]["phi"] - c["lower_left"]["phi"],
                c["upper_right"]["phi"] - c["upper_left"]["phi"],
            )
            / (max(rectangle_phis) - min(rectangle_phis))
        )
        phi_bounds = (
            min(c["upper_left"]["phi"], c["lower_left"]["phi"]) - phi_resolution,
            max(c["upper_right"]["phi"], c["lower_right"]["phi"]) + phi_resolution,
        )
    logger.debug(f"phi_resolution: {phi_resolution}")
    logger.debug(f"phi_bounds: {phi_bounds}")

    converted_coordinates = converter.get_coordinates(
        resolution={"phi": phi_resolution},
        bounds={"phi": phi_bounds},
    )
    transforms = {str(dim): converter.conversion_for(dim) for dim in data.dims}
    if not from_trapezoid:
        transforms["phi"] = converter.phi_to_phi_forward
    result = convert_coordinates(
        arr=data,
        target_coordinates=converted_coordinates,
        coordinate_transform={
            "dims": list(data.dims),
            "transforms": transforms,
        },
    )
    assert isinstance(result, xr.DataArray)
    logger.debug("Reassigning index-like coordinates.")
    return result.assign_attrs(data.attrs)


T = TypeVar("T")


def _is_all_type(corners: list[T], type_: type[T]) -> TypeGuard[list[T]]:
    return all(isinstance(corner, type_) for corner in corners)


def _is_all_floats(corners: list[dict[str, float] | float]) -> TypeGuard[list[float]]:
    return all(isinstance(corner, float) for corner in corners)


def _is_all_dicts(corners: list[dict[str, float] | float]) -> TypeGuard[list[dict[str, float]]]:
    return all(isinstance(corner, dict) for corner in corners)


def _corners_typed_dict(
    corners: list[dict[str, float]],
) -> typed.typeddict.Dict[str, typed.typeddict.Dict[str, float]]:
    normal_dict_corners = _corners(corners)
    inter_dict_type = types.DictType(keyty=types.unicode_type, valty=types.float64)
    typed_dict_corners = typed.Dict.empty(key_type=types.unicode_type, value_type=inter_dict_type)
    for corner_position, coords in normal_dict_corners.items():
        each_corner = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
        for coord_name in ["eV", "phi"]:
            each_corner[coord_name] = coords[coord_name]
        typed_dict_corners[corner_position] = each_corner
    return typed_dict_corners


def _corners(corners: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    lower_left, upper_left, lower_right, upper_right = sorted(
        corners,
        key=operator.itemgetter("phi"),
    )
    lower_left, upper_left = sorted([lower_left, upper_left], key=operator.itemgetter("eV"))
    lower_right, upper_right = sorted([lower_right, upper_right], key=operator.itemgetter("eV"))

    return {
        "lower_left": lower_left,
        "upper_left": upper_left,
        "lower_right": lower_right,
        "upper_right": upper_right,
    }
