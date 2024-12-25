"""Infrastructure code for defining coordinate transforms and momentum conversion."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from arpes.debug import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import MOMENTUM

    from .calibration import DetectorCalibration

__all__ = ["K_SPACE_BORDER", "MOMENTUM_BREAKPOINTS", "CoordinateConverter"]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

K_SPACE_BORDER = 0.02
MOMENTUM_BREAKPOINTS: list[float] = [
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
]


class CoordinateConverter:
    """Infrastructure code to support a new coordinate conversion routine.

    In order to do coordinate conversion from c_i to d_i, we need to give functions
    c_i(d_j), i.e. to implement the inverse transform. This is so that we convert by
    interpolating the function from a regular grid of d_i values back to the original
    data expressed in c_i.

    From this, we can see what responsibilities these conversion classes hold:

    * They need to specify how to calculate c_i(d_j)
    * They need to cache computations so that computations of c_i(d_j) can be performed
      efficiently for different coordinates c_i
    * Because they know how to do the inverse conversion, they need to know how to choose
      reasonable grid bounds for the forward transform, so that this can be handled
      automatically.

    These different roles and how they are accomplished are discussed in detail below.
    """

    def __init__(
        self,
        arr: xr.DataArray,
        dim_order: list[str] | None = None,
        *,
        calibration: DetectorCalibration | None = None,
    ) -> None:
        """Intern the volume so that we can check on things during computation."""
        self.arr = arr
        self.dim_order = dim_order
        self.calibration = calibration
        self.phi: NDArray[np.float64] | None = None

    @staticmethod
    def prep(arr: xr.DataArray) -> None:
        """Perform preprocessing of the array to convert before we start.

        The CoordinateConverter.prep method allows you to pre-compute some transformations
        that are common to the individual coordinate transform methods as an optimization.

        This is useful if you want the conversion methods to have separation of concern,
        but if it is advantageous for them to be able to share a computation of some
        variable. An example of this is in BE-kx-ky conversion, where computing k_p_tot
        is a step in both converting kx and ky, so we would like to avoid doing it twice.

        Of course, you can neglect this function entirely. Another technique is to simple
        cache computations as they arrive. This is the technique that is used in
        ConvertKxKy below
        """
        assert isinstance(arr, xr.DataArray)

    @property
    def is_slit_vertical(self) -> bool:
        """For hemispherical analyzers, whether the slit is vertical or horizontal.

        This is an ARPES specific detail, so this conversion code is not strictly general, but
        a future refactor could just push these details to a subclass.
        """
        # 89 - 91 degrees
        angle_tolerance = 1.0
        angle_unit = self.arr.S.angle_unit
        if angle_unit.startswith(("Deg", "deg")):
            return float(np.abs(self.arr.S.lookup_offset_coord("alpha") - 90.0)) < angle_tolerance
        return np.abs(self.arr.S.lookup_offset_coord("alpha") - np.pi / 2) < np.deg2rad(
            angle_tolerance,
        )

    @staticmethod
    def kspace_to_BE(
        binding_energy: NDArray[np.float64],
        *args: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """The energy conservation equation for ARPES.

        This does not depend on any details of the angular conversion (it's the identity) so we can
        put the conversion code here in the base class.
        """
        if args:
            pass
        return binding_energy

    def conversion_for(
        self,
        dim: Hashable,
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Fetches the method responsible for calculating `dim` from momentum coordinates."""
        assert isinstance(dim, str)
        return self.kspace_to_BE

    def identity_transform(self, axis_name: Hashable, *args: Incomplete) -> NDArray[np.float64]:
        """Just returns the coordinate requested from args.

        Useful if the transform is the identity.
        """
        assert isinstance(self.dim_order, list)
        return args[self.dim_order.index(str(axis_name))]

    def get_coordinates(
        self,
        resolution: dict[MOMENTUM, float] | None = None,
        bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    ) -> dict[Hashable, NDArray[np.float64]]:
        """Calculates the coordinates which should be used in momentum space.

        Args:
            resolution(dict): Represents conversion resolution
                key: momentum name, such as "kp", value: resolution, typical value is 0.001
            bounds(dict, optional): bounds of the momentum coordinates

        Returns: dict[str, NDArray[np.float]
            Object that is to be used the coordinates in the momentum converted data.
            Thus the keys are "kp", "kx", and "eV", but not "phi"
        """
        resolution = resolution if resolution is not None else {}
        bounds = bounds if bounds is not None else {}
        return {k: v.values for k, v in self.arr.coords.items() if k == "eV"}
