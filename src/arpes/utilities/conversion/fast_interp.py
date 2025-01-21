"""Provides extremely fast 2D and 3D linear interpolation.

This is used for momentum conversion in place of the scipy
GridInterpolator where it is possible to do so. It is many many
times faster than the grid interpolator and together with other optimizations
resulted in a 50x improvement in the momentum conversion time for
ARPES data in PyARPES.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import numba
import numpy as np

from arpes.debug import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "Interpolator",
]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@numba.njit
def to_fractional_coordinate(
    coord: float,
    initial: float,
    delta: float,
) -> float:
    return (coord - initial) / delta


@numba.njit
def _i1d(xd: float, c0: float, c1: float) -> float:
    return c0 * (1 - xd) + c1 * xd


@numba.njit
def raw_lin_interpolate_1d(xd: float, c0: float, c1: float) -> float:
    return _i1d(xd, c0, c1)


@numba.njit
def raw_lin_interpolate_2d(  # noqa: PLR0913
    xd: float,
    yd: float,
    c00: float,
    c01: float,
    c10: float,
    c11: float,
) -> float:
    # project to 1D
    c0 = _i1d(xd, c00, c10)
    c1 = _i1d(xd, c01, c11)

    return _i1d(yd, c0, c1)


@numba.njit
def raw_lin_interpolate_3d(  # noqa: PLR0913
    xd: float,
    yd: float,
    zd: float,
    c000: float,
    c001: float,
    c010: float,
    c100: float,
    c011: float,
    c101: float,
    c110: float,
    c111: float,
) -> float:
    # project to 2D
    c00 = _i1d(xd, c000, c100)
    c01 = _i1d(xd, c001, c101)
    c10 = _i1d(xd, c010, c110)
    c11 = _i1d(xd, c011, c111)

    # project to 1D
    c0 = _i1d(yd, c00, c10)
    c1 = _i1d(yd, c01, c11)

    return _i1d(zd, c0, c1)


@numba.njit
def lin_interpolate_3d(  # noqa: PLR0913
    data: NDArray[np.float64],
    ix: int,
    iy: int,
    iz: int,
    ixp: int,
    iyp: int,
    izp: int,
    xd: float,
    yd: float,
    zd: float,
) -> float:
    return raw_lin_interpolate_3d(
        xd,
        yd,
        zd,
        data[ix][iy][iz],
        data[ix][iy][izp],
        data[ix][iyp][iz],
        data[ixp][iy][iz],
        data[ix][iyp][izp],
        data[ixp][iy][izp],
        data[ixp][iyp][iz],
        data[ixp][iyp][izp],
    )


@numba.njit
def lin_interpolate_2d(  # noqa: PLR0913
    data: NDArray[np.float64],
    ix: int,
    iy: int,
    ixp: int,
    iyp: int,
    xd: float,
    yd: float,
) -> float:
    return raw_lin_interpolate_2d(
        xd,
        yd,
        data[ix][iy],
        data[ix][iyp],
        data[ixp][iy],
        data[ixp][iyp],
    )


@numba.njit(parallel=True)
def interpolate_3d(  # noqa: PLR0913
    data: NDArray[np.float64],
    output: NDArray[np.float64],
    lower_corner_x: float,
    lower_corner_y: float,
    lower_corner_z: float,
    delta_x: float,
    delta_y: float,
    delta_z: float,
    shape_x: int,
    shape_y: int,
    shape_z: int,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    fill_value: float = np.nan,
) -> None:
    for i in numba.prange(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]) or np.isnan(z[i]):
            output[i] = fill_value
            continue

        ix = to_fractional_coordinate(x[i], lower_corner_x, delta_x)
        iy = to_fractional_coordinate(y[i], lower_corner_y, delta_y)
        iz = to_fractional_coordinate(z[i], lower_corner_z, delta_z)

        def _is_out_of_bounds(i: tuple[float, float, float], shape: tuple[int, int, int]) -> bool:
            return (
                i[0] < 0
                or i[1] < 0
                or i[0] < 0
                or i[0] >= shape[0]
                or i[1] >= shape[1]
                or i[2] >= shape[2]
            )

        if _is_out_of_bounds((ix, iy, iz), (shape_x, shape_y, shape_z)):
            output[i] = fill_value
            continue

        iix, iiy, iiz = math.floor(ix), math.floor(iy), math.floor(iz)
        iixp, iiyp, iizp = (
            min(iix + 1, shape_x - 1),
            min(iiy + 1, shape_y - 1),
            min(iiz + 1, shape_z - 1),
        )
        xd, yd, zd = ix - iix, iy - iiy, iz - iiz

        output[i] = lin_interpolate_3d(data, iix, iiy, iiz, iixp, iiyp, iizp, xd, yd, zd)


@numba.njit(parallel=True)
def interpolate_2d(  # noqa: PLR0913
    data: NDArray[np.float64],
    output: NDArray[np.float64],
    lower_corner_x: float,
    lower_corner_y: float,
    delta_x: float,
    delta_y: float,
    shape_x: int,
    shape_y: int,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    fill_value: float = np.nan,
) -> None:
    for i in numba.prange(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            output[i] = fill_value
            continue

        ix = to_fractional_coordinate(x[i], lower_corner_x, delta_x)
        iy = to_fractional_coordinate(y[i], lower_corner_y, delta_y)

        if ix < 0 or iy < 0 or ix >= shape_x - 1 or iy >= shape_y - 1:
            output[i] = fill_value
            continue

        iix, iiy = math.floor(ix), math.floor(iy)
        iixp, iiyp = (
            min(iix + 1, shape_x - 1),
            min(iiy + 1, shape_y - 1),
        )
        xd, yd = ix - iix, iy - iiy

        output[i] = lin_interpolate_2d(data, iix, iiy, iixp, iiyp, xd, yd)


@dataclass
class Interpolator:
    """Provides a Pythonic interface to fast gridded linear interpolation.

    More or less a drop-in replacement for scipy's RegularGridInterpolator,
    but much faster at the expense of not supporting any extrapolation.
    """

    lower_corner: list[float]
    delta: list[float]
    shape: list[int]
    data: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Convert data to floating point representation.

        Because we do linear not nearest neighbor interpolation this should be safe
        always.
        """
        self.data = self.data.astype(np.float64, copy=False)

    @classmethod
    def from_arrays(
        cls: type[Interpolator],
        xyz: list[NDArray[np.float64]],
        data: NDArray[np.float64],
    ) -> Interpolator:
        """Initializes the interpreter from a coordinate and data array.

        Args:
            xyz: A list of the coordinate arrays. Should be length 2 or 3
              because we provide 2D and 3D coordinate interpolation.
            data: The value of the interpolated function at the coordinate in `xyz`
        """
        lower_corner = [xi[0] for xi in xyz]
        delta = [xi[1] - xi[0] for xi in xyz]
        shape = [len(xi) for xi in xyz]
        return cls(lower_corner, delta, shape, data)

    def __call__(self, xi: NDArray[np.float64] | list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Performs linear interpolation at the coordinates given by `xi`.

        Whether 2D or 3D interpolation is used depends on the dimensionality of `xi` and
        `self.data` but of course they must match one another.

        Args:
            xi: A list or stacked array of the coordinates. Provides a [d, k] array
              of k points each with d dimensions/indices.

        Returns:
            The interpolated values f(x_i) at each point x_i, as a length k scalar array.
        """
        if isinstance(xi, np.ndarray):
            xi = xi.astype(np.float64, copy=False)
            xi = [xi[:, i] for i in range(self.data.ndim)]
        else:
            xi = [xii.astype(np.float64, copy=False) for xii in xi]
        output = np.zeros_like(xi[0])
        interpolator = {
            3: interpolate_3d,
            2: interpolate_2d,
        }[self.data.ndim]
        interpolator(
            self.data,
            output,
            *self.lower_corner,
            *self.delta,
            *self.shape,
            *xi,
        )
        return output
