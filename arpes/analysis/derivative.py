"""Derivative, curvature, and minimum gradient analysis."""
from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from arpes.provenance import provenance, update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from arpes._typing import DataType

__all__ = (
    "curvature2d",
    "curvature1d",
    "d2_along_axis",
    "d1_along_axis",
    "minimum_gradient",
)

DELTA = Literal[0, 1, -1]


def _nothing_to_array(x: xr.DataArray) -> xr.DataArray:
    """Dummy function for DataArray."""
    return x


def _vector_diff(
    arr: NDArray[np.float_],
    delta: tuple[DELTA, DELTA],
    n: int = 1,
) -> NDArray[np.float_]:
    """Computes finite differences along the vector delta, given as a tuple.

    Using delta = (0, 1) is equivalent to np.diff(..., axis=1), while
    using delta = (1, 0) is equivalent to np.diff(..., axis=0).

    Args:
        arr: The input array
        delta: iterable containing vector to take difference along
        n (int):  number of iteration  # TODO: CHECKME

    Returns:
        The finite differences along the translation vector provided.
    """
    if n == 0:
        return arr
    if n < 0:
        raise ValueError("Order must be non-negative but got " + repr(n))

    slice1: list[slice] | tuple[slice, ...] = [slice(None)] * arr.ndim
    slice2: list[slice] | tuple[slice, ...] = [slice(None)] * arr.ndim
    assert isinstance(slice1, list)
    assert isinstance(slice2, list)
    for dim, delta_val in enumerate(delta):
        if delta_val != 0:
            if delta_val < 0:
                slice2[dim] = slice(-delta_val, None)
                slice1[dim] = slice(None, delta_val)
            else:
                slice1[dim] = slice(delta_val, None)
                slice2[dim] = slice(None, -delta_val)

    slice1, slice2 = tuple(slice1), tuple(slice2)
    assert isinstance(slice1, tuple)
    assert isinstance(slice2, tuple)
    if n > 1:
        return _vector_diff(arr[slice1] - arr[slice2], delta, n - 1)

    return arr[slice1] - arr[slice2]


@update_provenance("Minimum Gradient")
def minimum_gradient(
    data: DataType,
    *,
    smooth_fn: Callable[[xr.DataArray], xr.DataArray] | None = None,
    delta: DELTA = 1,
) -> xr.DataArray:
    """Implements the minimum gradient approach to defining the band in a diffuse spectrum.

    Args:
        data(DataType): ARPES data (xr.DataArray is prefarable)
        smooth_fn(Callable| None): Smoothing function before applying the minimum graident method.
            Define like as:
            def warpped_filter(arr: xr.DataArray):
                return gaussian_filtter_arr(arr, {"eV": 0.05, "phi": np.pi/180})
        delta(DELTA): should not set. Use default 1

    Returns:
        The gradient of the original intensity, which enhances the peak position.
    """
    arr = normalize_to_spectrum(data)
    assert isinstance(arr, xr.DataArray)
    smooth_ = _nothing_to_array if smooth_fn is None else smooth_fn
    arr = smooth_(arr)
    return arr / _gradient_modulus(arr, delta=delta)


@update_provenance("Gradient Modulus")
def _gradient_modulus(data: DataType, *, delta: DELTA = 1) -> xr.DataArray:
    """Helper function for minimum gradient.

    Args:
        data(DataType): 2D data ARPES (or STM?)
        delta(int): Δ value, no need to change in most case.

    Returns: xr.DataArray
        [TODO:description]
    """
    spectrum = normalize_to_spectrum(data)
    assert isinstance(spectrum, xr.DataArray)
    values: NDArray[np.float_] = spectrum.values
    gradient_vector = np.zeros(shape=(8, *values.shape))

    gradient_vector[0, :-delta, :] = _vector_diff(values, (delta, 0))
    gradient_vector[1, :, :-delta] = _vector_diff(values, (0, delta))
    gradient_vector[2, delta:, :] = _vector_diff(values, (-delta, 0))
    gradient_vector[4, :-delta, :-delta] = _vector_diff(values, (delta, delta))
    gradient_vector[3, :, delta:] = _vector_diff(values, (0, -delta))
    gradient_vector[5, :-delta, delta:] = _vector_diff(values, (delta, -delta))
    gradient_vector[6, delta:, :-delta] = _vector_diff(values, (-delta, delta))
    gradient_vector[7, delta:, delta:] = _vector_diff(values, (-delta, -delta))

    data_copy = spectrum.copy(deep=True)
    data_copy.values = np.linalg.norm(gradient_vector, axis=0)
    return data_copy


@update_provenance("Maximum Curvature 1D")
def curvature1d(
    arr: xr.DataArray,
    dim: str = "",
    alpha: float = 0.1,
    smooth_fn: Callable[[xr.DataArray], xr.DataArray] | None = None,
) -> xr.DataArray:
    r"""Provide "1D-Maximum curvature analyais.

    Args:
        arr(xr.DataArray): ARPES data
        dim(str): dimension for maximum curvature
        alpha: regulation parameter, chosen semi-universally, but with
            no particular justification
        smooth_fn (Callable | None): smoothing function. Define like as:
            def warpped_filter(arr: xr.DataArray):
                return gaussian_filtter_arr(arr, {"eV": 0.05, "phi": np.pi/180})

    Returns:
        The curvature of the intensity of the original data.
    """
    assert isinstance(arr, xr.DataArray)
    assert alpha > 0
    if not dim:
        dim = str(arr.dims[0])
    smooth_ = _nothing_to_array if smooth_fn is None else smooth_fn
    arr = smooth_(arr)
    d_arr = arr.differentiate(dim)
    d2_arr = d_arr.differentiate(dim)
    #
    denominator = (alpha * abs(float(d_arr.min().values)) ** 2 + d_arr**2) ** 1.5
    filterd_arr = xr.DataArray(
        (d2_arr / denominator).values,
        arr.coords,
        arr.dims,
        attrs=copy.deepcopy(arr.attrs),
    )

    if "id" in arr.attrs:
        filterd_arr.attrs["id"] = arr.attrs["id"] + "_CV"

        provenance(
            filterd_arr,
            arr,
            {"what": "Maximum Curvature", "by": "1D", "alpha": alpha},
        )
    return filterd_arr


@update_provenance("Maximum Curvature 2D")
def curvature2d(
    arr: xr.DataArray,
    directions: tuple[str, str] = ("phi", "eV"),
    alpha: float = 0.1,
    weight2d: float = 1,
    smooth_fn: Callable[[xr.DataArray], xr.DataArray] | None = None,
) -> xr.DataArray:
    r"""Provide "2D-Maximum curvature analysis".

    Args:
        arr(xr.DataArray): ARPES data
        directions (tuple[str, str]): Dimension for apply the maximum curvature
        alpha: regulation parameter, chosen semi-universally, but with
            no particular justification
        weight2d(float): Weighiting between energy and angle axis.
            if weight2d >> 1, the output is esseitially same as one along "phi"
               (direction[0]) axis.
            if weight2d << 0, the output is essentially same as one along "eV"
               (direction[1])
        smooth_fn (Callable | None): smoothing function. Define like as:
            def warpped_filter(arr: xr.DataArray):
                return gaussian_filtter_arr(arr, {"eV": 0.05, "phi": np.pi/180})

    Returns:
        The curvature of the intensity of the original data.


    It should essentially same as the ``curvature`` function, but the ``weight`` argument is added.
    """
    assert isinstance(arr, xr.DataArray)
    assert alpha > 0
    assert weight2d != 0
    dx, dy = tuple(float(arr.coords[str(d)][1] - arr.coords[str(d)][0]) for d in arr.dims[:2])
    weight = (dx / dy) ** 2
    smooth_ = _nothing_to_array if smooth_fn is None else smooth_fn
    arr = smooth_(arr)
    dfx = arr.differentiate(directions[0])
    dfy = arr.differentiate(directions[1])
    d2fx = dfx.differentiate(directions[0])
    d2fy = dfy.differentiate(directions[1])
    d2fxy = dfx.differentiate(directions[1])
    if weight2d > 0:
        weight *= weight2d
    else:
        weight /= abs(weight2d)
    avg_x = abs(float(dfx.min().values))
    avg_y = abs(float(dfy.min().values))
    avg = max(avg_x**2, weight * avg_y**2)
    numerator = (
        (alpha * avg + weight * dfx * dfx) * d2fy
        - 2 * weight * dfx * dfy * d2fxy
        + weight * (alpha * avg + dfy * dfy) * d2fx
    )
    denominator = (alpha * avg + weight * dfx**2 + dfy**2) ** 1.5
    curv = xr.DataArray((numerator / denominator).values, arr.coords, arr.dims, attrs=arr.attrs)

    if "id" in curv.attrs:
        del curv.attrs["id"]
        provenance(
            curv,
            arr,
            {
                "what": "Curvature",
                "by": "2D_with_weight",
                "directions": directions,
                "alpha": alpha,
                "weight2d": weight2d,
            },
        )
    return curv


def dn_along_axis(
    arr: xr.DataArray,
    dim: str = "",
    smooth_fn: Callable[[xr.DataArray], xr.DataArray] | None = None,
    *,
    order: int = 2,
) -> xr.DataArray:
    """Like curvature, performs a second derivative.

    You can pass a function to use for smoothing through
    the parameter smooth_fn, otherwise no smoothing will be performed.

    You can specify the dimension (by dim) to take the derivative along with the axis param, which
    expects a string. If no axis is provided the axis will be chosen from among the available ones
    according to the preference for axes here, the first available being taken:

    ['eV', 'kp', 'kx', 'kz', 'ky', 'phi', 'beta', 'theta]

    Args:
        arr (xr.DataArray): ARPES data
        dim (str): dimension for derivative
        smooth_fn (Callable | None): smoothing function with DataArray as argument
        order: Specifies how many derivatives to take

    Returns:
        The nth derivative data.
    """
    assert isinstance(arr, xr.DataArray)
    if not dim:
        dim = str(arr.dims[0])
    smooth_ = _nothing_to_array if smooth_fn is None else smooth_fn
    dn_arr = smooth_(arr)
    for _ in range(order):
        dn_arr = dn_arr.differentiate(dim)

    if "id" in dn_arr.attrs:
        dn_arr.attrs["id"] = dn_arr.attrs["id"] + f"_dy{order}"
        provenance(
            dn_arr,
            arr,
            {
                "what": f"{order}th derivative",
                "by": "dn_along_axis",
                "axis": dim,
                "order": order,
            },
        )

    return dn_arr


d2_along_axis = functools.partial(dn_along_axis, order=2)
d1_along_axis = functools.partial(dn_along_axis, order=1)


def curvature(
    arr: xr.DataArray,
    directions: tuple[str, str] = ("phi", "eV"),
    alpha: float = 1,
) -> xr.DataArray:
    r"""Provides "curvature" analysis for band locations.

    Keep it for just compatilitiby

    Defined via

    .. math::

        C(x,y) = \frac{([C_0 + (df/dx)^2]\frac{d^2f}{dy^2} -
        2 \frac{df}{dx}\frac{df}{dy} \frac{d^2f}{dxdy} +
        [C_0 + (\frac{df}{dy})^2]\frac{d^2f}{dx^2})}{
            (C_0 (\frac{df}{dx})^2 + (\frac{df}{dy})^2)^{3/2}}


    of in the case of inequivalent dimensions x and y

    .. math::

        C(x,y) = \frac{[1 + C_x(\frac{df}{dx})^2]C_y
        \frac{d^2f}{dy^2} - 2 C_x  C_y  \frac{df}{dx}\frac{df}{dy}\frac{d^2f}{dxdy} +
        [1 + C_y (\frac{df}{dy})^2] C_x \frac{d^2f}{dx^2}}{
        (1 + C_x (\frac{df}{dx})^2 + C_y (\frac{df}{dy})^2)^{3/2}}

    (Eq. (14) in Rev. Sci. Instrum. 82, 043712 (2011).)

    where



    .. math::

        C_x = C_y (\frac{dx}{dy})^2

    The value of C_y can reasonably be taken to have the value

    .. math::

        (\frac{df}{dx})_\text{max}^2 + \left|\frac{df}{dy}\right|_\text{max}^2
        C_y = (\frac{dy}{dx}) (\left|\frac{df}{dx}\right|_\text{max}^2 +
        \left|\frac{df}{dy}\right|_\text{max}^2) \alpha

    for some dimensionless parameter :math:`\alpha`.

    Args:
        arr(xr.DataArray): ARPES data
        directions (tuple[str, str]): Dimension for apply the maximum curvature
        alpha: regulation parameter, chosen semi-universally, but with
            no particular justification

    Returns:
        The curvature of the intensity of the original data.
    """
    return curvature2d(arr, directions=directions, alpha=alpha, weight2d=1, smooth_fn=None)
