"""Derivative, curvature, and minimum gradient analysis."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import xarray as xr

from arpes.provenance import Provenance, provenance, update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from numpy.typing import NDArray


__all__ = (
    "curvature1d",
    "curvature2d",
    "d1_along_axis",
    "d2_along_axis",
    "minimum_gradient",
)

DELTA = Literal[0, 1, -1]


class D(NamedTuple):
    x: xr.DataArray
    y: xr.DataArray


class D2(NamedTuple):
    x: xr.DataArray
    y: xr.DataArray
    xy: xr.DataArray


def _nothing_to_array(x: xr.DataArray) -> xr.DataArray:
    """Dummy function for DataArray."""
    return x


def _vector_diff(
    arr: NDArray[np.float64],
    delta: tuple[DELTA, DELTA],
    n: int = 1,
) -> NDArray[np.float64]:
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
    data: xr.DataArray,
    *,
    smooth_fn: Callable[[xr.DataArray], xr.DataArray] | None = None,
    delta: DELTA = 1,
) -> xr.DataArray:
    """Implements the minimum gradient approach to defining the band in a diffuse spectrum.

    Args:
        data(DataType): ARPES data (xr.DataArray is prefarable)
        smooth_fn(Callable| None): Smoothing function before applying the minimum graident method.
            Define like:

            .. code-block:: python

                def warpped_filter(arr: xr.DataArray):
                    return gaussian_filtter_arr(arr, {"eV": 0.05, "phi": np.pi/180})

        delta(DELTA): should not set. Use default 1

    Returns:
        The gradient of the original intensity, which enhances the peak position.
    """
    arr = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(arr, xr.DataArray)
    smooth_ = _nothing_to_array if smooth_fn is None else smooth_fn
    arr = smooth_(arr)
    arr = arr.assign_attrs(data.attrs)
    return arr / _gradient_modulus(arr, delta=delta)


@update_provenance("Gradient Modulus")
def _gradient_modulus(
    data: xr.DataArray,
    *,
    delta: DELTA = 1,
) -> xr.DataArray:
    """Helper function for minimum gradient.

    Args:
        data(DataType): 2D data ARPES (or STM?)
        delta(int): Δ value, no need to change in most case.

    Returns: xr.DataArray
    """
    spectrum = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(spectrum, xr.DataArray)
    values: NDArray[np.float64] = spectrum.values
    gradient_vector = np.zeros(shape=(8, *values.shape))

    gradient_vector[0, :-delta, :] = _vector_diff(values, (delta, 0))
    gradient_vector[1, :, :-delta] = _vector_diff(values, (0, delta))
    gradient_vector[2, delta:, :] = _vector_diff(values, (-delta, 0))
    gradient_vector[4, :-delta, :-delta] = _vector_diff(values, (delta, delta))
    gradient_vector[3, :, delta:] = _vector_diff(values, (0, -delta))
    gradient_vector[5, :-delta, delta:] = _vector_diff(values, (delta, -delta))
    gradient_vector[6, delta:, :-delta] = _vector_diff(values, (-delta, delta))
    gradient_vector[7, delta:, delta:] = _vector_diff(values, (-delta, -delta))

    return spectrum.G.with_values(np.linalg.norm(gradient_vector, axis=0))


@update_provenance("Maximum Curvature 1D")
def curvature1d(
    arr: xr.DataArray,
    dim: Hashable = "",
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
                return gaussian_filtter_arr(arr, {"eV": 0.05, "phi": np.pi/180}, repeat_n=5)

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
    denominator = (alpha * abs(float(d_arr.min().values)) ** 2 + d_arr**2) ** 1.5
    filterd_arr = arr.G.with_values((d2_arr / denominator).values)

    if "id" in arr.attrs:
        filterd_arr.attrs["id"] = str(arr.attrs["id"]) + "_CV"
        provenance_context: Provenance = {
            "what": "Maximum Curvature",
            "by": "1D",
            "alpha": alpha,
        }
        provenance(filterd_arr, arr, provenance_context)
    return filterd_arr


@update_provenance("Maximum Curvature 2D")
def curvature2d(
    arr: xr.DataArray,
    dims: tuple[Hashable, Hashable] = ("phi", "eV"),
    alpha: float = 0.1,
    weight2d: float = 1,
    smooth_fn: Callable[[xr.DataArray], xr.DataArray] | None = None,
) -> xr.DataArray:
    r"""Provide "2D-Maximum curvature analysis".

    Args:
        arr(xr.DataArray): ARPES data
        dims (tuple[str, str]): Dimension for apply the maximum curvature
        alpha: regulation parameter, chosen semi-universally, but with
            no particular justification
        weight2d(float): Weighiting between energy and angle axis.
            if weight2d >> 1, the output is esseitially same as one along "phi"
               (direction[0]) axis.
            if weight2d << 0, the output is essentially same as one along "eV"
               (direction[1])
        smooth_fn (Callable | None): smoothing function. Define like as:
            def warpped_filter(arr: xr.DataArray):
                return gaussian_filtter_arr(arr, {"eV": 0.05, "phi": np.pi/180}, repeat_n=5)

    Returns:
        The curvature of the intensity of the original data.


    It should essentially same as the ``curvature`` function, but the ``weight`` argument is added.
    """
    assert isinstance(arr, xr.DataArray)
    assert alpha > 0
    assert weight2d != 0
    dx, dy = tuple(float(arr.coords[str(d)][1] - arr.coords[str(d)][0]) for d in arr.dims[:2])
    weight = (dx / dy) ** 2
    if smooth_fn is not None:
        arr = smooth_fn(arr)
    df = D(x=arr.differentiate(dims[0]), y=arr.differentiate(dims[1]))
    d2f: D2 = D2(
        x=df.x.differentiate(dims[0]),
        y=df.y.differentiate(dims[1]),
        xy=df.x.differentiate(dims[1]),
    )
    if weight2d > 0:
        weight *= weight2d
    else:
        weight /= abs(weight2d)
    avg_x = abs(float(df.x.min().values))
    avg_y = abs(float(df.y.min().values))
    avg = max(avg_x**2, weight * avg_y**2)
    numerator = (
        (alpha * avg + weight * df.x * df.x) * d2f.y
        - 2 * weight * df.x * df.y * d2f.xy
        + weight * (alpha * avg + df.y * df.y) * d2f.x
    )
    denominator = (alpha * avg + weight * df.x**2 + df.y**2) ** 1.5
    curv = arr.G.with_values((numerator / denominator).values)

    if "id" in curv.attrs:
        del curv.attrs["id"]
        provenance_context: Provenance = {
            "what": "Curvature",
            "by": "2D_with_weight",
            "dims": dims,
            "alpha": alpha,
            "weight2d": weight2d,
        }

        provenance(curv, arr, provenance_context)
    return curv


@update_provenance("Derivative")
def dn_along_axis(
    arr: xr.DataArray,
    dim: Hashable = "",
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
        dim (str): dimension for derivative (default is the first item in dims)
        smooth_fn (Callable | None): smoothing function. Define like as:

            def warpped_filter(arr: xr.DataArray):
                return gaussian_filtter_arr(arr, {"eV": 0.05, "phi": np.pi/180}, repeat_n=5)

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
    dn_arr = dn_arr.assign_attrs(arr.attrs)

    if "id" in dn_arr.attrs:
        dn_arr.attrs["id"] = str(dn_arr.attrs["id"]) + f"_dy{order}"
        provenance_context: Provenance = {
            "what": f"{order}th derivative",
            "by": "dn_along_axis",
            "axis": dim,
            "order": order,
        }

        provenance(
            dn_arr,
            arr,
            provenance_context,
        )

    return dn_arr


d2_along_axis = functools.partial(dn_along_axis, order=2)
d1_along_axis = functools.partial(dn_along_axis, order=1)


def curvature(
    arr: xr.DataArray,
    dims: tuple[str, str] = ("phi", "eV"),
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


    of in the case of inequivalent dimensions :math:`x` and :math:`y`

    .. math::

        C(x,y) = \frac{[1 + C_x(\frac{df}{dx})^2]C_y
        \frac{d^2f}{dy^2} - 2 C_x  C_y  \frac{df}{dx}\frac{df}{dy}\frac{d^2f}{dxdy} +
        [1 + C_y (\frac{df}{dy})^2] C_x \frac{d^2f}{dx^2}}{
        (1 + C_x (\frac{df}{dx})^2 + C_y (\frac{df}{dy})^2)^{3/2}}

    (Eq. (14) in Rev. Sci. Instrum. 82, 043712 (2011).)

    where



    .. math::

        C_x = C_y (\frac{dx}{dy})^2

    The value of :math:`C_y`` can reasonably be taken to have the value

    .. math::

        (\frac{df}{dx})_\text{max}^2 + \left|\frac{df}{dy}\right|_\text{max}^2
        C_y = (\frac{dy}{dx}) (\left|\frac{df}{dx}\right|_\text{max}^2 +
        \left|\frac{df}{dy}\right|_\text{max}^2) \alpha

    for some dimensionless parameter :math:`\alpha`.

    Args:
        arr (xr.DataArray): ARPES data
        dims (tuple[str, str]): Dimension for apply the maximum curvature
        alpha (float): regulation parameter, chosen semi-universally, but with
            no particular justification

    Returns:
        The curvature of the intensity of the original data.
    """
    return curvature2d(
        arr,
        dims=dims,
        alpha=alpha,
        weight2d=1,
        smooth_fn=None,
    )
