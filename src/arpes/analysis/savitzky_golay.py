"""Scipy cookbook implementations of the Savitzky Golay filter for xr.DataArrays."""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.signal
import xarray as xr

from arpes.constants import TWO_DIMENSION
from arpes.provenance import update_provenance

if TYPE_CHECKING:
    from collections.abc import Hashable

    from numpy.typing import NDArray


__all__ = ("savitzky_golay",)


@update_provenance("Savitzky Golay Filter")
def savitzky_golay(  # noqa: PLR0913
    data: xr.DataArray,
    window_size: int,
    order: int,
    deriv: int | Literal["col", "row", "both", None] = 0,
    rate: int = 1,
    dim: Hashable = "",
) -> xr.DataArray:
    """Implements a Savitzky Golay filter with given window size.

    You can specify "pass through" dimensions
    which will not be touched with the `dim` argument. This allows for filtering each frame of a map
    or each equal-energy contour in a 3D dataset, for instance.

    Args:
        data: Input data.
        window_size: Number of points in the window that the filter uses locally.
        order: The polynomial order used in the convolution.
        deriv: the order of the derivative to compute (default = 0 means only smoothing)
        rate : int (?)  default is 1.0
        dim (str): The dimension along which the filter is to be applied

    Returns:
        Smoothed data.
    """
    if isinstance(
        data,
        list | np.ndarray,
    ):
        return savitzky_golay_array(data, window_size, order, deriv, rate)

    if len(data.dims) == 1:
        transformed_data = savitzky_golay_array(data.values, window_size, order, deriv, rate)
    else:
        # only 1D, 2D, 3D supported for the moment
        assert len(data.dims) <= 3  # noqa: PLR2004

        if deriv == 0:
            deriv = None

        if len(data.dims) == TWO_DIMENSION + 1:
            if not dim:
                dim = data.dims[-1]
            return data.G.map_axes(
                dim,
                lambda d, _: savitzky_golay(
                    d,
                    window_size,
                    order,
                    deriv,
                    rate,
                ),
            )

        if len(data.dims) == TWO_DIMENSION:
            if not dim:
                transformed_data = savitzky_golay_2d(
                    data.values,
                    window_size,
                    order,
                    derivative=deriv,
                )
            else:
                return data.G.map_axes(
                    dim,
                    lambda d, _: savitzky_golay(
                        d,
                        window_size,
                        order,
                        deriv=deriv or 0,
                        rate=rate,
                        dim=None,
                    ),
                )

    return xr.DataArray(
        transformed_data,
        data.coords,
        data.dims,
        attrs=data.attrs.copy(),
    )


def savitzky_golay_2d(
    input_arr: NDArray[np.float_],
    window_size: int,
    order: int,
    derivative: Literal[None, "col", "row", "both"] = None,
) -> NDArray[np.float_]:
    """Implementation from the scipy cookbook before the Savit.

    This is changed now, so we should ideally migrate to use the new scipy implementation.

    Args:
        input_arr: Input 2D array, NDArray is generally assumed.
        window_size (int): window of Savitzky-Golay filter
        order (int): order of Savitzky-Golay filter
        derivative:  determine the convolution method


    Returns:
        Smoothed data
    """
    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    assert window_size % 2 == 1, "window_size size must be a positive odd number (>3)"
    assert window_size > 1, "window_size size must be a positive odd number (>3)"

    if window_size**2 < n_terms:
        msg = "order is too high for the window size"
        raise ValueError(msg)

    half_size = window_size // 2

    # exponents of the polynomial.
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(
        window_size**2,
    )

    # build matrix of system of equation
    A = np.empty((window_size**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = input_arr.shape[0] + 2 * half_size, input_arr.shape[1] + 2 * half_size
    Z = np.zeros(new_shape)
    # top band
    band = input_arr[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(
        np.flipud(input_arr[1 : half_size + 1, :]) - band,
    )
    # bottom band
    band = input_arr[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(
        np.flipud(input_arr[-half_size - 1 : -1, :]) - band,
    )
    # left band
    band = np.tile(input_arr[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(
        np.fliplr(input_arr[:, 1 : half_size + 1]) - band,
    )
    # right band
    band = np.tile(input_arr[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(
        np.fliplr(input_arr[:, -half_size - 1 : -1]) - band,
    )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = input_arr

    # top left corner
    band = input_arr[0, 0]
    Z[:half_size, :half_size] = band - np.abs(
        np.flipud(np.fliplr(input_arr[1 : half_size + 1, 1 : half_size + 1])) - band,
    )
    # bottom right corner
    band = input_arr[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(
        np.flipud(np.fliplr(input_arr[-half_size - 1 : -1, -half_size - 1 : -1])) - band,
    )

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(
        np.flipud(Z[half_size + 1 : 2 * half_size + 1, -half_size:]) - band,
    )
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(
        np.fliplr(Z[-half_size:, half_size + 1 : 2 * half_size + 1]) - band,
    )

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, m, mode="valid")
    if derivative == "col":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -c, mode="valid")
    if derivative == "row":
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode="valid")
    if derivative == "both":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(Z, -r, mode="valid"), scipy.signal.fftconvolve(
            Z,
            -c,
            mode="valid",
        )
    msg = """Need coorect setting about "derivative" (None, "col", "row", "both")"""
    raise RuntimeError(msg)


def savitzky_golay_array(
    y: NDArray[np.float_],
    window_size: int = 3,
    order: int = 1,
    deriv: int = 0,
    rate: int = 1,
) -> NDArray[np.float_]:
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Notes:
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.

    Examples:
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()

    References:
        * [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
          Data by Simplified Least Squares Procedures. Analytical
          Chemistry, 1964, 36 (8), pp 1627-1639.
        * [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
          W.H. Press, S.A. Teukolsky, W.G. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688

    Args:
        y : array_like, shape (N,)
          the values of the time history of the signal.
        window_size : int
          the length of the window. Must be an odd integer number.
        order : int
          the order of the polynomial used in the filtering.
          Must be less then `window_size` - 1.
        deriv: int
          the order of the derivative to compute (default = 0 means only smoothing)
        rate: int

    Returns:
        ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    """
    assert window_size % 2 == 1, "window_size size must be a positive odd number"
    assert window_size > 1, "window_size size must be a positive odd number and > 1"
    assert order < window_size - 1, "window_size is too small for the polynomials order"

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")
