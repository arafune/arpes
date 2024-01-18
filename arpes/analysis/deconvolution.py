"""Provides deconvolution implementations, especially for 2D Richardson-Lucy."""
from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy
import scipy.ndimage
import xarray as xr
from tqdm.notebook import tqdm

from arpes.fits.fit_models.functional_forms import gaussian
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import DataType

__all__ = (
    "deconvolve_ice",
    "deconvolve_rl",
    "make_psf1d",
)

TWODIMWENSION = 2
THREEDIMENSION = 3


@update_provenance("Approximate Iterative Deconvolution")
def deconvolve_ice(
    data: DataType,
    psf: NDArray[np.float_],
    n_iterations: int = 5,
    deg: int | None = None,
) -> DataType:
    """Deconvolves data by a given point spread function (PSF).

    The iterative convolution extrapolation method is used.
    The PSF is the impulse response of a focused optical imaging system.

    Args:
        data: input data
        psf(NDArray[np.float_): array as point spread function
        n_iterations: the number of convolutions to use for the fit
        deg: the degree of the fitting polynominial

    Returns:
        The deconvoled data in the same format.
    """
    arr = normalize_to_spectrum(data)
    if type(data) is np.ndarray:
        pass
    else:
        arr = arr.values

    if deg is None:
        deg = n_iterations - 3
    iteration_steps = list(range(1, n_iterations + 1))

    iteration_list = [arr]

    for _ in range(n_iterations - 1):
        iteration_list.append(scipy.ndimage.convolve(iteration_list[-1], psf))
    iteration_list = np.asarray(iteration_list)

    deconv = arr * 0
    for t, series in enumerate(iteration_list.T):
        coefs = np.polyfit(iteration_steps, series, deg=deg)
        poly = np.poly1d(coefs)
        deconv[t] = poly(0)

    if type(data) is np.ndarray:
        result = deconv
    else:
        result = normalize_to_spectrum(data).copy(deep=True)
        result.values = deconv
    return result


@update_provenance("Lucy Richardson Deconvolution")
def deconvolve_rl(
    data: DataType,
    psf: xr.DataArray | None = None,
    n_iterations: int = 10,
    axis: str = "",
    sigma: float = 0,
    mode: Literal["reflect", "constant", "nearest", "mirror", "wrap"] = "reflect",
    *,
    progress: bool = True,
) -> xr.DataArray:
    """Deconvolves data by a given point spread function using the Richardson-Lucy (RL) method.

    Args:
        data: input data
        axis
        sigma
        mode: pass to ndimage.convolve
        progress
        psf: for 1d, if not specified, must specify axis and sigma
        n_iterations: the number of convolutions to use for the fit

    Returns:
        The Richardson-Lucy deconvolved data.
    """
    arr = normalize_to_spectrum(data)

    if psf is None and axis != "" and sigma != 0:
        # if no psf is provided and we have the information to make a 1d one
        # note: this assumes gaussian psf
        psf = make_psf1d(data=arr, dim=axis, sigma=sigma)

    if len(data.dims) > 1:
        if not axis:
            # perform one-dimensional deconvolution of multidimensional data

            # support for progress bars
            def wrap_progress(
                x: Iterable[int],
                *args: Incomplete,
                **kwargs: Incomplete,
            ) -> Iterable[int]:
                if args:
                    for arg in args:
                        warnings.warn(
                            f"unused args is set in deconvolution.py/wrap_progress: {arg}",
                            stacklevel=2,
                        )
                if kwargs:
                    for k, v in kwargs.items():
                        warnings.warn(
                            f"unused args is set in deconvolution.py/wrap_progress: {k}: {v}",
                            stacklevel=2,
                        )
                return x

            if progress:
                wrap_progress = tqdm

            # dimensions over which to iterate
            other_dim = list(data.dims)
            other_dim.remove(axis)

            if len(other_dim) == 1:
                # two-dimensional data
                other_dim = other_dim[0]
                result = arr.copy(deep=True).transpose(
                    other_dim,
                    axis,
                )
                # not sure why the dims only seems to work in this order.
                # seems like I should be able to swap it to (axis,other_dim)
                # and also change the data collection to result[x_ind,y_ind],
                # but this gave different results

                for i, (_, iteration) in wrap_progress(
                    enumerate(arr.G.iterate_axis(other_dim)),
                    desc="Iterating " + other_dim,
                    total=len(arr[other_dim]),
                ):  # TODO: tidy this gross-looking loop
                    # indices of data being deconvolved
                    x_ind = xr.DataArray(list(range(len(arr[axis]))), dims=[axis])
                    y_ind = xr.DataArray([i] * len(x_ind), dims=[other_dim])
                    # perform deconvolution on this one-dimensional piece
                    deconv = deconvolve_rl(
                        data=iteration,
                        psf=psf,
                        n_iterations=n_iterations,
                        axis="",
                        mode=mode,
                    )
                    # build results out of these pieces
                    result[y_ind, x_ind] = deconv.values
            elif len(other_dim) == TWODIMWENSION:
                # three-dimensional data
                result = arr.copy(deep=True).transpose(*other_dim, axis)
                # not sure why the dims only seems to work in this order.
                # eems like I should be able to swap it to (axis,*other_dim) and also change the
                # data collection to result[x_ind,y_ind,z_ind], but this gave different results
                for i, (_od0, iteration0) in wrap_progress(
                    enumerate(arr.G.iterate_axis(other_dim[0])),
                    desc="Iterating " + str(other_dim[0]),
                    total=len(arr[other_dim[0]]),
                ):  # TODO: tidy this gross-looking loop
                    for j, (_od1, iteration1) in wrap_progress(
                        enumerate(iteration0.G.iterate_axis(other_dim[1])),
                        desc="Iterating " + str(other_dim[1]),
                        total=len(arr[other_dim[1]]),
                        leave=False,
                    ):  # TODO:  tidy this gross-looking loop
                        # indices of data being deconvolved
                        x_ind = xr.DataArray(list(range(len(arr[axis]))), dims=[axis])
                        y_ind = xr.DataArray([i] * len(x_ind), dims=[other_dim[0]])
                        z_ind = xr.DataArray([j] * len(x_ind), dims=[other_dim[1]])
                        # perform deconvolution on this one-dimensional piece
                        deconv = deconvolve_rl(
                            data=iteration1,
                            psf=psf,
                            n_iterations=n_iterations,
                            axis="",
                            mode=mode,
                        )
                        # build results out of these pieces
                        result[y_ind, z_ind, x_ind] = deconv.values
            elif len(other_dim) >= THREEDIMENSION:
                # four- or higher-dimensional data
                # TODO:  find way to compactify the different dimensionalities rather than having
                # separate code
                msg = "high-dimensional data not yet supported"
                raise NotImplementedError(msg)
        elif not axis:
            # crude attempt to perform multidimensional deconvolution.
            # not clear if this is currently working
            # TODO: may be able to do this as a sequence of one-dimensional deconvolutions, assuming
            # that the psf is separable (which I think it should be, if we assume it is a
            # multivariate gaussian with principle axes aligned with the dimensions)
            msg = "multi-dimensional convolutions not yet supported"
            raise NotImplementedError(msg)

            if not isinstance(arr, np.ndarray):
                arr = arr.values

            u = [arr]

            for i in range(n_iterations):
                c = scipy.ndimage.convolve(u[-1], psf, mode=mode)
                u.append(u[-1] * scipy.ndimage.convolve(arr / c, np.flip(psf, None), mode=mode))
                # careful about which axis (axes) to flip here...!
                # need to explicitly specify for some versions of numpy

            result = u[-1]
    else:
        if type(arr) is not np.ndarray:
            arr = arr.values
        u = [arr]
        for _ in range(n_iterations):
            c = scipy.ndimage.convolve(u[-1], psf, mode=mode)
            u.append(u[-1] * scipy.ndimage.convolve(arr / c, np.flip(psf, 0), mode=mode))
            # not yet tested to ensure flip correct for asymmetric psf
            # note: need to explicitly specify axis number in np.flip in lower versions of numpy
        if type(data) is np.ndarray:
            result = u[-1].copy()
        else:
            result = normalize_to_spectrum(data).copy(deep=True)
            result.values = u[-1]
    with contextlib.suppress(Exception):
        return result.transpose(*arr.dims)


@update_provenance("Make 1D-Point Spread Function")
def make_psf1d(data: DataType, dim: str, sigma: float) -> xr.DataArray:
    """Produces a 1-dimensional gaussian point spread function for use in deconvolve_rl.

    Args:
        data (DataType): xarray object
        dim (str): dimension name
        sigma (float): sigma value

    Returns:
        A one dimensional point spread array.
    """
    arr = normalize_to_spectrum(data)
    psf = arr.copy(deep=True) * 0 + 1
    other_dims = list(arr.dims)
    other_dims.remove(dim)
    for od in other_dims:
        psf = psf[{od: 0}]
    return psf * gaussian(psf.coords[dim], np.mean(psf.coords[dim]), sigma)


@update_provenance("Make Point Spread Function")
def make_psf(data: DataType, sigmas: dict[str, float]) -> xr.DataArray:
    """Produces an n-dimensional gaussian point spread function for use in deconvolve_rl.

    Not yet operational.

    Args:
        data (DataType): input data
        sigmas (dict[str, float]): sigma values for each dimension.

    Returns:
        The PSF to use.
    """
    raise NotImplementedError

    arr = normalize_to_spectrum(data)
    dims = arr.dims

    psf = arr.copy(deep=True) * 0 + 1

    for dim in dims:
        other_dims = list(arr.dims)
        other_dims.remove(dim)

        psf1d = arr.copy(deep=True) * 0 + 1
        for od in other_dims:
            psf1d = psf1d[{od: 0}]

        if sigmas[dim] == 0:
            # TODO: may need to do subpixel correction for when the dimension has an even length
            psf1d = psf1d * 0
            psf1d[{dim: len(psf1d.coords[dim]) / 2}] = 1
        else:
            psf1d = psf1d * gaussian(psf1d.coords[dim], np.mean(psf1d.coords[dim]), sigmas[dim])

        psf = psf * psf1d
    return psf
