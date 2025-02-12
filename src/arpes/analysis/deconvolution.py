"""Provides deconvolution implementations, especially for 2D Richardson-Lucy."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import numpy as np
import scipy
import scipy.ndimage
import xarray as xr
from scipy.stats import multivariate_normal
from skimage.restoration import richardson_lucy

from arpes.debug import setup_logger
from arpes.fits.fit_models.functional_forms import gaussian
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from collections.abc import Hashable

    from numpy.typing import NDArray


__all__ = (
    "deconvolve_ice",
    "deconvolve_rl",
    "make_psf",
    "make_psf1d",
)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@update_provenance("Approximate Iterative Deconvolution")
def deconvolve_ice(
    data: xr.DataArray,
    psf: NDArray[np.float64],
    n_iterations: int = 5,
    deg: int | None = None,
) -> xr.DataArray | NDArray[np.float64]:
    """Deconvolves data by a given point spread function (PSF).

    The iterative convolution extrapolation method is used.
    The PSF is the impulse response of a focused optical imaging system.

    Args:
        data (xr.DataArray): input data
        psf: array as point spread function
        n_iterations (int): the number of convolutions to use for the fit
        deg (float): the degree of the fitting polynominial

    Returns:
        The deconvoled data in the same format.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    arr: NDArray[np.float64] = data.values
    if deg is None:
        deg = n_iterations - 3
    iteration_steps = list(range(1, n_iterations + 1))

    iteration_list = [arr]

    for _ in range(n_iterations - 1):
        iteration_list.append(scipy.ndimage.convolve(iteration_list[-1], psf))
    iteration_array = np.asarray(iteration_list)

    deconv = arr * 0
    for t, series in enumerate(iteration_array.T):
        coefs = np.polyfit(iteration_steps, series, deg=deg)
        poly = np.poly1d(coefs)
        deconv[t] = poly(0)

    return data.G.with_values(deconv, keep_attrs=True)


@update_provenance("Lucy Richardson Deconvolution")
def deconvolve_rl(
    data: xr.DataArray,
    psf: xr.DataArray,
    n_iterations: int = 10,
) -> xr.DataArray:
    """Deconvolves data by a given point spread function using the Richardson-Lucy (RL) method.

    Args:
        data: input data
        psf:  The point spread function.
        n_iterations: the number of convolutions to use for the fit

    Returns:
        The Richardson-Lucy deconvolved data.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    im_deconv = richardson_lucy(
        image=data.values,
        psf=psf.values,
        num_iter=n_iterations,
        filter_epsilon=None,
    )
    return data.G.with_values(im_deconv, keep_attrs=True)


@update_provenance("Make 1D-Point Spread Function")
def make_psf1d(
    data: xr.DataArray,
    dim: str,
    sigma: float,
) -> xr.DataArray:
    """Produces a 1-dimensional gaussian point spread function for use in deconvolve_rl.

    Args:
        data (DataType): xarray object
        dim (str): dimension name
        sigma (float): sigma value

    Returns:
        A one dimensional point spread array.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    psf = data.G.with_values(np.zeros_like(data.values) + 1)
    other_dims = list(data.dims)
    other_dims.remove(dim)
    for od in other_dims:
        psf = psf[{od: 0}]
    return psf * gaussian(x=psf.coords[dim], center=np.mean(psf.coords[dim]), sigma=sigma)


@update_provenance("Make Point Spread Function")
def make_psf(
    data: xr.DataArray,
    sigmas: dict[Hashable, float],
    *,
    fwhm: bool = True,
    clip: float | None = None,
) -> xr.DataArray:
    """Produces an n-dimensional gaussian point spread function for use in deconvolve_rl.

    Args:
        data (DataType): input data
        sigmas (dict[str, float]): sigma values for each dimension.
        fwhm (bool): if True, sigma is FWHM, not the standard deviation.
        clip (float | bool): clip the region by sigma-unit.

    Returns:
        The PSF to use.
    """
    strides = data.G.stride(generic_dim_names=False)
    logger.debug(f"strides: {strides}")
    assert set(strides) == set(sigmas)
    pixels: dict[Hashable, int] = dict(
        zip(
            data.dims,
            [i - 1 if i % 2 == 0 else i for i in data.shape],
            strict=True,
        ),
    )

    if fwhm:
        sigmas = {k: v / (2 * np.sqrt(2 * np.log(2))) for k, v in sigmas.items()}
    cov: NDArray[np.float64] = np.zeros((len(sigmas), len(sigmas)), dtype=np.float64)
    for i, dim in enumerate(data.dims):
        cov[i][i] = sigmas[dim] ** 2  # sigma is deviation, but multivariate_normal uses covariant
    logger.debug(f"cov: {cov}")

    psf_coords: dict[Hashable, NDArray[np.float64]] = {}
    for k in data.dims:
        psf_coords[str(k)] = np.linspace(
            -(pixels[str(k)] - 1) / 2 * strides[str(k)],
            (pixels[str(k)] - 1) / 2 * strides[str(k)],
            pixels[str(k)],
        )
    if LOGLEVEL == DEBUG:
        for k, v in psf_coords.items():
            logger.debug(
                f" psf_coords[{k}]: ±{np.max(v):.3f}",
            )
    coords = np.meshgrid(*[psf_coords[dim] for dim in data.dims], indexing="ij")
    coords_for_pdf_pos = np.stack(coords, axis=-1)  # point distribution function (pdf)
    logger.debug(f"shape of coords_for_pdf_pos: {coords_for_pdf_pos.shape}")
    psf = xr.DataArray(
        data=multivariate_normal(mean=np.zeros(len(sigmas)), cov=cov).pdf(
            x=coords_for_pdf_pos,
        ),
        dims=data.dims,
        coords=psf_coords,
        name="PSF",
    )
    if clip:
        clipping_region = {k: slice(-clip * v, clip * v) for k, v in sigmas.items()}
        return psf.sel(indexers=clipping_region)
    return psf
