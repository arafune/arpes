"""Common implementations of peaks, backgrounds for other models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lmfit.lineshapes import lorentzian
from scipy.ndimage import gaussian_filter
from scipy.special import erfc

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = (
    "affine_broadened_fd",
    "band_edge_bkg",
    "fermi_dirac",
    "fermi_dirac_affine",
    "gstep",
    "gstep_stdev",
    "gstepb",
)


def affine_broadened_fd(  # noqa: PLR0913
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 0.003,
    sigma: float = 0.02,
    const_bkg: float = 1,
    lin_slope: float = 0,
) -> NDArray[np.float64]:
    """Fermi function convoled with a Gaussian together with affine background.

    Args:
        x: value to evaluate function at
        center: center of the step.
        width: width of the step.
        sigma: The gaussian sigma as the convolution width.
        const_bkg: constant background.
        lin_slope: linear (affine) background slope.
    """
    dx = x - center
    x_scaling = x[1] - x[0]
    fermi = 1 / (np.exp(dx / width) + 1)
    return np.asarray(
        gaussian_filter(
            (const_bkg + lin_slope * dx) * fermi,
            sigma=sigma / x_scaling,
        ),
        dtype=np.float64,
    )


def fermi_dirac(
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 0.05,
    scale: float = 1,
) -> NDArray[np.float64]:
    r"""Fermi edge, with somewhat arbitrary normalization.

    :math:`\frac{scale}{\exp\left(\frac{x-center}{width} +1\right)}`
    """
    return scale / (np.exp((x - center) / width) + 1)


def gstepb(  # noqa: PLR0913
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 1,
    erf_amp: float = 1,
    lin_slope: float = 0,
    const_bkg: float = 0,
) -> NDArray[np.float64]:
    """Fermi function convoled with a Gaussian together with affine background.

    This accurately represents low temperature steps where thermal broadening is
    less substantial than instrumental resolution.

    Args:
        x: value to evaluate function at
        center: center of the step
        width: width of the step
        erf_amp: height of the step
        lin_slope: linear background slope
        const_bkg: constant background

    Returns:
        The step edge.
    """
    dx = x - center
    return const_bkg + lin_slope * np.min(dx, 0) + gstep(x, center, width, erf_amp)


def gstep(
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 1,
    erf_amp: float = 1,
) -> NDArray[np.float64]:
    r"""Fermi function convolved with a Gaussian.

    :math:`\frac{erf\_amp}{2} \tims \mathrm{erfc}\left(\frac{(x-center)}{w}\right)

    Args:
        x: value to evaluate fit at
        center: center of the step
        width: width of the step
        erf_amp: height of the step

    Returns:
        The step edge.
    """
    dx = x - center
    return erf_amp * erfc(dx / width) / 2


def band_edge_bkg(  # noqa: PLR0913
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 0.05,
    amplitude: float = 1,
    gamma: float = 0.1,
    lor_center: float = 0,
    lin_slope: float = 0,
    const_bkg: float = 0,
) -> NDArray[np.float64]:
    """Lorentzian plus affine background multiplied into fermi edge with overall offset.

    Todo: Reconsidering the Need.
    """
    return (lorentzian(x, gamma, lor_center, amplitude) + lin_slope * x + const_bkg) * fermi_dirac(
        x,
        center,
        width,
    )


def fermi_dirac_affine(
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 0.05,
    lin_slope: float = 0,
    const_bkg: float = 1,
) -> NDArray[np.float64]:
    """Fermi step edge with a linear background above the Fermi level."""
    return (const_bkg + lin_slope * x) / (np.exp((x - center) / width) + 1)


def gstep_stdev(
    x: NDArray[np.float64],
    center: float = 0,
    sigma: float = 1,
    erf_amp: float = 1,
) -> NDArray[np.float64]:
    """Fermi function convolved with a Gaussian.

    Args:
        x: value to evaluate fit at
        center: center of the step
        sigma: width of the step
        erf_amp: height of the step
    """
    dx = x - center
    return erf_amp * 0.5 * erfc(np.sqrt(2) * dx / sigma)
