"""Common implementations of peaks, backgrounds for other models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lmfit.lineshapes import lorentzian
from scipy.ndimage import gaussian_filter
from scipy.special import erfc

from arpes.constants import MAX_EXP_ARG

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = (
    "affine_broadened_fd",
    "band_edge_bkg",
    "fermi_dirac",
    "fermi_dirac_affine",
    "gstep",
    "gstepb",
    "gstepb_mult_lorentzian",
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
    x_diff = np.clip((x - center) / width, -MAX_EXP_ARG, MAX_EXP_ARG)

    return scale / (1 + np.exp(x_diff))


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
    x_scaling = x[1] - x[0]
    return np.asarray(
        gaussian_filter(
            fermi_dirac_affine(
                x=x,
                center=center,
                width=width,
                lin_slope=lin_slope,
                const_bkg=const_bkg,
            ),
            sigma=sigma / x_scaling,
        ),
        dtype=np.float64,
    )


def gstepb(  # noqa: PLR0913
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 1,
    erf_amp: float = 1,
    lin_slope: float = 0,
    const_bkg: float = 0,
) -> NDArray[np.float64]:
    """Complementary error function as a approximate of the Fermi function convoled with a Gaussian.

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
    return (const_bkg + lin_slope * (x - center)) * gstep(x, center, width, erf_amp)


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
    return erf_amp * erfc((x - center) / width) / 2


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
    return (
        lorentzian(x, gamma, lor_center, amplitude) + (lin_slope * x + const_bkg)
    ) * fermi_dirac(
        x=x,
        center=center,
        width=width,
    )


def fermi_dirac_affine(
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 0.05,
    lin_slope: float = 0,
    const_bkg: float = 1,
) -> NDArray[np.float64]:
    """Fermi step edge with a linear background above the Fermi level."""
    return (const_bkg + lin_slope * (x - center)) * fermi_dirac(x=x, center=center, width=width)


def gstepb_mult_lorentzian(  # noqa: PLR0913
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 1,
    erf_amp: float = 1,
    lin_slope: float = 0,
    const_bkg: float = 0,
    gamma: float = 1,
    lorcenter: float = 0,
) -> NDArray[np.float64]:
    """A Lorentzian multiplied by a gstepb background."""
    return gstepb(x, center, width, erf_amp, lin_slope, const_bkg) * lorentzian(
        x=x,
        sigma=gamma,
        center=lorcenter,
        amplitude=1,
    )
