"""Common implementations of peaks, backgrounds for other models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erfc  # pylint: disable=no-name-in-module

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = (
    "affine_bkg",
    "band_edge_bkg",
    "fermi_dirac",
    "fermi_dirac_affine",
    "gaussian",
    "gstep",
    "gstep_stdev",
    "gstepb",
    "lorentzian",
    "twolorentzian",
)


def affine_bkg(
    x: NDArray[np.float64],
    lin_bkg: float = 0,
    const_bkg: float = 0,
) -> NDArray[np.float64]:
    """An affine/linear background.

    Args:
        x: x-value as independent variable
        lin_bkg: coefficient of linear background
        const_bkg: constant background

    Returns:
        Background of the form
          lin_bkg * x + const_bkg
    """
    return lin_bkg * x + const_bkg


def gaussian(
    x: NDArray[np.float64],
    center: float = 0,
    sigma: float = 1,
    amplitude: float = 1,
) -> NDArray[np.float64]:
    """Some constants are absorbed here into the amplitude factor."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def lorentzian(
    x: NDArray[np.float64],
    gamma: float,
    center: float,
    amplitude: float,
) -> NDArray[np.float64]:
    """A straightforward Lorentzian."""
    return amplitude * (1 / (2 * np.pi)) * gamma / ((x - center) ** 2 + (0.5 * gamma) ** 2)


def fermi_dirac(
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 0.05,
    scale: float = 1,
) -> NDArray[np.float64]:
    """Fermi edge, with somewhat arbitrary normalization."""
    return scale / (np.exp((x - center) / width) + 1)


def gstepb(  # noqa: PLR0913
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 1,
    erf_amp: float = 1,
    lin_bkg: float = 0,
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
        lin_bkg: linear background slope
        const_bkg: constant background

    Returns:
        The step edge.
    """
    dx = x - center
    return const_bkg + lin_bkg * np.min(dx, 0) + gstep(x, center, width, erf_amp)


def gstep(
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 1,
    erf_amp: float = 1,
) -> NDArray[np.float64]:
    """Fermi function convolved with a Gaussian.

    Args:
        x: value to evaluate fit at
        center: center of the step
        width: width of the step
        erf_amp: height of the step

    Returns:
        The step edge.
    """
    dx = x - center
    return erf_amp * 0.5 * erfc(1.66511 * dx / width)


def band_edge_bkg(  # noqa: PLR0913
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 0.05,
    amplitude: float = 1,
    gamma: float = 0.1,
    lor_center: float = 0,
    offset: float = 0,
    lin_bkg: float = 0,
    const_bkg: float = 0,
) -> NDArray[np.float64]:
    """Lorentzian plus affine background multiplied into fermi edge with overall offset."""
    return (lorentzian(x, gamma, lor_center, amplitude) + lin_bkg * x + const_bkg) * fermi_dirac(
        x,
        center,
        width,
    ) + offset


def fermi_dirac_affine(  # noqa: PLR0913
    x: NDArray[np.float64],
    center: float = 0,
    width: float = 0.05,
    lin_bkg: float = 0,
    const_bkg: float = 0,
    scale: float = 1,
) -> NDArray[np.float64]:
    """Fermi step edge with a linear background above the Fermi level."""
    # Fermi edge with an affine background multiplied in
    return (scale + lin_bkg * x) / (np.exp((x - center) / width) + 1) + const_bkg


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


def twolorentzian(  # noqa: PLR0913
    x: NDArray[np.float64],
    gamma: float,
    t_gamma: float,
    center: float,
    t_center: float,
    amp: float,
    t_amp: float,
    lin_bkg: float,
    const_bkg: float,
) -> NDArray[np.float64]:
    """A double lorentzian model.

    **This is typically not necessary, as you can use the + operator on the Model instances.**
    For instance `LorentzianModel() + LorentzianModel(prefix='b')`.

    This mostly exists for people that prefer to do things the "Igor Way".

    Args:
        x: value-x as independent variable
        gamma: lorentzian gamma
        t_gamma: another lorentzian gamma
        center: peak position
        t_center: peak position for another lorenzian
        amp: amplitude
        t_amp: amplitude for another lorenzian
        lin_bkg: coefficient of linear background
        const_bkg: constant background

    Returns:
        A two peak structure.
    """
    L1 = lorentzian(x, gamma, center, amp)
    L2 = lorentzian(x, t_gamma, t_center, t_amp)
    AB = affine_bkg(x, lin_bkg, const_bkg)
    return L1 + L2 + AB
