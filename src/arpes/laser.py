"""Utilities for estimating quantities of interest when using a laser for photoemission."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

from .constants import HC
from .debug import setup_logger

if TYPE_CHECKING:
    import pint

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

__all__ = ("electrons_per_pulse", "wavelength_to_energy")


def wavelength_to_energy(wavelength_nm: float) -> float:
    """Return Energy of the light.

    Args:
        wavelength_nm (NDArray | float): wavelength of the light in nm unit.

    Returns: NDArray | float
        Photon energy in eV unit.
    """
    return HC / wavelength_nm


def electrons_per_pulse(
    photocurrent: pint.Quantity,
    repetition_rate: pint.Quantity,
    division_ratio: int = 1,
) -> float:
    """Calculates the number of photoemitted electrons per pulse for pulsed lasers.

    Either the pulse_rate or the division_ratio and the base_repetition_rate should be
    specified.

    Args:
        photocurrent: The photocurrent in `pint` current units (i.e. amps).
        repetition_rate: The repetition rate for the laser, in `pint` frequency units.
        division_ratio: The division_ratio for a pulse-picked laser. Optionally modifies the
          repetition rate used for the calculation.

    Returns:
        The expectation of the number of electrons emitted per pulse of the laser.
    """
    repetition_rate /= division_ratio
    eles_per_attocoulomb = 6.2415091
    atto_coulombs = (photocurrent / repetition_rate).to("attocoulomb")
    return (atto_coulombs * eles_per_attocoulomb).magnitude
