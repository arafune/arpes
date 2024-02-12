"""Utilities for estimating quantities of interest when using a laser for photoemission."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pint

__all__ = ("electrons_per_pulse",)


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
