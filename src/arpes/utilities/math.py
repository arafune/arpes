"""Math snippets used elsewhere in PyARPES."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import numpy as np

from arpes.constants import K_BOLTZMANN_EV_KELVIN
from arpes.debug import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "fermi_distribution",
    "inv_fermi_distribution",
    "polarization",
]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def polarization(up: NDArray[np.float64], down: NDArray[np.float64]) -> NDArray[np.float64]:
    """The equivalent normalized difference for a two component signal."""
    return (up - down) / (up + down)


def inv_fermi_distribution(
    energy: float,
    temperature: float,
    mu: float = 0.0,
) -> float:
    """Expects energy in eV and temperature in Kelvin."""
    return np.exp((energy - mu) / (K_BOLTZMANN_EV_KELVIN * temperature)) + 1.0


def fermi_distribution(
    energy: float,
    temperature: float,
) -> float:
    """Expects energy in eV and temperature in Kelvin."""
    return 1.0 / inv_fermi_distribution(energy, temperature)
