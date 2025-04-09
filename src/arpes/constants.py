"""Useful constants for experiments and conversions.

Much of this is collected from past students, especially Jeff's 'Cstes.ipf'.

Some of this will disappear in future updates, as we move away from magic constants towards
bundling necessary information on endstation classes.
"""

from __future__ import annotations

import numpy as np

# eV, A reasonablish value if you aren't sure for the particular sample
WORK_FUNCTION = 4.3

METERS_PER_SECOND_PER_EV_ANGSTROM = (
    151927  # converts from eV * angstrom to meters/second velocity units
)
HBAR = 1.0545718176461565e-34
HBAR_PER_EV = 6.582119569509067e-16
# gives the energy lifetime relationship via tau = -hbar / np.imag(self_energy)


BARE_ELECTRON_MASS = 9.109383701e-31  # kg
HBAR_SQ_EV_PER_ELECTRON_MASS = 0.475600805657  # hbar^2 / m0 in eV^2 s^2 / kg
HBAR_SQ_EV_PER_ELECTRON_MASS_ANGSTROM_SQ = 7.619964  # (hbar^2) / (m0 * angstrom ^2) in eV
ELECTRON_CHARGE = 1.60217663e-19

K_BOLTZMANN_EV_KELVIN = 8.617333262145178e-5  # in units of eV / Kelvin
K_BOLTZMANN_MEV_KELVIN = 1000 * K_BOLTZMANN_EV_KELVIN  # meV / Kelvin

HC = 1239.8419843320028  # in units of eV * nm

# Lanzara lab specific
STRAIGHT_TOF_LENGTH = 0.937206
SPIN_TOF_LENGTH = 1.1456
DLD_LENGTH = 1.1456  # This isn't correct but it should be a reasonable guess

K_INV_ANGSTROM = 0.5123167219534328
HV_CONVERSION = 3.814697265625

TWO_DIMENSION = 2


MAX_EXP_ARG = np.log(np.finfo(np.float64).max)
