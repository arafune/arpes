r"""Contains models and utilities for "curve fitting" the forward momentum conversion function.

This allows for rapid fine tuning of the angle offsets used in momentum conversion.

If the user calculates, places, or curve fits for the high symmetry
locations in the data $H_i(\phi,\psi,\theta,\beta,\chi)$, these can be used
as waypoints to find a set of $\Delta\phi$, $\Delta\theta$, $\Delta\chi$, etc.
that minimize

$$
\sum_i \text{min}_j |text{P}(H_i, \Delta\phi, \Delta\theta, \ldots) - S_j|^2
$$

where $S_j$ enumerates the high symmetry points of the known Brillouin zone, and $\text{P}$
is the function that maps forwards from angle to momenta. This can also
be used to calculate moiré information, but using the (many) available
high symmetry points of the moiré superlattice to get a finer estimate of
relative angle alignment, lattice incommensuration, and strain than is possible
using the constituent lattices and Brillouin zones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arpes.constants import TWO_DIMENSION
from arpes.utilities.conversion.forward import convert_coordinates

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arpes._typing import DataType


def k_points_residual(
    coords_dataset: DataType,
    dimensionality: int = 2,
) -> NDArray[np.float_]:
    """[TODO:summary].

    Args:
        coords_dataset: [TODO:description]
        dimensionality: [TODO:description]

    Returns:
        [TODO:description]
    """
    momentum_coordinates = convert_coordinates(coords_dataset)
    if dimensionality == TWO_DIMENSION:
        return np.asarray(
            [
                np.diagonal(momentum_coordinates.kx.values),
                np.diagonal(momentum_coordinates.ky.values),
            ],
        )
    return np.asarray(
        [
            np.diagonal(momentum_coordinates.kx.values),
            np.diagonal(momentum_coordinates.ky.values),
            np.diagonal(momentum_coordinates.kz.values),
        ],
    )


"""
def minimum_forward_error(
    coordinate_samples,
    phi_offset=0,
    psi_offset=0,
    theta_offset=0,
    beta_offset=0,
    chi_offset=0,
    high_symmetry_points=None,
):
"""
r"""Sets offsets for a coordinate sample dataset before converting them all to momentum.

    Then, for each sample, the closest high symmetry point among the provided
    `high_symmetry_points` is calculated, and the distance to the high symmetry point obtained.
    The distance of each of the coordinate samples to these symmetry points is returned, and the
    optimizer adjusts the offsets to find a "best" set in the sense of least total L2 distance
    to the symmetry points.

    If the coordinate samples are labelled as described above as H_i, then we return

    $$
    \text{min}_j |text{P}(H_i, \Delta\phi, \Delta\theta, \ldots) - S_j|
    $$

    and the optimizer attempts to optimize for

    $$
    \sum_i \left((\text{min}_j |text{P}(H_i, \Delta\phi, \Delta\theta, \ldots) - S_j|^2)\right)^2.
    $$

    We can therefore control the metric by returning a different distance back to the optimizer,
    if desired. For instance, the L1 distance can be optimized if desired by instead returning

    $$
    \text{min}_j |text{P}(H_i, \Delta\phi, \Delta\theta, \ldots) - S_j|^\frac{1}{2}
    $$

    Args:
        coordinate_samples: (N, 6 + 1)
        phi_offset
        psi_offset
        theta_offset
        beta_offset
        chi_offset
        high_symmetry_points

    Returns:
        The MSE distance error for this set of coordinate offsets.
    """