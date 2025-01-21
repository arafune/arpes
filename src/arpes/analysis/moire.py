"""Tools for analyzing moirés and data on moiré heterostructures in particular.

All of the moirés discussed here are on hexagonal crystal systems.

NOTE: most of functions don't work.
    Modify the original plot_bz (bz_plot) in ASE should be modified to work would be more elegant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from ase.cell import Cell
from ase.dft.bz import bz_plot
from ase.lattice import HEX2D
from scipy.spatial.distance import pdist

from arpes.plotting.bz import Rotation, Translation

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "angle_between_vectors",
    "calc_commensurate_moire_cell",
    "generate_other_lattice_points",
    "mod_points_to_lattice",
    "plot_simple_moire_unit_cell",
    "unique_points",
]

RTOL = 1e-07


def mod_points_to_lattice(
    pts: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Projects points to lattice equivalent ones in the first primitive cell."""
    rmat = np.asarray([[0, -1], [1, 0]])
    ra, rb = rmat @ a, rmat @ b
    ra, rb = ra / (np.sqrt(3) / 2), rb / (np.sqrt(3) / 2)

    return pts + np.outer(np.ceil(pts @ rb), a) + np.outer(np.ceil(pts @ -ra), b)


def generate_other_lattice_points(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    ratio: float,
    order: int = 1,
    angle: float = 0,
) -> NDArray[np.float64]:
    """Generates (a, b, angle) superlattice points."""
    ratio = max(np.abs(ratio), 1 / np.abs(ratio))
    cosa, sina = np.cos(angle), np.sin(angle)
    rmat = np.asarray([[cosa, -sina], [sina, cosa]])
    a, b = rmat @ (ratio * a), rmat @ (ratio * b)

    ias = np.arange(-order, order + 1)
    pts = (a[None, None, :] * ias[None, :, None]) + (b[None, None, :] * ias[:, None, None])
    pts = pts.reshape(len(ias) ** 2, 2)

    # not quite correct, since we need the manhattan distance

    ds = np.stack(
        [
            (np.outer(ias[None, :], (ias[None, :] * 0 + 1))),
            (np.outer(ias[None, :] * 0 + 1, (ias[None, :]))),
        ],
        axis=-1,
    ).reshape(len(ias) ** 2, 2)

    dabs = np.abs(np.sum(ds, axis=1))
    dist = np.max(np.abs(ds), axis=1)
    sign = np.sign(ds)
    sign = sign[:, 0] == sign[:, 1]
    dist[sign] = dabs[sign]

    return pts[dist <= order]


def unique_points(pts: list[list[float]]) -> NDArray[np.float64]:
    """Makes a collection of points unique by removing duplicates."""
    return np.vstack([np.array(u) for u in {tuple(p) for p in pts}])


def generate_segments(
    grouped_points: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> Iterator[NDArray[np.float64]]:
    moded = mod_points_to_lattice(grouped_points, a, b)
    g1d = np.diff(np.sum(grouped_points, axis=1))
    m1d = np.diff(np.sum(moded, axis=1))

    low_index = 0
    for split_index in np.nonzero(np.abs(m1d - g1d) > 1e-11)[0]:  # noqa: PLR2004
        yield moded[low_index : split_index + 1]
        low_index = split_index + 1

    yield moded[low_index:]


def minimum_distance(
    pts: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    moded = np.stack([mod_points_to_lattice(x, a, b) for x in pts], axis=1)
    return np.min(np.stack([pdist(x) for x in moded], axis=-1), axis=0)


def angle_between_vectors(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Calculates the angle between two vectors using the law of cosines."""
    return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def calc_commensurate_moire_cell(
    underlayer_a: float,
    overlayer_a: float,
    relative_angle_rad: float = 0,
    *,
    swap_angle: bool = False,
) -> dict[str, Any]:
    """Calculates nearly commensurate moire unit cells for two hexagonal lattices.

    Args:
        underlayer_a: [TODO:description]
        overlayer_a: [TODO:description]
        relative_angle_rad: [TODO:description]
        swap_angle: [TODO:description]

    Returns:
        [TODO:description]
    """
    from ase.dft.bz import bz_vertices
    from ase.dft.kpoints import get_special_points

    underlayer_direct: Cell = HEX2D(a=underlayer_a).tocell()
    overlayer_direct: Cell = HEX2D(a=overlayer_a).tocell()

    underlayer_icell = underlayer_direct.reciprocal()
    overlayer_icell = overlayer_direct.reciprocal()

    underlayer_k = np.dot(underlayer_icell.array.T, get_special_points(underlayer_direct)["K"])
    overlayer_k = Rotation.from_rotvec([0, 0, relative_angle_rad]).apply(
        np.dot(overlayer_icell.array.T, get_special_points(overlayer_direct)["K"]),
    )

    moire_k = underlayer_k - overlayer_k
    moire_a = underlayer_a * (np.linalg.norm(underlayer_k) / np.linalg.norm(moire_k))
    moire_angle = angle_between_vectors(underlayer_k, moire_k)

    if swap_angle:
        moire_angle = -moire_angle

    moire_cell = HEX2D(float(moire_a)).tocell()
    moire_cell = Cell(Rotation.from_rotvec([0, 0, moire_angle]).apply(moire_cell))
    moire_icell = moire_cell.reciprocal()

    moire_bz_points = bz_vertices(moire_icell)[0][0][:, :2]
    return {
        "k_points": (underlayer_k, overlayer_k, moire_k),
        "moire_a": moire_a,
        "moire_k": moire_k,
        "moire_cell": moire_cell,
        "moire_icell": moire_icell,
        "moire_bz_points": moire_bz_points,
        "moire_bz_angle": moire_angle,
    }


def plot_simple_moire_unit_cell(
    lattice_consts: tuple[float, float],
    relative_angle_rad: float,
    ax: Axes | None = None,
    *,
    offset: bool = True,
    swap_angle: bool = False,
) -> None:
    """Plots a diagram of a moiré unit cell.

    In this plot, two-hexagonal-layer is assumed.

    Args:
        lattice_consts: lattice constants of the underlayer and overlayer.
        relative_angle_rad: Angle between two layers in radian.
        ax: [TODO:description]
        offset: [TODO:description]
        swap_angle: [TODO:description]
    """
    underlayer_a, overlayer_a = lattice_consts

    if ax is None:
        _, ax = plt.subplots()

    bz_plot(
        cell=HEX2D(a=underlayer_a).tocell(),
        linewidth=1,
        ax=ax,
        paths=[],
    )
    bz_plot(
        cell=HEX2D(a=overlayer_a).tocell(),
        linewidth=1,
        ax=ax,
        paths=[],
        transformations=[Rotation.from_rotvec([0, 0, relative_angle_rad])],
    )

    moire_info = calc_commensurate_moire_cell(
        underlayer_a,
        overlayer_a,
        relative_angle_rad,
        swap_angle=swap_angle,
    )
    moire_k = moire_info["moire_k"]

    k_offset = Rotation.from_rotvec([0, 0, np.deg2rad(120)]).apply(moire_k) if offset else 0

    bz_plot(
        cell=HEX2D(a=moire_info["moire_a"]).tocell(),
        linewidth=1,
        ax=ax,
        paths=[],
        transformations=[
            Rotation.from_rotvec([0, 0, -moire_info["moire_bz_angle"]]),
            Translation(moire_info["k_points"][0] + k_offset),
        ],
    )
