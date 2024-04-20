"""Contains specifications and paths to BZs for some common materials.

These are
useful if you know them ahead of hand, either from a picture, from a DFT
calculation, tight binding calculation, or explicit specification.

This is used in the interactive BZ explorer in order to help with orienting
yourself in momentum.

Each zone definition corresponds to the following things:

1. The geometry, this gives the physical Brillouin zone size and shape
2. The material work function, if available
3. The material inner potential, if available
4. The material name
"""

from __future__ import annotations

import functools
import pathlib
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from ase.dft.bz import bz_vertices
from ase.lattice import HEX2D

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase.cell import Cell
    from numpy.typing import NDArray

A_GRAPHENE = 2.46 / (2 * np.pi)
A_WS2 = 3.15 / (2 * np.pi)
A_WSe2 = 3.297 / (2 * np.pi)


def bz_points_for_hexagonal_lattice(a: float = 1) -> NDArray[np.float_]:
    """Calculates the Brillouin zone corners for a triangular (colloq. hexagona) lattice.

    Args:
        a (float): lattice constant of the hexagonal lattice.

    Returns (NDArray[np.float_]):
        Brillouin zone points
    """
    cell = HEX2D(a=a)
    icell = cell.tocell().reciprocal()
    bz_vertices_ = bz_vertices(icell, dim=2)

    # get the first face which has six points, this is the top or bottom
    # face of the cell
    return bz_vertices_[0][0][:, :2]


def image_for(file: str) -> str:
    """Loads a preview image showing the Brillouin zone shape."""
    f = pathlib.Path(__file__).parent / ".." / "example_data" / "brillouin_zones" / file
    return str(f.absolute())


class MaterialParams2D(TypedDict, total=False):
    """Material Parameters."""

    name: str
    work_function: float
    inner_potential: float
    bz_points: Callable[..., NDArray[np.float_]]
    image: str
    image_waypoints: list[list[float]]
    image_src: str
    cell: Cell


SURFACE_ZONE_DEFINITIONS: dict[str, MaterialParams2D] = {
    "2H-WS2": {
        "name": "2H-Tungsten Disulfide",
        "work_function": np.nan,
        "inner_potential": np.nan,
        "cell": HEX2D(a=3.15),
        "bz_points": functools.partial(
            bz_points_for_hexagonal_lattice,
            a=A_WS2,
        ),  # TODO: Revise
    },
    "Graphene": {
        "name": "Graphene",
        "work_function": np.nan,
        "inner_potential": np.nan,
        "cell": HEX2D(a=2.46),
        "bz_points": functools.partial(
            bz_points_for_hexagonal_lattice,
            a=A_GRAPHENE,
        ),  # TODO: Revise
    },
    "2H-WSe2": {
        "name": "Tungsten Diselenide",
        "work_function": np.nan,
        "inner_potential": np.nan,
        "cell": HEX2D(a=3.297),
        "bz_points": functools.partial(
            bz_points_for_hexagonal_lattice,
            a=A_WSe2,
        ),  # TODO: Revise
    },
    "1T-TiSe2": {
        "name": "1T-Titanium Diselenide",
        "work_function": np.nan,
        "inner_potential": np.nan,
        "image": image_for("1t-tise2-bz.png"),
        "image_waypoints": [
            # everywhere waypoints are pixel_x, pixel_y, mom_x, mom_y
            # two waypoints are required in order to specify
            [],
            [],
        ],
        "image_src": "https://arxiv.org/abs/1712.04967",
    },
    "Td-WTe2": {
        "name": "Td-Tungsten Ditelluride",
        "work_function": np.nan,
        "inner_potential": np.nan,
        "image": image_for("td-wte2-bz.png"),
        "image_waypoints": [
            [445, 650, -0.4, -0.2],
            [1470, 166, 0.4, 0.2],
        ],
        "image_src": "https://arxiv.org/abs/1603.08508",
    },
    "NCCO": {
        "name": "Nd_{2-x}Ce_xCuO_4",
        "work_function": np.nan,
        "inner_potential": np.nan,
        "image": image_for("cuprate-bz.png"),
        "image_waypoints": [
            [],
            [],
        ],
        "image_src": "https://vishiklab.faculty.ucdavis.edu/wp-content/uploads/sites/394/2016/12/ARPES-studies-of-cuprates-online.pdf",
    },
    "Bi2212": {
        "name": "Bi_2Sr_2CaCu_2O_{8+x}",
        "work_function": np.nan,
        "inner_potential": np.nan,
        "image": image_for("cuprate-bz.png"),
        "image_waypoints": [
            [],
            [],
        ],
        "image_src": "https://vishiklab.faculty.ucdavis.edu/wp-content/uploads/sites/394/2016/12/ARPES-studies-of-cuprates-online.pdf",
    },
    "1H-NbSe2": {
        "name": "1H-Niobium Diselenide",
        "work_function": np.nan,
        "inner_potential": np.nan,
        "image": image_for("1h-nbse2-bz.png"),
        "image_waypoints": [
            [],
            [],
        ],
        "image_src": "https://www.nature.com/articles/s41467-018-03888-4",
    },
    "1H-TaS2": {
        "name": "1H-Tantalum Disulfide",
        "work_function": np.nan,
        "inner_potential": np.nan,
        "image": image_for("1h-tas2-bz.png"),
        "image_waypoints": [
            [],
            [],
        ],
        "image_src": "https://www.nature.com/articles/s41467-018-03888-4",
    },
}
