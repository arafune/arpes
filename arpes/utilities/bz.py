"""Utilities related to computing Brillouin zones, masks, and selecting data.

TODO: Standardize this module around support for some other library that has proper
Brillouin zone plotting, like in ASE.

This module also includes tools for masking regions of data against
Brillouin zones.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.path
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import DataType

__all__ = (
    "bz_symmetry",
    "bz_cutter",
    "reduced_bz_selection",
    "reduced_bz_axes",
    "reduced_bz_mask",
    "reduced_bz_poly",
    "reduced_bz_axis_to",
    "reduced_bz_E_mask",
    "axis_along",
    "hex_cell",
    "hex_cell_2d",
    "orthorhombic_cell",
    "process_kpath",
)


_SYMMETRY_TYPES: dict[tuple[str, ...], str] = {
    ("G", "X", "Y"): "rect",
    ("G", "X"): "square",
    ("G", "X", "BX"): "hex",
}

_POINT_NAMES_FOR_SYMMETRY: dict[str, set[str]] = {
    "rect": {"G", "X", "Y"},
    "square": {"G", "X"},
    "hex": {"G", "X", "BX"},
}

TWO_DIMENSIONAL = 2


class SpecialPoint(NamedTuple):
    name: str
    negate: bool
    bz_coord: NDArray[np.float_] | list[float] | tuple[float, ...]


def as_3d(points: NDArray[np.float_]) -> NDArray[np.float_]:
    """Takes a 2D points list and zero pads to convert to a 3D representation."""
    return np.concatenate([points, points[:, 0][:, None] * 0], axis=1)


def as_2d(points: NDArray[np.float_]) -> NDArray[np.float_]:
    """Takes a 3D points and converts to a 2D representation by dropping the z coordinates."""
    return points[:, :2]


def parse_single_path(path: str) -> list[SpecialPoint]:
    """Converts a path given by high symmetry point names to numerical coordinate arrays."""
    # first tokenize
    tokens = [name for name in re.split(r"([A-Z][a-z0-9]*(?:\([0-9,\s]+\))?)", path) if name]

    # normalize Gamma to G
    tokens = [token.replace("Gamma", "G") for token in tokens]

    # convert to standard format
    points = []
    for token in tokens:
        name, rest = token[0], token[1:]
        negate = False
        if rest and rest[0] == "n":
            negate = True
            rest = rest[1:]

        bz_coords: tuple[float, float, float] | tuple[float, float] = (
            0.0,
            0.0,
            0.0,
        )
        if rest:
            rest = "".join(c for c in rest if c not in "( \t\n\r)")
            bz_coords = tuple([int(c) for c in rest.split(",")])

        if len(bz_coords) == TWO_DIMENSIONAL:
            bz_coords = (*list(bz_coords), 0)
        points.append(SpecialPoint(name=name, negate=negate, bz_coord=bz_coords))

    return points


def parse_path(paths: str | list[str]) -> list[list[SpecialPoint]]:
    """Converts paths to arrays with the coordinate locations for those paths.

    Args:
        paths: [TODO:description]

    Returns:
        [TODO:description]
    """
    if isinstance(paths, str):
        # some manual string work in order to make sure we do not split on commas inside BZ indices
        idxs = []
        for i, p in enumerate(paths):
            if p == ",":
                c = Counter(paths[:i])
                if c["("] - c[")"] == 0:
                    idxs.append(i)

        paths = list(paths)
        for idx in idxs:
            paths[idx] = ":"

        paths = "".join(paths)
        paths = paths.split(":")

    return [parse_single_path(p) for p in paths]


def special_point_to_vector(
    special_point: SpecialPoint,
    icell: NDArray[np.float_],
    special_points: dict[str, NDArray[np.float_]],
) -> NDArray[np.float_]:
    """Converts a single special point to its coordinate vector.

    Args:
        special_point: (SpecialPoint) SpecialPoint object.
        icell (NDArray[np.float_]): Reciprocal lattice cell.
        special_points (dict:str, NDArray[np.float_]): Special points in mementum space.

    Returns:
        [TODO:description]
    """
    base = np.dot(icell.T, special_points[special_point.name])

    if special_point.negate:
        base = -np.array(base)

    coord = np.array(special_point.bz_coord)
    return base + coord.dot(icell)


def process_kpath(
    paths: str | list[str],
    cell: NDArray[np.float_,],
    special_points: dict[str, NDArray[np.float_]] | None = None,
) -> list[list[NDArray[np.float_]]]:
    """Converts paths consiting of point definitions to raw coordinates.

    Args:
        paths: [TODO:description]
        cell (NDArray[np.float_]): Three vector representing the unit cell .
        special_points (dict:str, NDArray[np.float_]): Special points in momentum space.
              c.f. ) get_special_points( ((1, 0, 0),(0, 1, 0), (0, 0, 1)))
                       {'G': array([0., 0., 0.]),
                        'M': array([0.5, 0.5, 0. ]),
                        'R': array([0.5, 0.5, 0.5]),
                        'X': array([0. , 0.5, 0. ])}

    Returns:
        [TODO:description]
    """
    if len(cell) == TWO_DIMENSIONAL:
        cell = [[*c, 0] for c in cell] + [[0, 0, 1]]

    icell = np.linalg.inv(cell).T

    if special_points is None:
        from ase.dft.kpoints import get_special_points

        special_points = get_special_points(cell)
    assert isinstance(special_points, dict)

    return [
        [special_point_to_vector(elem, icell, special_points) for elem in p]
        for p in parse_path(paths)
    ]


# Some common Brillouin zone formats
def orthorhombic_cell(a: float = 1, b: float = 1, c: float = 1) -> list[list[float]]:
    """Lattice constants for an orthorhombic unit cell."""
    return [[a, 0, 0], [0, b, 0], [0, 0, c]]


def hex_cell(a: float = 1, c: float = 1) -> list[list[float]]:
    """Calculates lattice vectors for a triangular lattice with lattice constants `a` and `c`."""
    return [[a, 0, 0], [-0.5 * a, 3**0.5 / 2 * a, 0], [0, 0, c]]


def hex_cell_2d(a: float = 1) -> list[list[float]]:
    """Calculates lattice vectors for a triangular lattice with lattice constant `a`."""
    return [[a, 0], [-0.5 * a, 3**0.5 / 2 * a]]


def flat_bz_indices_list(
    bz_indices_list: Sequence[Sequence[float]] | None = None,
) -> list[tuple[int, ...]]:
    """Calculate a flat representation of a repeated Brillouin zone specification.

    This is useful for plotting extra Brillouin zones or generating high symmetry points,
    lines, and planes.

    If None is provided, the first BZ is assumed.

    ```
    None -> [(0,0)]
    ```

    If an explicit zone is provided or a list of zones is provided, these are
    returned

    ```
    [(0,1,0), (-1, -1, 2)] -> [(0,1,0), (-1, -1, 2)]
    ```

    Additionally, tuples are unpacked into ranges

    ```
    [((-2, 1), 1)] -> [(-2, 1), (-1, 1), (0, 1)]
    ```
    """
    if bz_indices_list is None:
        bz_indices_list = [(0, 0)]

    assert len(bz_indices_list[0]) in {2, 3}

    indices = []
    if len(bz_indices_list[0]) == 2:  # noqa: PLR2004
        for bz_x, bz_y in bz_indices_list:
            rx = range(bz_x, bz_x + 1) if isinstance(bz_x, int) else range(*bz_x)
            ry = range(bz_y, bz_y + 1) if isinstance(bz_y, int) else range(*bz_y)
            for x, y in itertools.product(rx, ry):
                indices.append((x, y))
    else:
        for bz_x, bz_y, bz_z in bz_indices_list:
            rx = range(bz_x, bz_x + 1) if isinstance(bz_x, int) else range(*bz_x)
            ry = range(bz_y, bz_y + 1) if isinstance(bz_y, int) else range(*bz_y)
            rz = range(bz_z, bz_z + 1) if isinstance(bz_z, int) else range(*bz_z)
            for x, y, z in itertools.product(rx, ry, rz):
                indices.append((x, y, z))

    return indices


def generate_2d_equivalent_points(
    points: NDArray[np.float_],
    icell: NDArray[np.float_],
    bz_indices_list: Sequence[Sequence[float]] | None = None,
) -> NDArray[np.float_]:
    """Generates the equivalent points in higher order Brillouin zones."""
    points_list = []
    for x, y in flat_bz_indices_list(bz_indices_list):
        points_list.append(
            points[:, :2]
            + x
            * icell[0][
                None,
                :2,
            ]
            + y
            * icell[1][
                None,
                :2,
            ],
        )

    return np.unique(np.concatenate(points_list), axis=0)


def build_2dbz_poly(
    vertices: NDArray[np.float_] | None = None,
    icell: NDArray[np.float_] | None = None,
    cell: Sequence[Sequence[float]] | None = None,
) -> dict[str, list[float]]:
    """Converts brillouin zone or equivalent information to a polygon mask.

    This mask can be used to mask away data outside the zone boundary.
    """
    from ase.dft.bz import bz_vertices  # pylint: disable=import-error

    from arpes.analysis.mask import raw_poly_to_mask

    assert cell is not None or vertices is not None or icell is not None

    if vertices is None:
        if icell is None:
            icell = np.linalg.inv(cell).T

        vertices = bz_vertices(icell)

    points, _ = vertices[0]  # points, normal
    points_2d = [p[:2] for p in points]

    return raw_poly_to_mask(points_2d)


def bz_symmetry(flat_symmetry_points) -> str | None:
    """Determines symmetry from a list of the symmetry points."""
    if isinstance(flat_symmetry_points, dict):
        flat_symmetry_points = flat_symmetry_points.items()

    largest_identified = 0
    symmetry: str | None = None

    point_names = {k for k, _ in flat_symmetry_points}

    for points, sym in _SYMMETRY_TYPES.items():
        if all(p in point_names for p in points) and len(points) > largest_identified:
            symmetry = sym
            largest_identified = len(points)

    return symmetry


def reduced_bz_axis_to(
    data: DataType,
    symbol: str,
    *,
    include_E: bool = False,  # noqa: N803
) -> NDArray[np.float_]:
    """Calculates a displacement vector to a modded high symmetry point."""
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {
        k: np.array([v.get(d, 0) for d in data.dims if d in v or include_E and d == "eV"])
        for k, v in points.items()
    }
    if symmetry == "rect":
        if symbol == "X":
            return coords_by_point["X"] - coords_by_point["G"]
        return coords_by_point["Y"] - coords_by_point["G"]
    if symmetry == "square":
        raise NotImplementedError
        return coords_by_point["X"] - coords_by_point["G"]
    if symmetry == "hex":
        if symbol == "X":
            return coords_by_point["X"] - coords_by_point["G"]
        return coords_by_point["BX"] - coords_by_point["G"]
    raise NotImplementedError


def reduced_bz_axes(data: DataType) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Calculates displacement vectors to high symmetry points in the first Brillouin zone."""
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {k: np.array([v[d] for d in data.dims if d in v]) for k, v in points.items()}
    if symmetry == "rect":
        dx = coords_by_point["X"] - coords_by_point["G"]
        dy = coords_by_point["Y"] - coords_by_point["G"]
    elif symmetry == "square":
        raise NotImplementedError
        dx = coords_by_point["X"] - coords_by_point["G"]
        dy = coords_by_point["X"] - coords_by_point["G"]
    elif symmetry == "hex":
        dx = coords_by_point["X"] - coords_by_point["G"]
        dy = coords_by_point["BX"] - coords_by_point["G"]
    else:
        raise NotImplementedError

    return dx, dy


def axis_along(data: DataType, symbol: str) -> float:
    """Determines which axis lies principally along the direction G->S."""
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {k: np.array([v[d] for d in data.dims if d in v]) for k, v in points.items()}

    dS = coords_by_point[symbol] - coords_by_point["G"]

    max_value = -np.inf
    max_dim = None
    for dD, d in zip(dS, [d for d in data.dims if d != "eV"]):
        if np.abs(dD) > max_value:
            max_value = np.abs(dD)
            max_dim = d
    assert isinstance(max_dim, float)
    return max_dim


def reduced_bz_poly(data: DataType, *, scale_zone: bool = False) -> NDArray[np.float_]:
    """Returns a polynomial representing the reduce first Brillouin zone."""
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]
    dx, dy = reduced_bz_axes(data)
    if scale_zone:
        # should be good enough, reevaluate later
        dx = 3 * dx
        dy = 3 * dy

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}
    coords_by_point = {
        k: np.array([v.get(d, 0) for d in data.dims if d in v]) for k, v in points.items()
    }

    if symmetry == "hex":
        return np.array(
            [
                coords_by_point["G"],
                coords_by_point["G"] + dx,
                coords_by_point["G"] + dy,
            ],
        )

    return np.array(
        [
            coords_by_point["G"],
            coords_by_point["G"] + dx,
            coords_by_point["G"] + dx + dy,
            coords_by_point["G"] + dy,
        ],
    )


def reduced_bz_E_mask(
    data: DataType,
    symbol: str,
    e_cut: float,
    *,
    scale_zone: bool = False,
) -> NDArray[np.float_]:
    """Calculates a mask for data which contains points below an energy cutoff."""
    symmetry_points, _ = data.S.symmetry_points()
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}
    coords_by_point = {
        k: np.array([v.get(d, 0) for d in data.dims if d in v or d == "eV"])
        for k, v in points.items()
    }

    dx_to = reduced_bz_axis_to(data, symbol, include_E=True)
    if scale_zone:
        dx_to = dx_to * 3
    dE = np.array([0 if d != "eV" else e_cut for d in data.dims])

    poly_points = np.array(
        [
            coords_by_point["G"],
            coords_by_point["G"] + dx_to,
            coords_by_point["G"] + dx_to + dE,
            coords_by_point["G"] + dE,
        ],
    )

    skip_col = None
    for i in range(poly_points.shape[1]):
        if np.all(poly_points[:, i] == poly_points[0, i]):
            skip_col = i

    assert skip_col is not None
    selector_val = poly_points[0, skip_col]
    poly_points = np.concatenate(
        (poly_points[:, 0:skip_col], poly_points[:, skip_col + 1 :]),
        axis=1,
    )

    selector = {}
    selector[data.dims[skip_col]] = selector_val
    sdata = data.sel(**selector, method="nearest")

    path = matplotlib.path.Path(poly_points)
    grid = np.array(
        [a.ravel() for a in np.meshgrid(*[data.coords[d] for d in sdata.dims], indexing="ij")],
    ).T
    mask = path.contains_points(grid)
    return np.reshape(mask, sdata.data.shape)


def reduced_bz_mask(data: DataType, **kwargs: Incomplete) -> NDArray[np.float_]:
    """Calculates a mask for the first Brillouin zone of a piece of data."""
    symmetry_points, _ = data.S.symmetry_points()
    bz_dims = tuple(d for d in data.dims if d in next(iter(symmetry_points.values()))[0])

    poly_points = reduced_bz_poly(data, **kwargs)
    extra_dims_shape = tuple(len(data.coords[d]) for d in data.dims if d in bz_dims)

    path = matplotlib.path.Path(poly_points)
    grid = np.array(
        [a.ravel() for a in np.meshgrid(*[data.coords[d] for d in bz_dims], indexing="ij")],
    ).T
    mask = path.contains_points(grid)
    return np.reshape(mask, extra_dims_shape)


def reduced_bz_selection(data: DataType) -> DataType:
    """Sets data outside the Brillouin zone mask for a piece of data to be nan."""
    mask = reduced_bz_mask(data)

    data = data.copy()
    data.data[np.logical_not(mask)] = np.nan

    return data


def bz_cutter(symmetry_points, *, reduced: bool = True):
    """Cuts data so that it areas outside the Brillouin zone are masked away.

    TODO: UNFINISHED.
    """

    def build_bz_mask(data) -> None:
        pass

    def cutter(data, cut_value: float = np.nan):
        mask = build_bz_mask(data)

        out = data.copy()
        out.data[mask] = cut_value

        return out

    return cutter
