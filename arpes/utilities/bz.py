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
from ase.dft.kpoints import get_special_points

from arpes.constants import TWO_DIMENSION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _typeshed import Incomplete
    from numpy.typing import ArrayLike, NDArray

    from arpes._typing import DataType, XrTypes

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


class SpecialPoint(NamedTuple):
    name: str
    negate: bool
    bz_coord: NDArray[np.float_] | Sequence[float] | tuple[float, float, float]


def make_special_points(cell: Sequence[Sequence[float]] | NDArray[np.float_]) -> list[SpecialPoint]:
    """Make a list of Special Points from the cell vectors.

    Args:
        cell (Sequence[Sequence[float]] | NDArray[float]): Matrix of the cell (3x3).
        if (2x2) matrix (0, 0, 1) is padded.

    Returns:
        list[SpecialPoint]
    """
    cell_array = np.array(cell)
    if cell_array.shape == (2, 2):
        cell_array = np.array(
            [[*c, 0] for c in cell_array] + [[0, 0, 1]],
        )
    assert cell_array.shape == (3, 3), "cell must be set as 3D."
    return [
        SpecialPoint(name=k, negate=False, bz_coord=v) for k, v in get_special_points(cell).items()
    ]


def as_2d(points_3d: ArrayLike) -> NDArray[np.float_]:
    """Takes a 3D points and converts to a 2D representation by dropping the z coordinates."""
    np_points = np.array(points_3d)
    return np_points[:, :2]


def parse_single_path(path: str) -> list[SpecialPoint]:
    """Converts a path given by high symmetry point names to numerical coordinate arrays.

    Args:
        path: [TODO:description]

    Returns:
        [TODO:description]

    ToDo: Shold be removed.  Use ase.
    """
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

        if len(bz_coords) == TWO_DIMENSION:
            bz_coords = (*list(bz_coords), 0)
        points.append(SpecialPoint(name=name, negate=negate, bz_coord=bz_coords))

    return points


def _parse_path(paths: str | list[str]) -> list[list[SpecialPoint]]:
    """Converts paths to arrays with the coordinate locations for those paths.

    Args:
        paths: [TODO:description]

    Returns:
        [TODO:description]

    ToD: Test
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

    ToDo: Test
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
          The key is the name of symmetry point, the value is coordinates in the momentum space.
              c.f. ) get_special_points( ((1, 0, 0),(0, 1, 0), (0, 0, 1)))
                       {'G': array([0., 0., 0.]),
                        'M': array([0.5, 0.5, 0. ]),
                        'R': array([0.5, 0.5, 0.5]),
                        'X': array([0. , 0.5, 0. ])}

    Returns:
        [TODO:description]

    ToDo: Test
    """
    if len(cell) == TWO_DIMENSION:
        cell = [[*c, 0] for c in cell] + [[0, 0, 1]]

    icell = np.linalg.inv(cell).T

    if special_points is None:
        from ase.dft.kpoints import get_special_points

        special_points = get_special_points(cell)
    assert isinstance(special_points, dict)

    return [
        [special_point_to_vector(elem, icell, special_points) for elem in p]
        for p in _parse_path(paths)
    ]


# Some common Brillouin zone formats
def orthorhombic_cell(a: float = 1, b: float = 1, c: float = 1) -> list[list[float]]:
    """Lattice constants for an orthorhombic unit cell.

    Args:
        a: lattice constant of along a-axis.
        b: lattice constant of along b-axis.
        c: lattice constant of along c-axis.

    Returns:
        [TODO:description]
    """
    return [[a, 0, 0], [0, b, 0], [0, 0, c]]


def hex_cell(a: float = 1, c: float = 1) -> list[list[float]]:
    """Calculates lattice vectors for a triangular lattice with lattice constants `a` and `c`.

    Args:
        a: lattice constant of along a-axis.
        c: lattice constant of along c-axis.

    Returns:
        [TODO:description]
    """
    return [[a, 0, 0], [-0.5 * a, np.sqrt(3) / 2 * a, 0], [0, 0, c]]


def hex_cell_2d(a: float = 1) -> list[list[float]]:
    """Calculates lattice vectors for a triangular lattice with lattice constant `a`.

    Args:
        a: lattice constant of along a-axis.

    Returns:
        list of list(2x2-list) that represent 2D triangular lattice.
    """
    return [[a, 0], [-0.5 * a, np.sqrt(3) / 2 * a]]


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

    Args:
        bz_indices_list: [TODO:description]

    Returns:
        [TODO:description]


    ToDo: Test
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
    """Generates the equivalent points in higher order Brillouin zones.

    [TODO:description]

    Args:
        points: [TODO:description]
        icell: [TODO:description]
        bz_indices_list: [TODO:description]

    Returns:
        [TODO:description]

    ToDo: Test
    """
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

    Args:
        vertices: [TODO:description]
        icell: [TODO:description]
        cell: [TODO:description]

    Returns:
        [TODO:description]

    ToDo:Test
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
    """Determines symmetry from a list of the symmetry points.

    [TODO:description]

    Args:
        flat_symmetry_points ([TODO:type]): [TODO:description]

    Returns:
        [TODO:description]

    ToDo: Test
    """
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
    data: XrTypes,
    symbol: str,
    *,
    include_E: bool = False,  # noqa: N803
) -> NDArray[np.float_]:
    """Calculates a displacement vector to a modded high symmetry point.

    [TODO:description]

    Args:
        data: [TODO:description]
        symbol: [TODO:description]
        include_E: [TODO:description]

    Returns:
        [TODO:description]

    Raises:
        [TODO:name]: [TODO:description]
        [TODO:name]: [TODO:description]

    ToDo: Test
    """
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    assert symmetry
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


def reduced_bz_axes(data: XrTypes) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Calculates displacement vectors to high symmetry points in the first Brillouin zone.

    Args:
        data: [TODO:description]

    Returns:
        [TODO:description]

    Raises:
        [TODO:name]: [TODO:description]
        [TODO:name]: [TODO:description]

    ToDo: Test
    """
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


def axis_along(data: XrTypes, symbol: str) -> float:
    """Determines which axis lies principally along the direction G->S.

    [TODO:description]

    Args:
        data: [TODO:description]
        symbol: [TODO:description]

    Returns:
        [TODO:description]

    ToDo: Test
    """
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v[0] for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {k: np.array([v[d] for d in data.dims if d in v]) for k, v in points.items()}

    dS = coords_by_point[symbol] - coords_by_point["G"]

    max_value = -np.inf
    max_dim = None
    for dD, d in zip(dS, [d for d in data.dims if d != "eV"], strict=False):
        if np.abs(dD) > max_value:
            max_value = np.abs(dD)
            max_dim = d
    assert isinstance(max_dim, float)
    return max_dim


def reduced_bz_poly(data: XrTypes, *, scale_zone: bool = False) -> NDArray[np.float_]:
    """Returns a polynomial representing the reduce first Brillouin zone.

    [TODO:description]

    Args:
        data: [TODO:description]
        scale_zone: [TODO:description]

    Returns:
        [TODO:description]

    ToDo: Test
    """
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
    data: XrTypes,
    symbol: str,
    e_cut: float,
    *,
    scale_zone: bool = False,
) -> NDArray[np.float_]:
    """Calculates a mask for data which contains points below an energy cutoff.

    Args:
        data: [TODO:description]
        symbol: [TODO:description]
        e_cut: [TODO:description]
        scale_zone: [TODO:description]

    Returns:
        [TODO:description]

    ToDo: Test
    """
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


def reduced_bz_mask(data: XrTypes, **kwargs: Incomplete) -> NDArray[np.float_]:
    """Calculates a mask for the first Brillouin zone of a piece of data.

    Args:
        data: [TODO:description]
        kwargs: [TODO:description]

    Returns:
        [TODO:description]

    ToDo: Test
    """
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
    """Sets data outside the Brillouin zone mask for a piece of data to be nan.

    [TODO:description]

    Args:
        data: [TODO:description]

    Returns:
        [TODO:description]
    """
    mask = reduced_bz_mask(data)

    data = data.copy()
    data.data[np.logical_not(mask)] = np.nan

    return data


def bz_cutter(symmetry_points, *, reduced: bool = True):
    """Cuts data so that it areas outside the Brillouin zone are masked away.

    [TODO:description]

    Args:
        symmetry_points ([TODO:type]): [TODO:description]
        reduced: [TODO:description]

    TODO: UNFINISHED, Test
    """

    def build_bz_mask(data) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            data ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """

    def cutter(data, cut_value: float = np.nan):
        """[TODO:summary].

        [TODO:description]

        Args:
            data ([TODO:type]): [TODO:description]
            cut_value: [TODO:description]
        """
        mask = build_bz_mask(data)

        out = data.copy()
        out.data[mask] = cut_value

        return out

    return cutter
