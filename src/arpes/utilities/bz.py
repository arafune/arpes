"""Utilities related to computing Brillouin zones, masks, and selecting data.

TODO: Standardize this module around support for some other library that has proper
Brillouin zone plotting, like in ASE.

This module also includes tools for masking regions of data against
Brillouin zones.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal, TypeVar

import matplotlib.path
import numpy as np
from ase.dft.bz import bz_vertices
from ase.dft.kpoints import bandpath

from arpes.constants import TWO_DIMENSION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _typeshed import Incomplete
    from ase.cell import Cell
    from numpy.typing import NDArray

    from arpes._typing import DataType, XrTypes

__all__ = (
    "bz_symmetry",
    "reduced_bz_selection",
    "reduced_bz_axes",
    "reduced_bz_mask",
    "reduced_bz_poly",
    "reduced_bz_axis_to",
    "reduced_bz_E_mask",
    "axis_along",
    "process_kpath",
)

BRAVAISLATTICE = Literal["RECT", "SQR", "HEX2D"]

_SYMMETRY_TYPES: dict[tuple[str, ...], str] = {
    ("G", "X", "Y", "S"): "RECT",
    ("G", "X", "M"): "SQR",
    ("G", "X", "K"): "HEX2D",
}

_POINT_NAMES_FOR_SYMMETRY: dict[
    BRAVAISLATTICE | None,
    set[str],
] = {  #  see ase.lattice.BravaisLattice
    "RECT": {"G", "X", "Y", "S"},
    "SQR": {"G", "X", "M"},
    "HEX2D": {"G", "X", "K"},
}

T = TypeVar("T")


def process_kpath(
    path: str,
    cell: Cell,
) -> NDArray[np.float_]:
    """Converts paths consiting of point definitions to raw coordinates.

    Args:
        path: String that represents the high symmetry points such as "GMK".
        cell (Cell): ASE Cell object

    Returns:
        Get Cartesian kpoints of the bandpath.
    """
    bp = bandpath(path=path, cell=cell, npoints=len(path))
    return bp.cartesian_kpts()


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
    if len(bz_indices_list[0]) == TWO_DIMENSION:
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


def build_2dbz_poly(
    cell: Cell,
) -> dict[str, list[float]]:
    """Converts brillouin zone or equivalent information to a polygon mask.

    This mask can be used to mask away data outside the zone boundary.

    Args:
        cell (Cell): ASE Cell object

    Returns:
        [TODO:description]

    ToDo:Test
    """
    from arpes.analysis.mask import raw_poly_to_mask

    icell = cell.reciprocal()
    vertices = bz_vertices(icell)

    points, _ = vertices[0]  # points, normal
    points_2d = [p[:2] for p in points]

    return raw_poly_to_mask(points_2d)


def bz_symmetry(flat_symmetry_points: dict | None) -> BRAVAISLATTICE | None:
    """Determines symmetry from a list of the symmetry points.

    Args:
        flat_symmetry_points ([TODO:type]): [TODO:description]

    Returns:
        [TODO:description]
    """
    if isinstance(flat_symmetry_points, dict):
        flat_symmetry_points = flat_symmetry_points.items()

    largest_identified = 0
    symmetry: BRAVAISLATTICE | None = None

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
    bravais_lattice: BRAVAISLATTICE | None = bz_symmetry(
        data.S.iter_own_symmetry_points,
    )
    point_names = _POINT_NAMES_FOR_SYMMETRY[bravais_lattice]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {
        k: np.array([v.get(d, 0) for d in data.dims if d in v or include_E and d == "eV"])
        for k, v in points.items()
    }
    if bravais_lattice == "RECT":
        if symbol == "X":
            return coords_by_point["X"] - coords_by_point["G"]
        return coords_by_point["Y"] - coords_by_point["G"]
    if bravais_lattice == "SQR":
        raise NotImplementedError
        return coords_by_point["X"] - coords_by_point["G"]
    if bravais_lattice == "HEX2D":
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
    bravais_lattice: BRAVAISLATTICE | None = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[bravais_lattice]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v for k, v in symmetry_points.items() if k in point_names}

    coords_by_point = {k: np.array([v[d] for d in data.dims if d in v]) for k, v in points.items()}
    if bravais_lattice == "RECT":
        dx = coords_by_point["X"] - coords_by_point["G"]
        dy = coords_by_point["Y"] - coords_by_point["G"]
    elif bravais_lattice == "SQR":
        raise NotImplementedError
        dx = coords_by_point["X"] - coords_by_point["G"]
        dy = coords_by_point["X"] - coords_by_point["G"]
    elif bravais_lattice == "HEX2D":
        dx = coords_by_point["X"] - coords_by_point["G"]
        dy = coords_by_point["BX"] - coords_by_point["G"]  # TODO: Revise it (What is BX?)
    else:
        raise NotImplementedError

    return dx, dy


def axis_along(data: XrTypes, symbol: str) -> float:
    """Determines which axis lies principally along the direction G->S.

    Args:
        data (xr.DataArray | xr.Dataset): ARPES data
        symbol: [TODO:description]

    Returns:
        [TODO:description]

    ToDo: Test
    """
    symmetry = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[symmetry]

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v for k, v in symmetry_points.items() if k in point_names}

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

    Args:
        data: [TODO:description]
        scale_zone: [TODO:description]

    Returns:
        [TODO:description]

    ToDo: Test
    """
    bravais_latticey = bz_symmetry(data.S.iter_own_symmetry_points)
    point_names = _POINT_NAMES_FOR_SYMMETRY[bravais_latticey]
    dx, dy = reduced_bz_axes(data)
    if scale_zone:
        # should be good enough, reevaluate later
        dx = 3 * dx
        dy = 3 * dy

    symmetry_points, _ = data.S.symmetry_points()
    points = {k: v for k, v in symmetry_points.items() if k in point_names}
    coords_by_point = {
        k: np.array([v.get(d, 0) for d in data.dims if d in v]) for k, v in points.items()
    }

    if bravais_latticey == "HEX2D":
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
    points = {k: v for k, v in symmetry_points.items() if k in point_names}
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
    sdata = data.sel(selector, method="nearest")

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

    Args:
        data: [TODO:description]

    Returns:
        [TODO:description]
    """
    mask = reduced_bz_mask(data)

    data = data.copy()
    data.data[np.logical_not(mask)] = np.nan

    return data
