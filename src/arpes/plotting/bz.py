"""Utilities related to plotting Brillouin zones and data onto them."""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, TypeAlias

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ase.dft.bz import bz_plot, bz_vertices
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

from arpes.analysis.mask import apply_mask_to_coords
from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger
from arpes.utilities.bz import build_2dbz_poly, process_kpath
from arpes.utilities.geometry import polyhedron_intersect_plane

from .utils import path_for_plot

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from _typeshed import Incomplete
    from ase.cell import Cell
    from matplotlib.figure import Figure
    from matplotlib.typing import ColorType
    from mpl_toolkits.mplot3d import Axes3D
    from numpy.typing import NDArray


__all__ = (
    "bz2d_segments",
    "overplot_standard",
    "plot_data_to_bz",
    "plot_data_to_bz2d",
)

LOGLEVEL = (DEBUG, INFO)[1]
logger = setup_logger(__name__, LOGLEVEL)


class Translation:
    """Base translation class, meant to provide some extension over rotations.

    Rotations are available from `scipy.spatial.transform.Rotation`.
    """

    def __init__(self, translation_vector: Sequence[float]) -> None:
        self.dim = len(translation_vector)
        self.translation_vector: NDArray[np.float64] = np.asarray(translation_vector)

    def apply(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """Applies the translation to a set of vectors.

        If this transform is D-dimensional (for D=2,3) and is applied to a different
        dimensional set of vectors, a ValueError will be thrown due to the dimension
        mismatch.

        ```
        self.apply(self.apply(vectors)) == vectors
        ```

        Args:
            vectors: array_like with shape (2 or 3,) or (N, 2 or 3)
        """
        vectors = np.asarray(vectors)

        if vectors.ndim > TWO_DIMENSION or vectors.shape[-1] not in {2, 3}:
            msg = "Expected a 2D or 3D vector (2 or 3,)"
            msg += f" of list of vectors (N, 2 or 3,), instead receivied: {vectors.shape}"
            raise ValueError(
                msg,
            )

        return vectors + self.translation_vector


Transformation: TypeAlias = Rotation | Translation


def segments_standard(
    cell: Cell,
    transformations: list | None = None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    return bz2d_segments(cell, transformations)


def overplot_standard(
    cell: Cell,
    repeat: tuple[int, int, int] | tuple[int, int] = (1, 1, 1),
    transforms: list | None = None,
) -> Callable[[Axes], Axes]:
    """A higher order function to plot a Brillouin zone over a plot.

    Args:
        cell (Cell): ASE Cell object for BZ drawing.
        repeat (tuple[int, int, int]): Set the repeating draw of BZ. default is (1, 1, 1),
            no repeat.
        transforms: List of linear transformation (scipy.spatial.transform.Rotation)

    Returns:
        Axes:
    """
    if transforms is None:
        transforms = [Rotation.from_rotvec([0, 0, 0])]

    logger.debug(f"transforms: {transforms}")

    def overplot_the_bz(ax: Axes) -> Axes:
        ax = bz_plot(
            cell=cell,
            ax=ax,
            paths=[],
            repeat=repeat,
            transforms=transforms,
            zorder=5,
        )
        ax.set_axis_on()
        return ax

    return overplot_the_bz


def apply_transformations(
    points: NDArray[np.float64],
    transformations: list[Transformation] | None = None,
) -> NDArray[np.float64]:
    """Applies a series of transformations to a sequence of vectors or a single vector.

    Args:
        points: point coordinate
        transformations: list of Transformation (Translation / Rotation)

    Returns:
        The collection of transformed points.
    """
    transformations = transformations if transformations is not None else []

    for transformation in transformations:
        points = transformation.apply(points)

    return points


def plot_plane_to_bz(
    cell: Cell,
    plane: str | list[NDArray[np.float64]],
    ax: Axes3D,
    facecolor: ColorType = "red",
) -> None:
    """Plots a 2D cut plane onto a Brillouin zone.

    Args:
        cell (Cell): ASE cell object
        plane: [TODO:description]
        ax: [TODO:description]
        special_points: [TODO:description]
        facecolor: [TODO:description]
    """
    warnings.warn(
        "This method will be deprecated.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(plane, str):
        plane_points: list[NDArray[np.float64]] = process_kpath(
            plane,
            cell,
        )[0]
    else:
        plane_points = plane

    d1, d2 = plane_points[1] - plane_points[0], plane_points[2] - plane_points[0]

    faces = [p[0] for p in bz_vertices(np.linalg.inv(cell).T)]
    pts = polyhedron_intersect_plane(faces, np.cross(d1, d2), plane_points[0])

    collection = Poly3DCollection([pts])
    collection.set_facecolor(facecolor)
    ax.add_collection3d(collection, zs=0, zdir="z")


def plot_data_to_bz(
    data: xr.DataArray,
    cell: Cell,
    **kwargs: Incomplete,
) -> Path | tuple[Figure | None, Axes]:
    """A dimension agnostic tool used to plot ARPES data onto a Brillouin zone."""
    if len(data) == TWO_DIMENSION + 1:
        raise NotImplementedError

    return plot_data_to_bz2d(data, cell, **kwargs)


def plot_data_to_bz2d(  # noqa: PLR0913
    data_array: xr.DataArray,
    cell: Cell,
    rotate: float | None = None,
    shift: NDArray[np.float64] | None = None,
    scale: float | None = None,
    ax: Axes | None = None,
    out: str | Path = "",
    bz_number: Sequence[float] | None = None,
    *,
    mask: bool = True,
    **kwargs: Incomplete,
) -> Path | tuple[Figure | None, Axes]:
    """Plots data onto the 2D Brillouin zone.

    Args:
        data_array: Data to plot
        cell(Cell): ASE Cell object (Real space)
        rotate: [TODO:description]
        shift: [TODO:description]
        scale: [TODO:description]
        ax (Axes): [TODO:description]
        out: [TODO:description]
        bz_number: [TODO:description]
        mask: [TODO:description]
        kwargs: [TODO:description]

    Returns:
        [TODO:description]
    """
    assert data_array.S.is_kspace, "You must k-space convert data before plotting to BZs"
    assert isinstance(data_array, xr.DataArray), "data_array must be xr.DataArray, not Dataset"

    if bz_number is None:
        bz_number = (0, 0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
        bz_plot(cell, paths="all", ax=ax)
    assert isinstance(ax, Axes)

    icell = cell.reciprocal()

    # Prep coordinates and mask
    raveled = data_array.G.meshgrid(as_dataset=True)
    dims = data_array.dims
    if rotate is not None:
        c, s = np.cos(rotate), np.sin(rotate)
        rotation = np.array([(c, -s), (s, c)])

        raveled = raveled.G.transform_meshgrid(dims, rotation)

    if scale is not None:
        raveled = raveled.G.scale_meshgrid(dims, scale)

    if shift is not None:
        raveled = raveled.G.shift_meshgrid(dims, shift)

    copied = data_array.values.copy()

    if mask:
        built_mask = apply_mask_to_coords(raveled, build_2dbz_poly(cell=cell), dims)
        copied[built_mask.T] = np.nan

    cmap = kwargs.get("cmap", mpl.colormaps["Blues"])
    if isinstance(cmap, str):
        cmap = mpl.colormaps.get_cmap(cmap)

    cmap.set_bad((1, 1, 1, 0))

    delta_x = np.dot(np.array(bz_number), np.array(icell)[:2, 0])
    delta_y = np.dot(np.array(bz_number), np.array(icell)[:2, 1])

    ax.pcolormesh(
        raveled.data_vars[dims[0]].values + delta_x,
        raveled.data_vars[dims[1]].values + delta_y,
        copied.T,
        cmap=cmap,
    )

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


def bz2d_segments(
    cell: Cell,
    transformations: list[Transformation] | None = None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Calculates the line segments corresponding to a 2D BZ."""
    segments_x = []
    segments_y = []
    assert cell.rank == TWO_DIMENSION

    for points, _ in twocell_to_bz1(cell)[0]:
        transformed_points = apply_transformations(points, transformations)
        x, y, _ = np.concatenate([transformed_points, transformed_points[:1]]).T
        segments_x.append(x)
        segments_y.append(y)

    return segments_x, segments_y


def twocell_to_bz1(
    cell: Cell,
) -> tuple[list[tuple[NDArray[np.float64], NDArray[np.float64]]], Cell, Cell]:
    icell = cell.reciprocal()
    bz1 = bz_vertices(icell, dim=cell.rank)
    return bz1, icell, cell
