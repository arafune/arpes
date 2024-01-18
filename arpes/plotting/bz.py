"""Utilities related to plotting Brillouin zones and data onto them."""
# pylint: disable=import-error
from __future__ import annotations

import itertools
import warnings
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, TypeAlias

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

from arpes.analysis.mask import apply_mask_to_coords
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.bz import build_2dbz_poly, hex_cell_2d, process_kpath
from arpes.utilities.bz_spec import A_GRAPHENE, A_WS2, A_WSe2
from arpes.utilities.geometry import polyhedron_intersect_plane

from .utils import path_for_plot

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from _typeshed import Incomplete
    from matplotlib.figure import Figure
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike, NDArray

    from arpes._typing import DataType

__all__ = (
    "annotate_special_paths",
    "bz2d_plot",
    "bz3d_plot",
    "bz_plot",
    "plot_data_to_bz",
    "plot_data_to_bz2d",
    "plot_data_to_bz3d",
    "plot_plane_to_bz",
    "bz2d_segments",
    "overplot_standard",
)

overplot_library = {
    "graphene": lambda: {"cell": hex_cell_2d(A_GRAPHENE)},
    "ws2": lambda: {"cell": hex_cell_2d(A_WS2)},
    "wwe2": lambda: {"cell": hex_cell_2d(A_WSe2)},
}


LOGLEVEL = (DEBUG, INFO)[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def segments_standard(
    name: str = "graphene",
    rotate: float = 0.0,
) -> tuple[list[NDArray[np.float_]], list[NDArray[np.float_]]]:
    name = name.lower()
    specification = overplot_library[name]()
    transformations = []
    if rotate:
        transformations = [Rotation.from_rotvec([0, 0, rotate])]

    return bz2d_segments(specification["cell"], transformations)


def overplot_standard(
    name: str = "graphene",
    repeat: tuple[int, int, int] | tuple[int, int] | None = None,
    rotate: float = 0,
) -> Callable[[Axes], Axes]:
    """A higher order function to plot a Brillouin zone over a plot."""
    specification = overplot_library[name]()
    transformations = []

    if rotate:
        transformations = [Rotation.from_rotvec([0, 0, rotate])]

    def overplot_the_bz(ax: Axes) -> Axes:
        return bz_plot(
            cell=specification["cell"],
            linewidth=2,
            ax=ax,
            paths=[],
            repeat=repeat,
            set_equal_aspect=False,
            hide_ax=False,
            transformations=transformations,
            zorder=5,
            linestyle="-",
        )

    return overplot_the_bz


class Translation:
    """Base translation class, meant to provide some extension over rotations.

    Rotations are available from `scipy.spatial.transform.Rotation`.
    """

    translation_vector = None

    def __init__(self, translation_vector: ArrayLike) -> None:
        self.translation_vector = np.asarray(translation_vector)

    def apply(self, vectors: ArrayLike, *, inverse: bool = False) -> NDArray[np.float_]:
        """Applies the translation to a set of vectors.

        If this transform is D-dimensional (for D=2,3) and is applied to a different
        dimensional set of vectors, a ValueError will be thrown due to the dimension
        mismatch.

        An inverse flag is available in order to apply the inverse coordinate transform.
        Up to numerical accuracy,

        ```
        self.apply(self.apply(vectors), inverse=True) == vectors
        ```

        Args:
            vectors: array_like with shape (2 or 3,) or (N, 2 or 3)
            inverse: Applies the inverse coordinate transform instead
        """
        vectors = np.asarray(vectors)

        if vectors.ndim > 2 or vectors.shape[-1] not in {2, 3}:  # noqa: PLR2004
            msg = "Expected a 2D or 3D vector (2 or 3,)"
            msg += f" of list of vectors (N, 2 or 3,), instead receivied: {vectors.shape}"
            raise ValueError(
                msg,
            )

        single_vector = False
        if vectors.ndim == 1:
            single_vector = True
            vectors = vectors[None, :]  # expand dims

        result = vectors - self.translation_vector if inverse else vectors + self.translation_vector

        return result if not single_vector else result[0]


Transformation: TypeAlias = Rotation | Translation


def apply_transformations(
    points: NDArray[np.float_],
    transformations: list[Transformation] | None = None,
    *,
    inverse: bool = False,
) -> NDArray[np.float_]:
    """Applies a series of transformations to a sequence of vectors or a single vector.

    Args:
        points: point coordinate
        transformations: list of Transformation (Translation / Rotation)
        inverse (bool): Applies the inverse coordinate transform instead

    Returns:
        The collection of transformed points.
    """
    if transformations is None:
        transformations = []

    for transformation in transformations:
        points = transformation.apply(points, inverse=inverse)

    return points


def plot_plane_to_bz(
    cell: Sequence[Sequence[float]],
    plane: str | list[NDArray[np.float_]],
    ax: Axes,
    special_points: dict[str, NDArray[np.float_]] | None = None,
    facecolor: ColorType = "red",
) -> None:
    """Plots a 2D cut plane onto a Brillouin zone."""
    from ase.dft.bz import bz_vertices

    if isinstance(plane, str):
        plane_points: list[NDArray[np.float_]] = process_kpath(
            plane,
            cell,
            special_points=special_points,
        )[0]
    else:
        plane_points = plane

    d1, d2 = plane_points[1] - plane_points[0], plane_points[2] - plane_points[0]

    faces = [p[0] for p in bz_vertices(np.linalg.inv(cell).T)]
    pts = polyhedron_intersect_plane(faces, np.cross(d1, d2), plane_points[0])

    collection = Poly3DCollection([pts])
    collection.set_facecolor(facecolor)
    ax.add_collection3d(collection, zs="z")


def plot_data_to_bz(
    data: DataType,
    cell: Sequence[Sequence[float]],
    **kwargs: Incomplete,
):
    """A dimension agnostic tool used to plot ARPES data onto a Brillouin zone."""
    if len(data) == 3:  # noqa: PLR2004
        return plot_data_to_bz3d(data, cell, **kwargs)

    return plot_data_to_bz2d(data, cell, **kwargs)


def plot_data_to_bz2d(  # noqa: PLR0913
    data: DataType,
    cell: Sequence[Sequence[float]],
    rotate: float | None = None,
    shift: NDArray[np.float_] | None = None,
    scale: float | None = None,
    ax: Axes | None = None,
    out: str | Path = "",
    bz_number: Sequence[float] | None = None,
    *,
    mask: bool = True,
    **kwargs: Incomplete,
) -> Path | tuple[Figure, Axes]:
    """Plots data onto a 2D Brillouin zone."""
    data_array = normalize_to_spectrum(data)

    assert data_array.S.is_kspace, "You must k-space convert data before plotting to BZs"
    assert isinstance(data_array, xr.DataArray)

    if bz_number is None:
        bz_number = (0, 0)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
        bz2d_plot(cell, paths="all", ax=ax)
    assert isinstance(ax, Axes)

    if len(cell) == 2:  # noqa: PLR2004
        cell = [[*list(c), 0] for c in cell] + [[0, 0, 1]]

    icell = np.linalg.inv(cell).T

    # Prep coordinates and mask
    raveled = data_array.G.meshgrid(as_dataset=True)
    dims = data_array.dims
    if rotate is not None:
        c, s = np.cos(rotate), np.sin(rotate)
        rotation = np.array([(c, -s), (s, c)])

        raveled = raveled.G.transform_coords(dims, rotation)

    if scale is not None:
        raveled = raveled.G.scale_coords(dims, scale)

    if shift is not None:
        raveled = raveled.G.shift_coords(dims, shift)

    copied = data_array.values.copy()

    if mask:
        built_mask = apply_mask_to_coords(raveled, build_2dbz_poly(cell=cell), dims)
        copied[built_mask.T] = np.nan

    cmap = kwargs.get("cmap", matplotlib.colormaps["Blues"])
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps.get_cmap(cmap)

    cmap.set_bad((1, 1, 1, 0))

    delta_x = np.dot(np.array(bz_number), icell[:2, 0])
    delta_y = np.dot(np.array(bz_number), icell[:2, 1])

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


def plot_data_to_bz3d(
    data: DataType,
    cell: Sequence[Sequence[float]],
    **kwargs: Incomplete,
) -> None:
    """Plots ARPES data onto a 3D Brillouin zone."""
    msg = "plot_data_to_bz3d is not implemented yet."
    logger.debug(f"id of data: {data.attrs.get('id', None)}")
    logger.debug(f"cell: {cell}")
    if kwargs:
        for k, v in kwargs.items():
            logger.debug(f"kwargs; k: {k}, v: {v}")
    raise NotImplementedError(msg)


def bz_plot(cell: Sequence[Sequence[float]], *args, **kwargs: Incomplete) -> Axes:
    """Dimension generic BZ plot which uses the cell dimension to delegate."""
    logger.debug(f"size of cell is: {format(len(cell))}")
    if len(cell) > 2:  # noqa: PLR2004
        return bz3d_plot(cell, *args, **kwargs)

    return bz2d_plot(cell, *args, **kwargs)


def bz3d_plot(
    cell: Sequence[Sequence[float]],
    paths: str | list[str | float] | None = None,
    kpoints: Sequence[Sequence[float]] | None = None,
    ax: Axes | None = None,
    elev: float | None = None,
    scale: float = 1,
    repeat: tuple[int, int, int] | None = None,
    transformations: list[Transformation] | None = None,
    *,
    vectors: bool = False,
    hide_ax: bool = True,
    **kwargs: Incomplete,
) -> Axes:
    """For now this is lifted from ase.dft.bz.bz3d_plot with some modifications.

    All copyright and licensing terms for this and bz2d_plot are those of the current release of ASE
    (Atomic Simulation Environment).
    """
    try:
        from ase.dft.bz import bz_vertices  # dynamic because we do not require ase
    except ImportError:
        warnings.warn(
            "You will need to install ASE (Atomic Simulation Environment) to use this feature.",
            stacklevel=2,
        )
        msg = "You will need to install ASE before using Brillouin Zone plotting"
        raise ImportError(msg)

    class Arrow3D(FancyArrowPatch):
        def __init__(
            self,
            xs: Sequence[float],
            ys: Sequence[float],
            zs: Sequence[float],
            *args,
            **kwargs: Incomplete,
        ) -> None:
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer) -> None:
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    icell = np.linalg.inv(cell).T

    if isinstance(paths, str):
        from ase.cell import Cell
        from ase.dft.kpoints import parse_path_string

        cell_structure = Cell(cell).get_bravais_lattice()
        special_points: dict[str, NDArray[np.float_]] = cell_structure.get_special_points()
        path_string = cell_structure.special_path if paths == "all" else paths
        paths = []
        for names in parse_path_string(path_string):
            points = []
            for name in names:
                points.append(np.dot(icell.T, special_points[name]))
                paths.append((names, points))

    if ax is None:
        fig: Figure = plt.figure(figsize=(5, 5))
        ax = fig.gca(projection="3d")
    assert isinstance(ax, Axes3D)

    azim = np.pi / 5
    elev = elev or np.pi / 6
    x = np.sin(azim)
    y = np.cos(azim)
    view = [x * np.cos(elev), y * np.cos(elev), np.sin(elev)]

    bz1 = bz_vertices(icell)

    maxp = 0.0

    if repeat is None:
        repeat = (
            1,
            1,
            1,
        )

    dx, dy, dz = icell[0], icell[1], icell[2]
    rep_x: int | tuple[int, int]
    rep_y: int | tuple[int, int]
    rep_z: int | tuple[int, int]
    rep_x, rep_y, rep_z = repeat

    if isinstance(rep_x, int):
        rep_x = (0, rep_x)
    if isinstance(rep_y, int):
        rep_y = (0, rep_y)
    if isinstance(rep_z, int):
        rep_z = (0, rep_z)

    c = kwargs.pop("c", "k")
    c = kwargs.pop("color", c)

    for ix, iy, iz in itertools.product(range(*rep_x), range(*rep_y), range(*rep_z)):
        delta = dx * ix + dy * iy + dz * iz

        for points, normal in bz1:
            color = c

            ls = ":" if np.dot(normal, view) < 0 else "-"

            cosines = np.dot(icell, normal) / np.linalg.norm(normal) / np.linalg.norm(icell, axis=1)
            for idx, cosine in enumerate(cosines):
                if np.abs(np.abs(cosine) - 1) < 1e-6 and False:  # debugging this
                    tup = [rep_x, rep_y, rep_z][idx]
                    current = [ix, iy, iz][idx]

                    if cosine < 0:
                        current = current - 1

                    if tup[0] < current + 1 < tup[1]:
                        color = (1, 1, 1, 0)

                    if current + 1 != tup[1] and current != tup[0]:
                        ls = ":"
                        color = "blue"

            x, y, z = np.concatenate([points, points[:1]]).T
            x, y, z = x + delta[0], y + delta[1], z + delta[2]

            ax.plot(x, y, z, c=color, ls=ls, **kwargs)
            maxp = max(maxp, points.max())

    if vectors:
        ax.add_artist(
            Arrow3D(
                [0, icell[0, 0]],
                [0, icell[0, 1]],
                [0, icell[0, 2]],
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color="k",
            ),
        )
        ax.add_artist(
            Arrow3D(
                [0, icell[1, 0]],
                [0, icell[1, 1]],
                [0, icell[1, 2]],
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color="k",
            ),
        )
        ax.add_artist(
            Arrow3D(
                [0, icell[2, 0]],
                [0, icell[2, 1]],
                [0, icell[2, 2]],
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color="k",
            ),
        )
        maxp = max(maxp, 0.6 * icell.max())

    if paths is not None:
        for names, points in paths:
            x, y, z = np.array(points).T
            ax.plot(x, y, z, c="r", ls="-")

            for name, point in zip(names, points):
                x, y, z = point
                if name == "G":
                    name = "\\Gamma"
                elif len(name) > 1:
                    name = name[0] + "_" + name[1]
                ax.text(x, y, z, "$" + name + "$", ha="center", va="bottom", color="r")

    if kpoints is not None:
        for p in kpoints:
            ax.scatter(p[0], p[1], p[2], c="b")

    if hide_ax:
        ax.set_axis_off()
        ax.autoscale_view(tight=True)

    s = maxp / 0.5 * 0.45 * scale
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_zlim(-s, s)
    ax.set_aspect("equal")

    ax.view_init(azim=azim / np.pi * 180, elev=elev / np.pi * 180)


def annotate_special_paths(
    ax: Axes,
    paths: list[str] | str,
    cell: Sequence[Sequence[float]] | None = None,
    transformations=None,
    offset: dict[str, float | Sequence[float]] | None = None,
    special_points: dict[str, NDArray[np.float_]] | None = None,
    labels=None,
    **kwargs: Incomplete,
) -> None:
    """Annotates user indicated paths in k-space by plotting lines (or points) over the BZ."""
    logger.debug(f"ax: {ax}")
    logger.debug(f"paths: {paths}")
    logger.debug(f"cell: {cell}")
    logger.debug(f"offset: {offset}")
    logger.debug(f"special_points: {special_points}")
    logger.debug(f"labels: {labels}")
    if kwargs:
        for k, v in kwargs.items():
            logger.debug(f"kwargs: kyes: {k}, value: {v}")

    if paths == "":
        msg = "Must provide a proper path."
        raise ValueError(msg)

    if isinstance(paths, str):
        if labels is None:
            labels = paths

        converted_paths = process_kpath(paths, cell, special_points=special_points)

        if not isinstance(labels[0], list):
            labels = [labels]

        labels = [list(label) for label in labels]
        paths = list(zip(labels, converted_paths))

    fontsize = kwargs.pop("fontsize", 14)

    if offset is None:
        offset = {}

    two_d = True
    try:
        ax.get_zlim()
        two_d = False
    except AttributeError:
        pass

    for names, points in paths:
        x, y, z = np.array(points).T

        if two_d:
            ax.plot(x, y, c="r", ls="-", **kwargs)
        else:
            ax.plot(x, y, z, c="r", ls="-", **kwargs)

        for name, point in zip(names, points):
            x, y, z = point
            display_name = name
            if display_name == "G":
                display_name = "\\Gamma"
            elif len(name) > 1:
                name = name[0] + "_" + name[1]

            off = offset.get(name, 0)
            try:
                if two_d:
                    x, y = x + off[0], y + off[1]
                else:
                    x, y, z = x + off[0], y + off[1], z + off[2]

            except TypeError:
                if two_d:
                    x, y = x + off, y + off
                else:
                    x, y, z = x + off, y + off, z + off

            if two_d:
                ax.text(
                    x,
                    y,
                    "$" + display_name + "$",
                    ha="center",
                    va="bottom",
                    color="r",
                    fontsize=fontsize,
                )
            else:
                ax.text(
                    x,
                    y,
                    z,
                    "$" + display_name + "$",
                    ha="center",
                    va="bottom",
                    color="r",
                    fontsize=fontsize,
                )


def bz2d_segments(
    cell: Sequence[Sequence[float]],
    transformations: list[Transformation] | None = None,
) -> tuple[list[NDArray[np.float_]], list[NDArray[np.float_]]]:
    """Calculates the line segments corresponding to a 2D BZ."""
    segments_x = []
    segments_y = []

    for points, _normal in twocell_to_bz1(cell)[0]:
        points = apply_transformations(points, transformations)
        x, y, z = np.concatenate([points, points[:1]]).T
        segments_x.append(x)
        segments_y.append(y)

    return segments_x, segments_y


def twocell_to_bz1(cell):
    from ase.dft.bz import bz_vertices

    # 2d in x-y plane
    if len(cell) > 2:  # noqa: PLR2004
        assert all(abs(cell[2][0:2]) < 1e-6)  # noqa: PLR2004
        assert all(abs(cell.T[2][0:2]) < 1e-6)  # noqa: PLR2004
    else:
        cell = [[*list(c), 0] for c in cell] + [[0, 0, 1]]
    icell = np.linalg.inv(cell).T
    try:
        bz1 = bz_vertices(icell[:3, :3], dim=2)
    except TypeError:
        bz1 = bz_vertices(icell[:3, :3])
    return bz1, icell, cell


def bz2d_plot(
    cell: Sequence[Sequence[float]],
    paths: str | list[float] | None = None,
    points: Sequence[float] | None = None,
    repeat: tuple[int, int] | None = None,
    ax: Axes | None = None,
    transformations: list[Transformation] | None = None,
    offset: dict[str, float | Sequence[float]] | None = None,
    *,
    hide_ax: bool = True,
    vectors: bool = False,
    set_equal_aspect: bool = True,
    **kwargs: Incomplete,
) -> Axes:
    """This piece of code modified from ase.ase.dft.bz.py:bz2d_plot.

    It follows copyright and license for ASE.

    Plots a Brillouin zone corresponding to a given unit cell
    """
    logger.debug(f"cell: {cell}")
    logger.debug(f"paths: {paths}")
    logger.debug(f"points: {points}")
    logger.debug(f"repeat: {repeat}")
    logger.debug(f"transformations: {transformations}")
    logger.debug(f"hide_ax: {hide_ax}")
    logger.debug(f"vectors: {vectors}")
    logger.debug(f"set_equal_aspect: {set_equal_aspect}")
    if kwargs:
        for k, v in kwargs.items():
            logger.debug(f"kwargs: kyes: {k}, value: {v}")
    kpoints = points
    bz1, icell, cell = twocell_to_bz1(cell)
    if ax is None:
        ax = plt.axes()

    if isinstance(paths, str):
        from ase.cell import Cell
        from ase.dft.kpoints import parse_path_string

        cell_structure = Cell(cell).get_bravais_lattice()
        special_points = cell_structure.get_special_points()
        path_string = cell_structure.special_path if paths == "all" else paths
        paths = []
        for names in parse_path_string(path_string):
            points = []
            for name in names:
                points.append(np.dot(icell.T, special_points[name]))
                paths.append((names, points))

    maxp = 0.0
    kwargs.setdefault("color", "black")
    kwargs.setdefault("linestyle", "-")
    for points, _normal in bz1:
        transformed_points = apply_transformations(points, transformations)
        x, y, z = np.concatenate([transformed_points, transformed_points[:1]]).T

        ax.plot(x, y, **kwargs)
        maxp = max(maxp, points.max())

    rep_x: int | tuple[int, int]
    rep_y: int | tuple[int, int]
    if repeat is not None:
        dx, dy = icell[0], icell[1]

        rep_x, rep_y = repeat
        if isinstance(rep_x, int):
            rep_x = (0, rep_x)
        if isinstance(rep_y, int):
            rep_y = (0, rep_y)

        for ix, iy in itertools.product(range(*rep_x), range(*rep_y)):
            delta = dx * ix + dy * iy

            for points, _normal in bz1:
                x, y, z = np.concatenate([points, points[:1]]).T
                x, y = x + delta[0], y + delta[1]
                x, y, z = apply_transformations(np.asarray([x, y, z]).T, transformations).T

                c = kwargs.pop("c", "k")
                c = kwargs.pop("color", c)
                ax.plot(x, y, **kwargs)
                maxp = max(maxp, points.max())

    if vectors:
        ax.arrow(
            0,
            0,
            icell[0, 0],
            icell[0, 1],
            lw=1,
            color="k",
            length_includes_head=True,
            head_width=0.03,
            head_length=0.05,
        )
        ax.arrow(
            0,
            0,
            icell[1, 0],
            icell[1, 1],
            lw=1,
            color="k",
            length_includes_head=True,
            head_width=0.03,
            head_length=0.05,
        )

    if paths is not None:
        annotate_special_paths(ax, paths, offset=offset, transformations=transformations)

    if kpoints is not None:
        for p in kpoints:
            ax.scatter(p[0], p[1], c="b")

    if hide_ax:
        ax.set_axis_off()
        ax.autoscale_view(tight=True)

    if set_equal_aspect:
        ax.set_aspect("equal")
    return ax
