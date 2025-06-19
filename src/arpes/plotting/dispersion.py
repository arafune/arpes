"""Plotting routines related to 2D ARPES cuts and dispersions."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from arpes.io import load_data
from arpes.preparation import normalize_dim
from arpes.provenance import save_plot_provenance
from arpes.utilities import bz
from arpes.utilities.conversion import remap_coords_to, slice_along_path

from .utils import label_for_colorbar, label_for_dim, label_for_symmetry_point, path_for_plot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure, FigureBase
    from numpy.typing import NDArray

    from arpes._typing import PColorMeshKwargs, XrTypes
    from arpes.models.band import Band

__all__ = (
    "cut_dispersion_plot",
    "fancy_dispersion",
    "hv_reference_scan",
    "labeled_fermi_surface",
    "plot_dispersion",
    "reference_scan_fermi_surface",
    "scan_var_reference_plot",
)


@save_plot_provenance
def plot_dispersion(
    spectrum: xr.DataArray,
    bands: Sequence[Band],
    out: str | Path = "",
) -> Axes | Path:
    """Plots an ARPES cut with bands over it."""
    ax = spectrum.S.plot()

    for band in bands:
        plt.scatter(band.center.values, band.coords[band.dims[0]].values)

    if out:
        filename = path_for_plot(out)
        plt.savefig(filename)
        return filename
    return ax


class CutDispersionPlotParam(TypedDict, total=False):
    cmap: Colormap | str | None
    title: str


@save_plot_provenance
def cut_dispersion_plot(  # noqa: PLR0913, PLR0915
    data: xr.DataArray,
    e_floor: float | None = None,
    ax: Axes3D | None = None,
    *,
    include_symmetry_points: bool = True,
    out: str | Path = "",
    quality: Literal["paper", "high", "low"] = "high",
    **kwargs: Unpack[CutDispersionPlotParam],
) -> Path | None:
    """Makes a 3D cut dispersion plot.

    At the moment this only supports rectangular BZs.

    Args:
        data: The 3D data to plot
        e_floor: The energy of the bottom cut "floor"
        title: A title for the plot
        ax: The axes to plot to
        include_symmetry_points: Whether to include annotated symmetry points
        out: Where to save the file, optional
        quality: Controls output figure DPI
        kwargs: pass to
    """
    # to get nice labeled edges you could use shapely
    sampling = {
        "paper": 400,
        "high": 100,
        "low": 40,
    }.get(quality, 100)

    assert "eV" in data.dims
    assert e_floor is not None

    new_dim_order = list(data.dims)
    new_dim_order.remove("eV")
    new_dim_order = [*new_dim_order, "eV"]
    data = data.transpose(*new_dim_order)

    # prep data to be used for rest of cuts
    lower_part = data.sel(eV=slice(None, 0))
    floor = lower_part.S.fat_sel(eV=e_floor)

    bz_mask: NDArray[np.float64] = bz.reduced_bz_mask(data=lower_part, scale_zone=True)
    left_mask: NDArray[np.float64] = bz.reduced_bz_E_mask(
        data=lower_part,
        symbol="X",
        e_cut=e_floor,
        scale_zone=True,
    )
    right_mask: NDArray[np.float64] = bz.reduced_bz_E_mask(
        data=lower_part,
        symbol="Y",
        e_cut=e_floor,
        scale_zone=True,
    )

    def mask_for(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return left_mask if x.shape == left_mask.shape else right_mask

    x_dim, y_dim, z_dim = tuple(new_dim_order)
    x_coords, y_coords, _ = (
        data.coords[x_dim],
        data.coords[y_dim],
        data.coords[z_dim],
    )

    if ax is None:
        fig: FigureBase = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    assert isinstance(ax, Axes3D)

    kwargs.setdefault(
        "title",
        "{} Cut Through Symmetry Points".format(data.S.label.replace("_", " ")),
    )
    kwargs.setdefault("cmap", plt.get_cmap("Blues"))
    title = kwargs.pop("title")
    ax.set_title(title)

    colormap = plt.get_cmap("Blues")

    # color fermi surface
    fermi_surface = data.S.fat_sel(eV=0.0)
    Xs, Ys = np.meshgrid(x_coords, y_coords)

    Zs = np.zeros(fermi_surface.data.shape)
    Zs[bz_mask] = np.nan
    scale_colors = max(np.max(fermi_surface.data), np.max(floor.data))
    ax.plot_surface(
        X=Xs,
        Y=Ys,
        Z=Zs.T,
        facecolors=colormap(fermi_surface.data.T / scale_colors),
        shade=False,
        vmin=-1,
        vmax=1,
        antialiased=True,
        rcount=sampling,
        ccount=sampling,
    )

    # color right edge
    right_sel = {}
    edge_val = np.max(lower_part.coords[x_dim].data)
    right_sel[x_dim] = edge_val
    right_edge = lower_part.S.fat_sel(**right_sel)

    Ys, Zs = np.meshgrid(lower_part.coords[y_dim], lower_part.coords[z_dim])
    Xs = np.ones(shape=right_edge.shape) * edge_val
    Xs[mask_for(Xs)] = np.nan
    ax.plot_surface(
        X=Xs.T,
        Y=Ys,
        Z=Zs,
        facecolors=colormap(right_edge.data.T / scale_colors),
        vmin=-1,
        vmax=1,
        shade=False,
        antialiased=True,
        rcount=sampling,
        ccount=sampling,
    )

    # color left edge
    left_sel = {}
    edge_val = np.min(lower_part.coords[y_dim].data)
    left_sel[y_dim] = edge_val
    left_edge = lower_part.S.fat_sel(**left_sel)

    Xs, Zs = np.meshgrid(lower_part.coords[x_dim], lower_part.coords[z_dim])
    Ys = np.ones(left_edge.shape) * edge_val
    Ys[mask_for(Ys)] = np.nan
    ax.plot_surface(
        X=Xs,
        Y=Ys.T,
        Z=Zs,
        facecolors=colormap(left_edge.data.T / scale_colors),
        vmin=-1,
        vmax=1,
        antialiased=True,
        shade=False,
        rcount=sampling,
        ccount=sampling,
    )

    # selection region
    # floor
    Xs, Ys = np.meshgrid(floor.coords[x_dim], floor.coords[y_dim])
    Zs = np.ones(floor.data.shape) * e_floor
    Zs[np.logical_not(bz_mask)] = np.nan
    ax.plot_surface(
        X=Xs,
        Y=Ys,
        Z=Zs.T,
        facecolors=colormap(floor.data.T / scale_colors),
        vmin=-1,
        vmax=1,
        shade=False,
        antialiased=True,
        rcount=sampling,
        ccount=sampling,
    )

    # determine the axis along X, Y
    axis_X = bz.axis_along(data, "X")
    axis_Y = bz.axis_along(data, "Y")

    # left and right inset faces
    inset_face = slice_along_path(
        arr=lower_part,
        interpolation_points=["G", "X"],
        axis_name=axis_X,
        extend_to_edge=True,
    ).sel(
        eV=slice(e_floor, None),
    )
    Xs, Zs = np.meshgrid(inset_face.coords[axis_X], inset_face.coords[z_dim])
    Ys = np.ones(inset_face.data.shape)
    if y_dim == axis_X:
        Ys *= data.S.phi_offset
        Xs += data.S.map_angle_offset
        Xs, Ys = Ys, Xs
    else:
        Ys *= data.S.map_angle_offset
        Xs += data.S.phi_offset
    ax.plot_surface(
        X=Xs,
        Y=Ys,
        Z=Zs,
        facecolors=colormap(inset_face.data / scale_colors),
        shade=False,
        antialiased=True,
        zorder=1,
        rcount=sampling,
        ccount=sampling,
    )
    inset_face = slice_along_path(
        arr=lower_part,
        interpolation_points=["G", "Y"],
        axis_name=axis_Y,
        extend_to_edge=True,
    ).sel(
        eV=slice(e_floor, None),
    )
    Ys, Zs = np.meshgrid(inset_face.coords[axis_Y], inset_face.coords[z_dim])
    Xs = np.ones(inset_face.data.shape)
    if x_dim == axis_Y:
        Xs *= data.S.map_angle_offset
        Ys += data.S.phi_offset
        Xs, Ys = Ys, Xs
    else:
        Xs *= data.S.phi_offset
        Ys += data.S.map_angle_offset

    ax.plot_surface(
        X=Xs,
        Y=Ys,
        Z=Zs,
        facecolors=colormap(inset_face.data / scale_colors),
        shade=False,
        antialiased=True,
        zorder=1,
        rcount=sampling,
        ccount=sampling,
    )

    ax.set_xlabel(label_for_dim(data, x_dim))
    ax.set_ylabel(label_for_dim(data, y_dim))
    ax.set_zlabel(label_for_dim(data, z_dim))

    zlim = ax.get_zlim3d()
    if include_symmetry_points:
        for point_name, point_location in data.S.iter_symmetry_points:
            coords = [point_location.get(d, 0.02) for d in new_dim_order]
            ax.scatter(*zip(coords, strict=True), marker=".", color="red", zorder=1000)
            coords[new_dim_order.index("eV")] += 0.1
            ax.text(
                *coords,
                label_for_symmetry_point(point_name),
                color="red",
                ha="center",
                va="top",
            )

    ax.set_zlim3d(*zlim)
    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()
    return None


@save_plot_provenance
def hv_reference_scan(
    data: XrTypes,
    e_cut: float = -0.05,
    bkg_subtraction: float = 0.8,
    **kwargs: Unpack[LabeledFermiSurfaceParam],
) -> Path | Axes:
    """A reference plot for photon energy scans. Used internally by other code."""
    fs = data.S.fat_sel(eV=e_cut)
    fs = normalize_dim(fs, "hv", keep_id=True)
    fs.data -= bkg_subtraction * np.mean(fs.data)
    fs.data[fs.data < 0] = 0

    out = kwargs.pop("out", None)

    lfs = labeled_fermi_surface(fs, **kwargs)
    assert isinstance(lfs, tuple)
    _, ax = lfs

    all_scans = data.attrs["df"]
    all_scans = all_scans[all_scans.id != data.attrs["id"]]
    all_scans = all_scans[
        (all_scans.spectrum_type != "xps_spectrum") | (all_scans.spectrum_type == "hv_map")
    ]

    scans_by_hv = defaultdict(list)
    for _, row in all_scans.iterrows():
        scan = load_data(row.id)

        scans_by_hv[round(scan.S.hv)].append(scan.S.label.replace("_", " "))

    dim_order = [ax.get_xlabel(), ax.get_ylabel()]
    handles = []
    handle_labels = []

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors_cycle = prop_cycle.by_key()["color"]

    for line_color, (hv, labels) in zip(colors_cycle, scans_by_hv.items(), strict=False):
        full_label = "\n".join(labels)

        # determine direction
        if dim_order[0] == "hv":
            # photon energy is along the x axis, we want an axvline
            handles.append(ax.axvline(hv, label=full_label, color=line_color))
        else:
            # photon energy is along the y axis, we want an axhline
            handles.append(ax.axhline(hv, label=full_label, color=line_color))

        handle_labels.append(full_label)

    plt.legend(handles, handle_labels)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()
    return ax


class LabeledFermiSurfaceParam(TypedDict, total=False):
    include_symmetry_points: bool
    include_bz: bool
    fermi_energy: float
    out: str | Path


@save_plot_provenance
def reference_scan_fermi_surface(
    data: xr.DataArray,
    **kwargs: Unpack[LabeledFermiSurfaceParam],
) -> Path | Axes:
    """A reference plot for Fermi surfaces. Used internally by other code.

    Warning: Not work correctly.  (Because S.referenced_scans has been removed.)
    """
    fs = data.S.fat_sel(eV=0)

    out = kwargs.pop("out", None)
    lfs = labeled_fermi_surface(fs, **kwargs)
    assert isinstance(lfs, tuple)
    _, ax = lfs

    referenced_scans = data.S.referenced_scans
    handles = []
    for index, row in referenced_scans.iterrows():
        scan = load_data(row.id)

        remapped_coords = remap_coords_to(scan, data)
        dim_order = [ax.get_xlabel(), ax.get_ylabel()]
        ls = ax.plot(
            remapped_coords[dim_order[0]],
            remapped_coords[dim_order[1]],
            label=index.replace("_", " "),
        )
        handles.append(ls[0])

    plt.legend(handles=handles)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return ax


@save_plot_provenance
def labeled_fermi_surface(  # noqa: PLR0913
    data: xr.DataArray,
    title: str = "",
    ax: Axes | None = None,
    *,
    include_symmetry_points: bool = True,
    include_bz: bool = True,
    out: str | Path = "",
    fermi_energy: float = 0,
) -> Path | tuple[Figure | None, Axes]:
    """Plots a Fermi surface with high symmetry points annotated onto it."""
    assert isinstance(data, xr.DataArray)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 7.0))
    assert isinstance(ax, Axes)

    if not title:
        title = "{} Fermi Surface".format(data.S.label.replace("_", " "))

    if "eV" in data.dims:
        data = data.S.fat_sel(eV=fermi_energy)

    mesh = data.S.plot(ax=ax)
    mesh.colorbar.set_label(label_for_colorbar(data))

    if data.S.is_differentiated:
        mesh.set_cmap("Blues")

    dim_order = [ax.get_xlabel(), ax.get_ylabel()]

    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel(label_for_dim(data, ax.get_ylabel()))
    ax.set_title(title)

    marker_color = "RdBu" if data.S.is_differentiated else "red"

    if include_bz:
        bz.bz_symmetry(data.S.iter_own_symmetry_points)

        warnings.warn("BZ region display not implemented.", stacklevel=2)

    if include_symmetry_points:
        for point_name, point_location in data.S.iter_symmetry_points:
            warnings.warn("Symmetry point locations are not k-converted", stacklevel=2)
            coords: Sequence[float] = [point_location[d] for d in dim_order]
            ax.plot(*coords, marker=".", color=marker_color)
            ax.annotate(
                label_for_symmetry_point(point_name),
                coords,
                color=marker_color,
                xycoords="data",
                textcoords="offset points",
                xytext=(0, -10),
                va="top",
                ha="center",
                fontsize=14,
            )

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax


@save_plot_provenance
def fancy_dispersion(
    data: xr.DataArray,
    title: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    *,
    include_symmetry_points: bool = True,
    **kwargs: Unpack[PColorMeshKwargs],
) -> Axes | Path:
    """Generates a 2D ARPES cut with additional annotations, useful for quick presentations.

    This function creates a plot of ARPES data with optional symmetry points and custom styling for
    quick visualization. It is designed to help create figures rapidly for presentations or reports.
    Symmetry points are annotated if `include_symmetry_points` is set to True.

    Args:
        data (xr.DataArray): ARPES data to plot.
        title (str): Title of the figure. If not provided, the title is derived from the dataset
            label.
        ax (Axes, optional): Matplotlib Axes object for plotting. If not provided, a new Axes is
            created.
        out (str | Path, optional): Output file path for saving the figure. If not provided, the
            figure is not saved.
        include_symmetry_points (bool): Whether to include symmetry points in the plot
            (default is True).
        kwargs: Additional keyword arguments passed to `xr.DataArray.plot()` for further
            customization.

    Returns:
        Axes | Path: The Axes object containing the plot, or the file path if the plot is saved.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    assert isinstance(ax, Axes)
    if not title:
        title = data.S.label.replace("_", " ")

    mesh = data.S.plot(ax=ax, **kwargs)
    mesh.colorbar.set_label(label_for_colorbar(data))

    if data.S.is_differentiated:
        mesh.set_cmap("Blues")

    original_x_label = ax.get_xlabel()
    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel(label_for_dim(data, ax.get_ylabel()))
    ax.set_title(title, fontsize=14)

    # This can probably be pulled out into a a helper
    marker_color = "RdBu" if data.S.is_differentiated else "red"
    if include_symmetry_points:
        for point_name, point_locations in data.attrs.get("symmetry_points", {}).items():
            assert isinstance(point_locations, list | tuple)
            for single_location in point_locations:
                coords = (
                    single_location[original_x_label],
                    ax.get_ylim()[1],
                )
                ax.plot(*coords, marker=11, color=marker_color)
                ax.annotate(
                    label_for_symmetry_point(point_name),
                    coords,
                    color=marker_color,
                    xycoords="data",
                    textcoords="offset points",
                    xytext=(0, -10),
                    va="top",
                    ha="center",
                )

    ax.axhline(0, color="red", alpha=0.8, linestyle="--", linewidth=1)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return ax


@save_plot_provenance
def scan_var_reference_plot(
    data: xr.DataArray,
    title: str = "",
    ax: Axes | None = None,
    norm: Normalize | None = None,
    out: str | Path = "",
) -> Axes | Path:
    """Generates a simple plot of a DataArray with appropriately labeled axes.

    This function is used internally by other scripts to quickly generate plots for DataArrays. It
    supports normalization and customization of axes labels and titles. The plot can optionally be
    saved to a file.

    Args:
        data (xr.DataArray): The input data to plot, typically a DataArray.
        title (str): The title of the plot. If not provided, it is derived from the DataArray label.
        ax (Axes, optional): The Matplotlib Axes object to plot on. If not provided, a new Axes is
            created.
        norm (Normalize, optional): Normalization to apply to the plot. Default is None.
        out (str | Path, optional): File path to save the plot. If not provided, the plot is not
            saved.

    Returns:
        Axes | Path: The Axes object containing the plot, or the file path if the plot is saved.
    """
    assert isinstance(data, xr.DataArray)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    assert isinstance(ax, Axes)
    if not title:
        title = data.S.label.replace("_", " ")

    plot = data.S.plot(norm=norm, ax=ax)
    plot.colorbar.set_label(label_for_colorbar(data))

    ax.set_xlabel(label_for_dim(data, ax.get_xlabel()))
    ax.set_ylabel(label_for_dim(data, ax.get_ylabel()))

    ax.set_title(title, font_size=14)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return ax
