"""Some common spatial plotting routines. Useful for contextualizing nanoARPES data."""

from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import gridspec, patches

from arpes.io import load_data
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.xarray import unwrap_xarray_item

from .annotations import annotate_point
from .utils import (
    ddata_daxis_units,
    fancy_labels,
    frame_with,
    path_for_plot,
    remove_colorbars,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from arpes._typing import DataType

__all__ = ("reference_scan_spatial", "plot_spatial_reference")


@save_plot_provenance
def plot_spatial_reference(
    reference_map: DataType,
    data_list: list[DataType],
    offset_list: list[dict[str, Any] | None] | None = None,
    annotation_list: list[str] | None = None,
    out: str | Path = "",
    *,
    plot_refs: bool = True,
) -> Path | tuple[Figure, list[Axes]]:
    """Helpfully plots data against a reference scanning dataset.

    This is essential to understand
    where data was taken and can be used early in the analysis phase in order to highlight the
    location of your datasets against core levels, etc.

    Args:
        reference_map: A scanning photoemission like dataset
        data_list: A list of datasets you want to plot the relative locations of
        offset_list: Optionally, offsets given as coordinate dicts
        annotation_list: Optionally, text annotations for the data
        out: Where to save the figure if we are outputting to disk
        plot_refs: Whether to plot reference figures for each of the pieces of data in `data_list`
    """
    if offset_list is None:
        offset_list = [None] * len(data_list)

    if annotation_list is None:
        annotation_list = [str(i + 1) for i in range(len(data_list))]

    normalize_to_spectrum(reference_map)

    n_references = len(data_list)
    if n_references == 1 and plot_refs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax_refs = [axes[1]]
    elif plot_refs:
        n_extra_axes = 1 + (n_references // 4)
        fig = plt.figure(figsize=(6 * (1 + n_extra_axes), 5))
        spec = gridspec.GridSpec(ncols=2 * (1 + n_extra_axes), nrows=2, figure=fig)
        ax = fig.add_subplot(spec[:2, :2])

        ax_refs = [
            fig.add_subplot(spec[i // (2 * n_extra_axes), 2 + i % (2 * n_extra_axes)])
            for i in range(n_references)
        ]
    else:
        ax_refs = []
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    with contextlib.suppress(Exception):
        reference_map = reference_map.S.spectra[0]

    reference_map = reference_map.S.mean_other(["x", "y", "z"])

    ref_dims = reference_map.dims[::-1]
    two_dimension = 2
    assert len(reference_map.dims) == two_dimension
    reference_map.S.plot(ax=ax, cmap="Blues")

    cmap = mpl.colormaps.get_cmap("Reds")
    rendered_annotations = []
    for i, (data, offset, annotation) in enumerate(
        zip(data_list, offset_list, annotation_list, strict=True),
    ):
        if offset is None:
            logical_offset = {}
            try:
                logical_offset["x"] = (
                    data.S.logical_offsets["x"] - reference_map.S.logical_offsets["x"]
                )
                logical_offset["y"] = (
                    data.S.logical_offsets["y"] - reference_map.S.logical_offsets["z"]
                )
                logical_offset["z"] = (
                    data.S.logical_offsets["y"] - reference_map.S.logical_offsets["z"]
                )
            except ValueError:
                logical_offset = {}
        else:
            logical_offset = offset

        coords = {c: unwrap_xarray_item(data.coords[c]) for c in ref_dims}
        n_array_coords = len(
            [cv for cv in coords.values() if isinstance(cv, np.ndarray | xr.DataArray)],
        )
        color = cmap(0.4 + (0.5 * i / len(data_list)))
        x = coords[ref_dims[0]] + logical_offset.get(str(ref_dims[0]), 0)
        y = coords[ref_dims[1]] + logical_offset.get(str(ref_dims[1]), 0)
        ref_x, ref_y = x, y
        off_x, off_y = 0, 0
        scale = 0.03

        if n_array_coords == 0:
            off_y = 1
            ax.scatter([x], [y], s=60, color=color)
        if n_array_coords == 1:
            if isinstance(x, np.ndarray | xr.DataArray):
                y = [y] * len(x)
                ref_x = np.min(x)
                off_x = -1
            else:
                x = [x] * len(y)
                ref_y = np.max(y)
                off_y = 1

            ax.plot(x, y, color=color, linewidth=3)
        if n_array_coords == two_dimension:
            off_y = 1
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            ref_x, ref_y = min_x, max_y

            color = cmap(0.4 + (0.5 * i / len(data_list)), alpha=0.5)
            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, facecolor=color)
            color = cmap(0.4 + (0.5 * i / len(data_list)))

            ax.add_patch(rect)

        dp = ddata_daxis_units(ax)
        text_location = np.asarray([ref_x, ref_y]) + dp * scale * np.asarray([off_x, off_y])
        text = ax.annotate(annotation, text_location, color="black", size=15)
        rendered_annotations.append(text)
        text.set_path_effects(
            [path_effects.Stroke(linewidth=2, foreground="white"), path_effects.Normal()],
        )
        if plot_refs:
            ax_ref = ax_refs[i]
            keep_preference = [
                *list(ref_dims),
                "eV",
                "temperature",
                "kz",
                "hv",
                "kp",
                "kx",
                "ky",
                "phi",
                "theta",
                "beta",
                "pixel",
            ]
            keep = [d for d in keep_preference if d in data.dims][:2]
            data.S.mean_other(keep).S.plot(ax=ax_ref)
            ax_ref.set_title(annotation)
            fancy_labels(ax_ref)
            frame_with(ax_ref, color=color, linewidth=3)

    ax.set_title("")
    remove_colorbars()
    fancy_labels(ax)
    plt.tight_layout()

    try:
        from adjustText import adjust_text

        adjust_text(
            rendered_annotations,
            ax=ax,
            avoid_points=False,
            avoid_objects=False,
            avoid_self=False,
            autoalign="xy",
        )
    except ImportError:
        pass

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, [ax, *ax_refs]


@save_plot_provenance
def reference_scan_spatial(
    data: DataType,
    out: str | Path = "",
) -> Path | tuple[Figure, NDArray[np.object_]]:
    """Plots the spatial content of a dataset, useful as a quick reference.

    Warning: Not work correctly.  (Because S.referenced_scans has been removed.)
    """
    data_arr = normalize_to_spectrum(data)

    assert isinstance(data_arr, xr.DataArray)

    dims = [d for d in data_arr.dims if d in {"cycle", "phi", "eV"}]

    summed_data = data_arr.sum(dims, keep_attrs=True)

    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    flat_axes = list(itertools.chain(*ax))

    summed_data.plot(ax=flat_axes[0])
    flat_axes[0].set_title(r"Full \textbf{eV} range")

    dims_except_eV = [d for d in dims if d != "eV"]
    summed_data = data_arr.sum(dims_except_eV)

    mul = 0.2
    rng = data_arr.coords["eV"].max().item() - data_arr.coords["eV"].min().item()
    offset = data_arr.coords["eV"].max().item()
    if offset > 0:
        offset = 0

    if rng > 3:
        mul = rng / 5.0

    for i in range(5):
        low_e, high_e = -mul * (i + 1) + offset, -mul * i + offset
        title = r"\textbf{eV}" + f": {low_e:.2g} to {high_e:.2g}"
        summed_data.sel(eV=slice(low_e, high_e)).sum("eV").plot(ax=flat_axes[i + 1])
        flat_axes[i + 1].set_title(title)

    y_range = flat_axes[0].get_ylim()
    x_range = flat_axes[0].get_xlim()
    delta_one_percent = ((x_range[1] - x_range[0]) / 100, (y_range[1] - y_range[0]) / 100)

    smart_delta = (2 * delta_one_percent[0], -1.5 * delta_one_percent[0])

    referenced = data_arr.S.referenced_scans

    # idea here is to collect points by those that are close together, then
    # only plot one annotation
    condensed = []
    cutoff = 3  # 3 percent
    for index, _ in referenced.iterrows():
        ff = load_data(index)

        x, y, _ = ff.S.sample_pos
        found = False
        for cx, cy, cl in condensed:
            if abs(cx - x) < cutoff * abs(delta_one_percent[0]) and abs(cy - y) < cutoff * abs(
                delta_one_percent[1],
            ):
                cl.append(index)
                found = True
                break

        if not found:
            condensed.append((x, y, [index]))

    for fax in flat_axes:
        for cx, cy, cl in condensed:
            annotate_point(
                fax,
                (
                    cx,
                    cy,
                ),
                ",".join([str(_) for _ in cl]),
                delta=smart_delta,
                fontsize="large",
            )

    plt.tight_layout()

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax
