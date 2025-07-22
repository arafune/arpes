"""For plotting band locations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import matplotlib.pyplot as plt
from matplotlib import colorbar
from matplotlib.axes import Axes

from arpes.provenance import save_plot_provenance

from .utils import label_for_colorbar, path_for_plot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.image import AxesImage

    from arpes._typing.base import XrTypes
    from arpes._typing.plotting import PColorMeshKwargs
    from arpes.models.band import Band

__all__ = ("plot_with_bands",)


@save_plot_provenance
def plot_with_bands(
    data: XrTypes,
    bands: Sequence[Band],
    title: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    **kwargs: Unpack[PColorMeshKwargs],
) -> Path | Axes:  # <== CHECKME the type may be NDArray[np.object_]
    """Makes a dispersion plot with bands overlaid.

    Args:
        data (xr.DataArray): ARPES experimental data.
        bands (Sequence[Band]): Collection of bands to overlay on the plot.
        title (str, optional): Title of the plot. Defaults to the label of `data.S`.
        ax (Axes, optional): Matplotlib axis to plot on. If None, a new figure is created.
        out (str or Path, optional): File path to save the plot. If empty, the plot is shown
            interactively.
        kwargs: Additional keyword arguments passed to `data.plot()`.

    Returns:
        Union[Path, Axes]: File path if `out` is specified; otherwise, the Matplotlib axis object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    assert isinstance(ax, Axes)

    if not title:
        title = data.S.label.replace("_", " ")

    mesh: AxesImage = data.S.plot(ax=ax, **kwargs)
    mesh_colorbar = mesh.colorbar
    assert isinstance(mesh_colorbar, colorbar.Colorbar)
    mesh_colorbar.set_label(label_for_colorbar(data))

    if data.S.is_differentiated:
        mesh.set_cmap("Blues")

    for band in bands:
        plt.scatter(band.center.values, band.coords[band.dims[0]].values)

    if out:
        filename = path_for_plot(out)
        plt.savefig(filename)
        return filename

    plt.show()
    return ax
