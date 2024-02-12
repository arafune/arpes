"""For plotting band locations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from arpes.provenance import save_plot_provenance

from .utils import label_for_colorbar, path_for_plot

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.image import AxesImage

    from arpes._typing import DataType, PColorMeshKwargs
    from arpes.models.band import Band

__all__ = ("plot_with_bands",)


@save_plot_provenance
def plot_with_bands(
    data: DataType,
    bands: Sequence[Band],
    title: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    **kwargs: Unpack[PColorMeshKwargs],
) -> Path | Axes:  # <== CHECKME the type may be NDArray[np.object_]
    """Makes a dispersion plot with bands overlaid.

    Args:
        data (DataType): ARPES experimental data
        bands: [TODO:description]
        title (str): title of the plot
        ax: [TODO:description]
        out: [TODO:description]
        kwargs: pass to data.plot()


    Returns:
        [TODO:description]
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    assert isinstance(ax, Axes)

    if not title:
        title = data.S.label.replace("_", " ")

    mesh: AxesImage = data.plot(ax=ax, **kwargs)
    mesh.colorbar.set_label(label_for_colorbar(data))

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
