"""Reference plots, for preliminary analysis."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import xarray as xr

from arpes.plotting.utils import fancy_labels
from arpes.preparation import normalize_dim
from arpes.utilities.conversion import convert_to_kspace

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


__all__ = ["make_overview", "make_reference_plots"]


def make_overview(
    data_all: Sequence[xr.DataArray],
    ncols: int = 3,
) -> tuple[Figure, list[Axes]]:
    """Build overview of the measured data.

    Args:
        data_all (list[xr.DataArray]): Summary of xr.DataArray
        ncols(int): number of columns

    Returns: tuple[Figure, list[Axes]]
        Overview of ARPES data.
    """
    assert isinstance(data_all, Sequence)
    num_figs = len(data_all)
    nrows = num_figs // ncols
    if num_figs % ncols:
        nrows += 1
    fig: Figure = plt.figure(figsize=(3 * ncols, 3 * nrows))
    ax: list[Axes] = []
    for i, spectrum in enumerate(data_all):
        ax.append(fig.add_subplot(nrows, ncols, i + 1))
        spectrum.S.transpose_to_front("eV").plot(
            ax=ax[i],
            add_labels=False,
            add_colorbar=False,
        )
        ax[i].text(
            0.01,
            0.91,
            f"ID:{spectrum.id}",
            color="white",
            transform=ax[i].transAxes,
        )
        fancy_labels(ax[i])
    return fig, ax


def make_reference_plots(df: pd.DataFrame, *, with_kspace: bool = False) -> None:
    """Makes standard reference plots for orienting oneself."""
    from areps.io import load_data

    try:
        df = df[df.spectrum_type != "xps_spectrum"]
    except TypeError:
        warnings.warn("Unable to filter out XPS files, did you attach spectra type?", stacklevel=2)

    # Make scans indicating cut locations
    for index, _row in df.iterrows():
        try:
            scan = load_data(index)

            if isinstance(scan, xr.Dataset):
                # make plot series normalized by current:
                scan.S.reference_plot(out=True)
            else:
                scan.S.reference_plot(out=True, use_id=False)

                if scan.S.spectrum_type == "spectrum":
                    # Also go and make a normalized version
                    normed = normalize_dim(scan, "phi")
                    normed.S.reference_plot(out=True, use_id=False, pattern="{}_norm_phi.png")

                    if with_kspace:
                        normalized = normalize_dim(scan, "hv")
                        kspace_converted = convert_to_kspace(normalized)
                        kspace_converted.S.reference_plot(
                            out=True,
                            use_id=False,
                            pattern="k_{}.png",
                        )

                        normed_k = normalize_dim(kspace_converted, "kp")
                        normed_k.S.reference_plot(
                            out=True,
                            use_id=False,
                            pattern="k_{}_norm_kp.png",
                        )

        except Exception:
            logger.exception(f"Cannot make plots for {index}")
