"""Analysis about the spectrum edges."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from skimage import feature

from arpes.analysis import rebin
from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger
from arpes.xarray_extensions.accessor.general import (
    GenericDataArrayAccessor,  # pylint: disable=unused-import, redefined-outer-name  # noqa: F401
    GenericDatasetAccessor,  # pylint: disable=unused-import, redefined-outer-name  # noqa: F401
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def find_spectrum_energy_edges(
    data: xr.DataArray,
    *,
    indices: bool = False,
) -> NDArray[np.floating] | NDArray[np.int_]:
    """Compute the angular edges of the spectrum over the specified energy range.

    This method identifies the low and high angular edges for each slice of the spectrum
    within a given energy range. The energy range is divided into slices using the specified
    `energy_division`. For each slice, edges are detected using the Canny edge detection
    algorithm after applying Gaussian smoothing.

    Args:
        data (xr.DataArray): ARPES data, "pixel" coordinates can be used.
        indices (bool, optional):
            If `True`, returns the edge positions as indices. If `False`, returns the
            edge positions as physical coordinates. Defaults to `False`.
        energy_division (float, optional):
            The step size for dividing the energy range. Smaller values provide finer
            resolution for edge detection. Defaults to 0.05.

    Returns:
        tuple[NDArray[np.floating], NDArray[np.floating], xr.DataArray]:
            - If `indices=True`:
                - Low edge indices.
                - High edge indices.
                - Corresponding energy coordinates.
            - If `indices=False`:
                - Low edge physical coordinates.
                - High edge physical coordinates.
                - Corresponding energy coordinates.

    Raises:
        ValueError: If the energy range is too narrow for proper edge detection.

    Example:
        ```python
        # Assuming `data` is an xarray.DataArray with "eV" and "phi" or "pixel" dimensions
        low_edges, high_edges, energy_coords = data.find_spectrum_angular_edges_full(
            indices=False, energy_division=0.1
        )

        print("High edges:", high_edges)
        print("Low edges:", low_edges)
        print("Energy coordinates:", energy_coords)
        ```

    Todo:
        - Add unit tests for edge cases and different data configurations.
        - Investigate optimal parameters for edge detection.
            (e.g., Gaussian filter size, thresholds).
    """
    assert isinstance(data, xr.DataArray)
    energy_marginal = data.sum([d for d in data.dims if d != "eV"])

    embed_size = 20
    embedded: NDArray[np.floating] = np.ndarray(shape=[embed_size, energy_marginal.sizes["eV"]])
    embedded[:] = energy_marginal.values
    embedded = ndi.gaussian_filter(embedded, embed_size / 3)

    edges = feature.canny(
        embedded,
        sigma=embed_size / 5,
        use_quantiles=True,
        low_threshold=0.1,
    )
    edges = np.where(edges[int(embed_size / 2)] == 1)[0]
    if indices:
        return edges

    delta = data.G.stride(generic_dim_names=False)
    return edges * delta["eV"] + data.coords["eV"].values[0]


def find_spectrum_angular_edges(
    data: xr.DataArray,
    *,
    angle_name: str = "phi",
    indices: bool = False,
) -> NDArray[np.floating] | NDArray[np.int_]:
    """Return angle position corresponding to the (1D) spectrum edge.

    Args:
        data (xr.DataArray): ARPES data, "pixel" coordinates can be used.
        angle_name (str): Angle name to find the edge
        indices (bool):  If True, return the index not the angle value.

    Returns: NDArray[np.floating] | NDArray[np.int_]
        Angle position
    """
    angular_dim: str = "pixel" if "pixel" in data.dims else angle_name
    assert isinstance(data, xr.DataArray)
    phi_marginal = data.sum(
        [d for d in data.dims if d != angular_dim],
    )

    embed_size = 20
    embedded: NDArray[np.floating] = np.ndarray(
        shape=[embed_size, phi_marginal.sizes[angular_dim]],
    )
    embedded[:] = phi_marginal.values
    embedded = ndi.gaussian_filter(embedded, embed_size / 3)

    # try to avoid dependency conflict with numpy v0.16

    edges = feature.canny(
        image=embedded,
        sigma=embed_size / 5,
        use_quantiles=True,
        low_threshold=0.2,
    )
    edges = np.where(edges[int(embed_size / 2)] == 1)[0]
    if indices:
        return edges

    delta = data.G.stride(generic_dim_names=False)
    return edges * delta[angular_dim] + data.coords[angular_dim].values[0]


def find_spectrum_angular_edges_full(
    data: xr.DataArray,
    *,
    indices: bool = False,
    energy_division: float = 0.05,
) -> tuple[NDArray[np.floating], NDArray[np.floating], xr.DataArray]:
    """Finds the angular edges of the spectrum based on energy slicing and rebinning.

    This method uses edge detection techniques to identify boundaries in the angular dimension.

    Args:
        data (xr.DataArray): ARPES data, "pixel" coordinates can be used.
        indices (bool, optional): If True, returns edge indices; if False, returns physical
            angular coordinates. Defaults to False.
        energy_division (float, optional): Specifies the energy division step for rebinning.
            Defaults to 0.05 eV.

    Returns:
        tuple: A tuple containing:
            - low_edges (NDArray[np.floating]): Values or indices of the low edges
                of the spectrum.
            - high_edges (NDArray[np.floating]): Values or indices of the high edges
                of the spectrum.
            - eV_coords (xr.DataArray): The coordinates of the rebinned energy axis.

    Todo:
        - Add unit tests for this function.
    """
    # as a first pass, we need to find the bottom of the spectrum, we will use this
    # to select the active region and then to rebin into course steps in energy from 0
    # down to this region
    # we will then find the appropriate edge for each slice, and do a fit to the edge locations
    energy_edge = find_spectrum_energy_edges(data)
    low_edge: np.floating = np.min(energy_edge) + energy_division
    high_edge: np.floating = np.max(energy_edge) - energy_division

    if high_edge - low_edge < 3 * energy_division:
        # Doesn't look like the automatic inference of the energy edge was valid
        high_edge = data.coords["eV"].max().item()
        low_edge = data.coords["eV"].min().item()

    angular_dim = "pixel" if "pixel" in data.dims else "phi"
    energy_cut = data.sel(eV=slice(low_edge, high_edge)).S.sum_other(["eV", angular_dim])

    n_cuts = int(np.ceil((high_edge - low_edge) / energy_division))
    new_shape = {"eV": n_cuts}
    new_shape[angular_dim] = energy_cut.sizes[angular_dim]
    logger.debug(f"new_shape: {new_shape}")
    rebinned = rebin(energy_cut, shape=new_shape)

    embed_size = 20
    embedded: NDArray[np.floating] = np.empty(
        shape=[embed_size, rebinned.sizes[angular_dim]],
    )
    low_edges = []
    high_edges = []
    for e_cut_index in range(rebinned.sizes["eV"]):
        e_slice = rebinned.isel(eV=e_cut_index)
        embedded[:] = e_slice.values
        embedded = ndi.gaussian_filter(embedded, embed_size / 1.5)  # < = Why 1.5

        edges: NDArray[np.bool_] = feature.canny(
            image=embedded,
            sigma=4,
            use_quantiles=False,
            low_threshold=0.7,
            high_threshold=1.5,
        )
        edges = np.where(edges[int(embed_size / 2)] == 1)[0]
        low_edges.append(np.min(edges))
        high_edges.append(np.max(edges))

    if indices:
        return np.array(low_edges), np.array(high_edges), rebinned.coords["eV"]

    delta = data.G.stride(generic_dim_names=False)

    return (
        np.array(low_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0],
        np.array(high_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0],
        rebinned.coords["eV"],
    )


def zero_spectrometer_edges(
    data: xr.DataArray,
    cut_margin: int = 0,
    interp_range: float | None = None,
    low: Sequence[float] | NDArray[np.floating] | None = None,
    high: Sequence[float] | NDArray[np.floating] | None = None,
) -> xr.DataArray:
    """Zeros out the spectrum data outside of the specified low and high edges.

    It uses the provided or inferred edge information, applying cut margins and optionally
    interpolating over a given range.

    Args:
        data (xr.DataArray): The spectrum data to process.
        cut_margin (int or float, optional): Margin to apply when invalidating data near edges.
            Use `int` for pixel-based margins or `float` for angular physical units.
            Defaults to 50 pixels or 0.08 in angular units, depending on the data type.
        interp_range (float or None, optional): Specifies the interpolation range for edge data.
            If provided, the edge values are interpolated within this range.
        low (Sequence[float], NDArray[np.floating], or None, optional): Low edge values.
            Use this to manually specify the low edge. Defaults to None.
            (automatically determined).
        high (Sequence[float], NDArray[np.floating], or None, optional): High edge values.
            Use this to manually specify the high edge. Defaults to None.
            (automatically determined).

    Returns:
        xr.DataArray: The spectrum data with values outside the edges set to zero.

    Todo:
        - Add tests.

    """
    assert isinstance(data, xr.DataArray)
    if low is not None:
        assert high is not None
        assert len(low) == len(high) == TWO_DIMENSION

        low_edges = low
        high_edges = high

    (
        low_edges,
        high_edges,
        rebinned_eV_coord,
    ) = find_spectrum_angular_edges_full(data, indices=True)

    angular_dim = "pixel" if "pixel" in data.dims else "phi"
    if not cut_margin:
        if "pixel" in data.dims:
            cut_margin = 50
        else:
            cut_margin = int(0.08 / data.G.stride(generic_dim_names=False)[angular_dim])
    elif isinstance(cut_margin, float):
        assert angular_dim == "phi"
        cut_margin = int(
            cut_margin / data.G.stride(generic_dim_names=False)[angular_dim],
        )

    if interp_range is not None:
        low_edge = xr.DataArray(low_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
        high_edge = xr.DataArray(high_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
        low_edge = low_edge.sel(eV=interp_range, method="nearest")
        high_edge = high_edge.sel(eV=interp_range, method="nearest")
    other_dims = list(data.dims)
    other_dims.remove("eV")
    other_dims.remove(angular_dim)
    copied = data.copy(deep=True).transpose("eV", angular_dim, ...)

    low_edges += cut_margin
    high_edges -= cut_margin

    for i, energy in enumerate(copied.coords["eV"].values):
        index = np.searchsorted(rebinned_eV_coord, energy)
        other = index + 1
        if other >= len(rebinned_eV_coord):
            other = len(rebinned_eV_coord) - 1
            index = len(rebinned_eV_coord) - 2

        low_index = int(np.interp(energy, rebinned_eV_coord, low_edges))
        high_index = int(np.interp(energy, rebinned_eV_coord, high_edges))
        copied.values[i, 0:low_index] = 0
        copied.values[i, high_index:-1] = 0

    return copied
