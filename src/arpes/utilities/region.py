"""Defines common region selections used programmatically elsewhere."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np

__all__ = ["REGIONS", "DesignatedRegions", "normalize_region"]


class DesignatedRegions(Enum):
    """Commonly used regions which can be used to select data programmatically."""

    # Angular windows
    NARROW_ANGLE = 0  # Narrow central region in the spectrometer
    WIDE_ANGLE = 1  # Just inside edges of spectremter data
    TRIM_EMPTY = 2  # Edges of spectrometer data

    # Energy windows
    BELOW_EF = 10  # Everything below e_F
    ABOVE_EF = 11  # Everything above e_F
    EF_NARROW = 12  # Narrow cut around e_F
    MESO_EF = 13  # Comfortably below e_F, pun on mesosphere

    # Effective energy windows, determined by Canny edge detection
    BELOW_EFFECTIVE_EF = 20  # Everything below e_F
    ABOVE_EFFECTIVE_EF = 21  # Everything above e_F
    EFFECTIVE_EF_NARROW = 22  # Narrow cut around e_F
    MESO_EFFECTIVE_EF = 23  # Comfortably below effective e_F, pun on mesosphere


REGIONS = {
    "copper_prior": {
        "eV": DesignatedRegions.MESO_EFFECTIVE_EF,
    },
    # angular can refer to either 'pixels' or 'phi'
    "wide_angular": {
        # angular can refer to either 'pixels' or 'phi'
        "angular": DesignatedRegions.WIDE_ANGLE,
    },
    "narrow_angular": {
        "angular": DesignatedRegions.NARROW_ANGLE,
    },
}


def normalize_region(
    region: Literal["copper_prior", "wide_angular", "narrow_angular"]
    | dict[str, DesignatedRegions],
) -> dict[str, DesignatedRegions]:
    """Converts named regions to an actual region."""
    if isinstance(region, str):
        return REGIONS[region]

    if isinstance(region, dict):
        return region

    msg = "Region should be either a string (i.e. an ID/alias) or an explicit dictionary."
    raise TypeError(
        msg,
    )


def find_spectrum_energy_edges(
    self,
    *,
    indices: bool = False,
) -> NDArray[np.float64] | NDArray[np.int_]:
    """Compute the angular edges of the spectrum over the specified energy range.

    This method identifies the low and high angular edges for each slice of the spectrum
    within a given energy range. The energy range is divided into slices using the specified
    `energy_division`. For each slice, edges are detected using the Canny edge detection
    algorithm after applying Gaussian smoothing.

    Args:
        indices (bool, optional):
            If `True`, returns the edge positions as indices. If `False`, returns the
            edge positions as physical coordinates. Defaults to `False`.
        energy_division (float, optional):
            The step size for dividing the energy range. Smaller values provide finer
            resolution for edge detection. Defaults to 0.05.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], xr.DataArray]:
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
    assert isinstance(
        self._obj,
        xr.DataArray,
    )  # if self._obj is xr.Dataset, values is  function
    energy_marginal = self._obj.sum([d for d in self._obj.dims if d != "eV"])

    embed_size = 20
    embedded: NDArray[np.float64] = np.ndarray(shape=[embed_size, energy_marginal.sizes["eV"]])
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

    delta = self._obj.G.stride(generic_dim_names=False)
    return edges * delta["eV"] + self._obj.coords["eV"].values[0]


def find_spectrum_angular_edges(
    self,
    *,
    angle_name: str = "phi",
    indices: bool = False,
) -> NDArray[np.float64] | NDArray[np.int_]:
    """Return angle position corresponding to the (1D) spectrum edge.

    Args:
        angle_name (str): angle name to find the edge
        indices (bool):  if True, return the index not the angle value.

    Returns: NDArray[np.float64] | NDArray[np.int64]
        Angle position
    """
    angular_dim: str = "pixel" if "pixel" in self._obj.dims else angle_name
    assert isinstance(self._obj, xr.DataArray)
    phi_marginal = self._obj.sum(
        [d for d in self._obj.dims if d != angular_dim],
    )

    embed_size = 20
    embedded: NDArray[np.float64] = np.ndarray(
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

    delta = self._obj.G.stride(generic_dim_names=False)
    return edges * delta[angular_dim] + self._obj.coords[angular_dim].values[0]


def find_spectrum_angular_edges_full(
    self,
    *,
    indices: bool = False,
    energy_division: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64], xr.DataArray]:
    """Finds the angular edges of the spectrum based on energy slicing and rebinning.

    This method uses edge detection techniques to identify boundaries in the angular dimension.

    Args:
        indices (bool, optional): If True, returns edge indices; if False, returns physical
            angular coordinates. Defaults to False.
        energy_division (float, optional): Specifies the energy division step for rebinning.
            Defaults to 0.05 eV.

    Returns:
        tuple: A tuple containing:
            - low_edges (NDArray[np.float64]): Values or indices of the low edges
                of the spectrum.
            - high_edges (NDArray[np.float64]): Values or indices of the high edges
                of the spectrum.
            - eV_coords (xr.DataArray): The coordinates of the rebinned energy axis.

    Todo:
        - Add unit tests for this function.
    """
    # as a first pass, we need to find the bottom of the spectrum, we will use this
    # to select the active region and then to rebin into course steps in energy from 0
    # down to this region
    # we will then find the appropriate edge for each slice, and do a fit to the edge locations
    energy_edge = self.find_spectrum_energy_edges()
    low_edge: np.float64 = np.min(energy_edge) + energy_division
    high_edge: np.float64 = np.max(energy_edge) - energy_division

    if high_edge - low_edge < 3 * energy_division:
        # Doesn't look like the automatic inference of the energy edge was valid
        high_edge = self._obj.coords["eV"].max().item()
        low_edge = self._obj.coords["eV"].min().item()

    angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
    energy_cut = self._obj.sel(eV=slice(low_edge, high_edge)).S.sum_other(["eV", angular_dim])

    n_cuts = int(np.ceil((high_edge - low_edge) / energy_division))
    new_shape = {"eV": n_cuts}
    new_shape[angular_dim] = energy_cut.sizes[angular_dim]
    logger.debug(f"new_shape: {new_shape}")
    rebinned = rebin(energy_cut, shape=new_shape)

    embed_size = 20
    embedded: NDArray[np.float64] = np.empty(
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

    delta = self._obj.G.stride(generic_dim_names=False)

    return (
        np.array(low_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0],
        np.array(high_edges) * delta[angular_dim] + rebinned.coords[angular_dim].values[0],
        rebinned.coords["eV"],
    )


def zero_spectrometer_edges(
    self,
    cut_margin: int = 0,
    interp_range: float | None = None,
    low: Sequence[float] | NDArray[np.float64] | None = None,
    high: Sequence[float] | NDArray[np.float64] | None = None,
) -> xr.DataArray:
    """Zeros out the spectrum data outside of the specified low and high edges.

    It uses the provided or inferred edge information, applying cut margins and optionally
    interpolating over a given range.

    Args:
        cut_margin (int or float, optional): Margin to apply when invalidating data near edges.
            Use `int` for pixel-based margins or `float` for angular physical units.
            Defaults to 50 pixels or 0.08 in angular units, depending on the data type.
        interp_range (float or None, optional): Specifies the interpolation range for edge data.
            If provided, the edge values are interpolated within this range.
        low (Sequence[float], NDArray[np.float64], or None, optional): Low edge values.
            Use this to manually specify the low edge. Defaults to None.
            (automatically determined).
        high (Sequence[float], NDArray[np.float64], or None, optional): High edge values.
            Use this to manually specify the high edge. Defaults to None.
            (automatically determined).

    Returns:
        xr.DataArray: The spectrum data with values outside the edges set to zero.

    Todo:
        - Add tests.

    """
    assert isinstance(self._obj, xr.DataArray)
    if low is not None:
        assert high is not None
        assert len(low) == len(high) == TWO_DIMENSION

        low_edges = low
        high_edges = high

    (
        low_edges,
        high_edges,
        rebinned_eV_coord,
    ) = self.find_spectrum_angular_edges_full(indices=True)

    angular_dim = "pixel" if "pixel" in self._obj.dims else "phi"
    if not cut_margin:
        if "pixel" in self._obj.dims:
            cut_margin = 50
        else:
            cut_margin = int(0.08 / self._obj.G.stride(generic_dim_names=False)[angular_dim])
    elif isinstance(cut_margin, float):
        assert angular_dim == "phi"
        cut_margin = int(
            cut_margin / self._obj.G.stride(generic_dim_names=False)[angular_dim],
        )

    if interp_range is not None:
        low_edge = xr.DataArray(low_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
        high_edge = xr.DataArray(high_edges, coords={"eV": rebinned_eV_coord}, dims=["eV"])
        low_edge = low_edge.sel(eV=interp_range, method="nearest")
        high_edge = high_edge.sel(eV=interp_range, method="nearest")
    other_dims = list(self._obj.dims)
    other_dims.remove("eV")
    other_dims.remove(angular_dim)
    copied = self._obj.copy(deep=True).transpose("eV", angular_dim, ...)

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


def wide_angle_selector(self, *, include_margin: bool = True) -> slice:
    """Generates a slice for selecting the wide angular range of the spectrum.

    Optionally includes a margin to slightly reduce the range.

    Args:
        include_margin (bool, optional): If True, includes a margin to shrink the range.
            Defaults to True.

    Returns:
        slice: A slice object representing the wide angular range of the spectrum.

    Todo:
        - Add tests.
        - Consider removing the function.

    """
    edges = self.find_spectrum_angular_edges()
    low_edge, high_edge = np.min(edges), np.max(edges)

    # go and build in a small margin
    if include_margin:
        if "pixels" in self._obj.dims:
            low_edge += 50
            high_edge -= 50
        else:
            low_edge += 0.05
            high_edge -= 0.05

    return slice(low_edge, high_edge)


def meso_effective_selector(self) -> slice:
    """Creates a slice to select the "meso-effective" range of the spectrum.

    The range is defined as the upper energy range from `max(energy_edge) - 0.3` to
    `max(energy_edge) - 0.1`.

    Returns:
        slice: A slice object representing the meso-effective energy range.

    Todo:
        - Add tests.
        - Consider removing the function.

    """
    energy_edge = self.find_spectrum_energy_edges()
    return slice(np.max(energy_edge) - 0.3, np.max(energy_edge) - 0.1)


def region_sel(
    self,
    *regions: Literal["copper_prior", "wide_angular", "narrow_angular"]
    | dict[str, DesignatedRegions],
) -> xr.DataArray:
    """Filters the data by selecting specified regions and applying those regions to the object.

    Regions can be provided as literal strings or as a dictionary of `DesignatedRegions`.

    Args:
        regions (Literal or dict[str, DesignatedRegions]): The regions to select.
            Valid regions include:
            - "copper_prior": A specific region.
            - "wide_angular": The wide angular region.
            - "narrow_angular": The narrow angular region.
            Alternatively, use the `DesignatedRegions` enumeration.

    Returns:
        XrTypes: The data with the selected regions applied.

    Raises:
        NotImplementedError: If a specified region cannot be resolved.

    Todo:
        - Add tests.
    """

    def process_region_selector(
        selector: slice | DesignatedRegions,
        dimension_name: str,
    ) -> slice | Callable[..., slice]:
        if isinstance(selector, slice):
            return selector

        options = {
            "eV": (
                DesignatedRegions.ABOVE_EF,
                DesignatedRegions.BELOW_EF,
                DesignatedRegions.EF_NARROW,
                DesignatedRegions.MESO_EF,
                DesignatedRegions.MESO_EFFECTIVE_EF,
                DesignatedRegions.ABOVE_EFFECTIVE_EF,
                DesignatedRegions.BELOW_EFFECTIVE_EF,
                DesignatedRegions.EFFECTIVE_EF_NARROW,
            ),
            "phi": (
                DesignatedRegions.NARROW_ANGLE,
                DesignatedRegions.WIDE_ANGLE,
                DesignatedRegions.TRIM_EMPTY,
            ),
        }

        options_for_dim = options.get(dimension_name, list(DesignatedRegions))
        assert selector in options_for_dim

        # now we need to resolve out the region
        resolution_methods = {
            DesignatedRegions.ABOVE_EF: slice(0, None),
            DesignatedRegions.BELOW_EF: slice(None, 0),
            DesignatedRegions.EF_NARROW: slice(-0.1, 0.1),
            DesignatedRegions.MESO_EF: slice(-0.3, -0.1),
            DesignatedRegions.MESO_EFFECTIVE_EF: self.meso_effective_selector,
            # Implement me
            # DesignatedRegions.TRIM_EMPTY: ,
            DesignatedRegions.WIDE_ANGLE: self.wide_angle_selector,
            # DesignatedRegions.NARROW_ANGLE: self.narrow_angle_selector,
        }
        resolution_method = resolution_methods[selector]
        if isinstance(resolution_method, slice):
            return resolution_method
        if callable(resolution_method):
            return resolution_method()

        msg = "Unable to determine resolution method."
        raise NotImplementedError(msg)

    obj = self._obj

    def unpack_dim(dim_name: str) -> str:
        if dim_name == "angular":
            return "pixel" if "pixel" in obj.dims else "phi"

        return dim_name

    for region in regions:
        # remove missing dimensions from selection for permissiveness
        # and to transparent composing of regions
        obj = obj.sel(
            {
                k: process_region_selector(v, k)
                for k, v in {unpack_dim(k): v for k, v in normalize_region(region).items()}.items()
                if k in obj.dims
            },
        )

    return obj


def find_spectrum_edge(data: xr.DataArray, *, axis_is_energy: bool = True, indices: bool = False):
    pass


def angle_selector(data: xr.DataArray, *, include_margin: bool = True):
    pass


def region_sel(data: xr.DataArray, regions) -> xr.DataArray:
    pass
