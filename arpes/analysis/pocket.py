"""Contains electron/hole pocket analysis routines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

from arpes.fits.fit_models import AffineBackgroundModel, LorentzianModel
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.conversion import slice_along_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete

    from arpes._typing import DataType, XrTypes
__all__ = (
    "curves_along_pocket",
    "edcs_along_pocket",
    "radial_edcs_along_pocket",
    "pocket_parameters",
)


def pocket_parameters(
    data: DataType,
    kf_method: Callable[..., float] | None = None,
    sel: dict[str, slice] | None = None,
    method_kwargs: Incomplete = None,
    **kwargs: Incomplete,
) -> dict[str, Any]:
    """Estimates pocket center, anisotropy, principal vectors, and extent.

    Since data can be converted forward it is generally advised to do
    this analysis in angle space before conversion if the pocket is not very large.

    Args:
        data: The input kx-ky or 2D angle map.
        kf_method: How to determine k_F for each slice.
        sel: An energy selection window near the Fermi surface.
        method_kwargs: Passed to the kf determination method
        kwargs: Passed to the radial selection method.

    Returns:
        Extracted asymmetry parameters.
    """
    slices, _ = curves_along_pocket(data, **kwargs)  # slices, angles =

    if kf_method is None:
        kf_method = find_kf_by_mdc

    if sel is None:
        sel = {"eV": slice(-0.03, 0.05)}

    kfs = [kf_method(s if sel is None else s.sel(sel), **(method_kwargs or {})) for s in slices]

    fs_dims = list(data.dims)
    if "eV" in fs_dims:
        fs_dims.remove("eV")

    locations = [
        {d: ss[d].sel(angle=kf, eV=0, method="nearest").item() for d in fs_dims}
        for kf, ss in zip(kfs, slices, strict=True)
    ]

    location_vectors = [[coord[d] for d in fs_dims] for coord in locations]
    as_ndarray = np.array(location_vectors)

    pca = PCA(n_components=2)
    pca.fit(as_ndarray)

    return {
        "locations": locations,
        "location_vectors": location_vectors,
        "center": {d: np.mean(np.array([coord[d] for coord in locations])) for d in fs_dims},
        "pca": pca.components_,
    }


@update_provenance("Collect EDCs projected at an angle from pocket")
def radial_edcs_along_pocket(
    data: xr.DataArray,
    angle: float,
    radii: tuple[float, float] = (0.0, 5.0),
    n_points: int = 0,
    select_radius: dict[str, float] | float | None = None,
    **kwargs: Incomplete,
) -> xr.Dataset:
    """Produces EDCs distributed radially along a vector from the pocket center.

    The pocket center should be passed through kwargs via `{dim}={value}`.

    Example:
        I.e. an appropriate call would be

        >>> radial_edcs_along_pocket(spectrum, np.pi / 4, (1, 4), phi=0.1, beta=0)

    Args:
        data (XrTypes): ARPES Spectrum.
        angle (float): Angle along the FS to cut against.
        radii (tuple[float, float]): The min and max for the angle/momentum equivalent radial
                                     coordinate.
        n_points: Number of EDCs, can be automatically inferred.
        select_radius: The radius used for selections along the radial curve.
        kwargs: Used to define the central point.

    Return:
        A 2D array which has an angular coordinate around the pocket center.
    """
    inner_radius, outer_radius = radii
    data_array = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    fermi_surface_dims = list(data_array.dims)

    assert "eV" in fermi_surface_dims
    fermi_surface_dims.remove("eV")

    center_point = {k: v for k, v in kwargs.items() if k in data_array.dims}
    center_as_vector = np.array([center_point.get(d, 0) for d in fermi_surface_dims])

    if not n_points:
        stride = data_array.G.stride(generic_dim_names=False)
        granularity = np.mean(np.array([stride[d] for d in fermi_surface_dims]))
        n_points = int(1.0 * (outer_radius - inner_radius) / granularity)

    if n_points <= 0:
        n_points = 10

    primitive = np.array([np.cos(angle), np.sin(angle)])
    far = center_as_vector + outer_radius * primitive
    near = center_as_vector + inner_radius * primitive
    vecs = zip(near, far, strict=True)

    radius_coord = np.linspace(inner_radius, outer_radius, n_points)

    data_vars = {}
    for d, points in dict(zip(fermi_surface_dims, vecs, strict=True)).items():
        data_vars[d] = xr.DataArray(
            np.array(np.linspace(points[0], points[1], n_points)),
            coords={"r": radius_coord},
            dims=["r"],
        )

    selection_coords = [{k: v[n] for k, v in data_vars.items()} for n in range(n_points)]

    edcs = [data_array.S.select_around(coord, radius=select_radius) for coord in selection_coords]

    for r, edc in zip(radius_coord, edcs, strict=True):
        edc.coords["r"] = r

    data_vars["data"] = xr.concat(edcs, dim="r")

    return xr.Dataset(data_vars, coords=data_vars["data"].coords)


def curves_along_pocket(
    data: xr.DataArray,
    n_points: int = 0,
    inner_radius: float = 0.0,
    outer_radius: float = 5.0,
    **kwargs: Incomplete,
) -> tuple[list[xr.DataArray], list[float]]:
    """Produces radial slices along a Fermi surface through a pocket.

    Evenly distributes perpendicular cuts along an
    ellipsoid. The major axes of the ellipsoid can be specified by `shape`
    but must be axis aligned.

    The inner and outer radius parameters control the endpoints of the
    resultant slices along the Fermi surface.

    Args:
        data: input data
        n_points (int):
        inner_radius:
        outer_radius:
        shape:
        kwargs:

    Returns:
        A tuple of two lists. The first list contains the slices and the second
        the coordinates of each slice around the pocket center.
    """
    data_array = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(data_array, xr.DataArray)
    fermi_surface_dims = list(data_array.dims)
    if "eV" in fermi_surface_dims:
        fermi_surface_dims.remove("eV")

    center_point = {str(k): v for k, v in kwargs.items() if k in data_array.dims}

    center_as_vector = np.array([center_point.get(dim_name, 0) for dim_name in fermi_surface_dims])

    if not n_points:
        # determine N approximately by the granularity
        n_points = np.min([len(data_array.coords[d].values) for d in fermi_surface_dims])

    stride = data_array.G.stride(generic_dim_names=False)
    resolution = np.min([v for s, v in stride.items() if s in fermi_surface_dims])

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    def slice_at_angle(theta: float) -> xr.DataArray:
        primitive = np.array([np.cos(theta), np.sin(theta)])
        far = center_as_vector + outer_radius * primitive

        return slice_along_path(
            data_array,
            [
                dict(zip(fermi_surface_dims, point, strict=True))
                for point in [center_as_vector, far]
            ],
            resolution=resolution,
        )

    slices = [slice_at_angle(theta) for theta in angles]

    max_ang = slices[0].coords["angle"].max().item()

    slices = [
        s.sel(angle=slice(max_ang * (1.0 * inner_radius / outer_radius), None)).isel(
            angle=slice(None, -1),
        )
        for s in slices
    ]

    for ang, s in zip(angles, slices, strict=True):
        s.coords["theta"] = ang

    # we do not xr.concat because the interpolated angular dim can actually be different on each due
    # to floating point nonsense
    return slices, angles


def find_kf_by_mdc(
    slice_data: xr.DataArray,
    offset: float = 0,
    **kwargs: Incomplete,
) -> float:
    """Finds the Fermi momentum by curve fitting an MDC.

    Offset is used to control the radial offset from the pocket for studies where
    you want to go slightly off the Fermi momentum.

    Args:
        slice_data: Input fit data.
        offset: Offset to add to the result
        kwargs: Passed as parameters to the fit routine.

    Returns:
        The fitting Fermi momentum.
    """
    slice_arr = (
        slice_data if isinstance(slice_data, xr.DataArray) else normalize_to_spectrum(slice_data)
    )

    assert isinstance(slice_arr, xr.DataArray)

    if "eV" in slice_arr.dims:
        slice_arr = slice_arr.sum("eV")

    lor = LorentzianModel()
    bkg = AffineBackgroundModel(prefix="b_")

    result = (lor + bkg).guess_fit(data=slice_arr, params=kwargs)
    return result.params["center"].value + offset


@update_provenance("Collect EDCs around pocket edge")
def edcs_along_pocket(
    data: XrTypes,
    kf_method: Callable[..., float] | None = None,
    select_radius: dict[str, float] | None = None,
    sel: dict[str, slice] | None = None,
    method_kwargs: Incomplete | None = None,
    **kwargs: Incomplete,
) -> xr.Dataset:
    """Collects EDCs around a pocket.

    This consists first in identifying the momenta
    around the pocket, and then integrating small windows around each of these points.

    Args:
        data: The input kx-ky or 2D angle map.
        kf_method: How to determine k_F for each slice.
        select_radius: The radius used for selections along the radial curve.
        sel: An energy selection window near the Fermi surface.
        method_kwargs: Passed to the kf determination method
        kwargs: Passed to the radial selection method.

    Returns:
        EDCs at the fermi momentum around a pocket.
    """
    slices, angles = curves_along_pocket(data, **kwargs)

    if kf_method is None:
        kf_method = find_kf_by_mdc

    if sel is None:
        sel = {"eV": slice(-0.05, 0.05)}

    kfs = [kf_method(s if sel is None else s.sel(sel), **(method_kwargs or {})) for s in slices]

    fs_dims = list(data.dims)
    if "eV" in fs_dims:
        fs_dims.remove("eV")

    locations = [
        {d: ss[d].sel(angle=kf, eV=0, method="nearest").item() for d in fs_dims}
        for kf, ss in zip(kfs, slices, strict=True)
    ]

    edcs = [data.S.select_around(_, radius=select_radius) for _ in locations]

    data_vars = {}
    index = np.array(angles)

    for d in fs_dims:
        data_vars[d] = xr.DataArray(
            np.array([location[d] for location in locations]),
            coords={"theta": index},
            dims=["theta"],
        )

    for ang, edc in zip(angles, edcs, strict=True):
        edc.coords["theta"] = ang

    data_vars["spectrum"] = xr.concat(edcs, dim="theta")

    return xr.Dataset(data_vars, coords=data_vars["spectrum"].coords)
