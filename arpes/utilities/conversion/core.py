"""Helper functions for coordinate transformations and user/analysis API.

All the functions here assume standard polar angles, as given in the
`data model documentation <https://arpes.readthedocs.io/spectra>`_.

Functions here must accept constants or numpy arrays as valid inputs,
so all standard math functions have been replaced by their equivalents out
of numpy. Array broadcasting should handle any issues or weirdnesses that
would encourage the use of direct iteration, but in case you need to write
a conversion directly, be aware that any functions here must work on arrays
as well for consistency with client code.

Everywhere:

Kinetic energy -> 'kinetic_energy'
Binding energy -> 'eV', for convenience (negative below 0)
Photon energy -> 'hv'

Better facilities should be added for ToFs to do simultaneous (timing, angle)
to (binding energy, k-space).
"""

from __future__ import annotations

import collections
import contextlib
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from itertools import pairwise
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from arpes.provenance import PROVENANCE, provenance, update_provenance
from arpes.utilities import normalize_to_spectrum

from .fast_interp import Interpolator
from .grids import (
    determine_axis_type,
    determine_momentum_axes_from_measurement_axes,
    is_dimension_convertible_to_mementum,
)
from .kx_ky_conversion import ConvertKp, ConvertKxKy
from .kz_conversion import ConvertKpKz

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arpes._typing import MOMENTUM
    from arpes.utilities.conversion.calibration import DetectorCalibration

__all__ = ["convert_to_kspace", "slice_along_path"]


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[0]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def grid_interpolator_from_dataarray(
    arr: xr.DataArray,
    fill_value: float = 0.0,
    method: Literal["linear", "nearest", "slinear", "cubic", "quintic", "pchip"] = "linear",
    *,
    bounds_error: bool = False,
) -> RegularGridInterpolator | Interpolator:
    """Translates an xarray.DataArray contents into a scipy.interpolate.RegularGridInterpolator.

    This is principally used for coordinate translations.
    """
    assert isinstance(arr, xr.DataArray)
    flip_axes: set[str] = set()
    for d in arr.dims:
        c = arr.coords[d]
        if len(c) > 1 and c[1] - c[0] < 0:
            flip_axes.add(str(d))
    values: NDArray[np.float_] = arr.values
    for dim in flip_axes:
        values = np.flip(values, arr.dims.index(dim))
    interp_points = [
        arr.coords[d].values[::-1] if d in flip_axes else arr.coords[d].values for d in arr.dims
    ]
    trace_size = [len(pts) for pts in interp_points]

    if method == "linear":
        logger.debug(f"Using fast_interp.Interpolator: size {trace_size}")
        return Interpolator.from_arrays(interp_points, values)
    return RegularGridInterpolator(
        points=interp_points,
        values=values,
        bounds_error=bounds_error,
        fill_value=fill_value,
        method=method,
    )


def slice_along_path(
    arr: xr.DataArray,
    interpolation_points: NDArray[np.float_] | None = None,
    axis_name: str = "",
    resolution: float = 0,
    n_points: int | None = None,
    *,
    extend_to_edge: bool = False,
    shift_gamma: bool = True,
) -> xr.DataArray:
    """Gets a cut along a path specified by waypoints in an array.

    TODO: There might be a little bug here where the last coordinate has a value of 0,
    causing the interpolation to loop back to the start point. For now I will just deal
    with this in client code where I see it until I understand if it is universal.

    Interpolates along a path through a volume. If the volume is higher dimensional than
    the desired path, the interpolation is broadcasted along the free dimensions. This allows
    one to specify a k-space path and receive the band structure along this path in k-space.

    Points can either by specified by coordinates, or by reference to symmetry points, should they
    exist in the source array. These symmetry points are translated to regular coordinates
    immediately, but are provided as a convenience. If not all points specify the same set of
    coordinates, an attempt will be made to unify the coordinates. As an example, if the specified
    path is (kx=0, ky=0, T=20) -> (kx=1, ky=1), the path will be made between
    (kx=0, ky=0, T=20) -> (kx=1, ky=1, T=20). On the other hand,
    the path (kx=0, ky=0, T=20) -> (kx=1, ky=1, T=40) -> (kx=0, ky=1) will result
    in an error because there is no way to break the ambiguity on the temperature for the last
    coordinate.

    A reasonable value will be chosen for the resolution, near the maximum resolution of any of
    the interpolated axes by default.

    This function transparently handles the entire path. An alternate approach would be to convert
    each segment separately and concatenate the interpolated axis with xarray.

    If the sentinel value 'G' for the Gamma point is included in the interpolation points, the
    coordinate axis of the interpolated coordinate will be shifted so that its value at the Gamma
    point is 0. You can opt out of this with the parameter 'shift_gamma'

    Args:
        arr: Source data
        interpolation_points: Path vertices
        axis_name: Label for the interpolated axis. Under special
            circumstances a reasonable name will be chosen,
        resolution: Requested resolution along the interpolated axis.
        n_points: The number of desired points along the output path. This will be inferred
            approximately based on resolution if not provided.
        extend_to_edge: Controls whether or not to scale the vector S -
            G for symmetry point S so that you interpolate
        shift_gamma: Controls whether the interpolated axis is shifted
            to a value of 0 at Gamma.
        **kwargs

    such as when the interpolation dimensions are kx and ky: in this case the interpolated
    dimension will be labeled kp. In mixed or ambiguous situations the axis will be labeled
    by the default value 'inter' to the edge of the available data

    Returns:
        xr.DataArray containing the interpolated data.
    """
    if interpolation_points is None:
        msg = "You must provide points specifying an interpolation path"
        raise ValueError(msg)

    parsed_interpolation_points = [
        (
            x
            if isinstance(x, Iterable) and not isinstance(x, str)
            else _extract_symmetry_point(x, arr, extend_to_edge=extend_to_edge)
        )
        for x in interpolation_points
    ]

    free_coordinates = list(arr.dims)
    seen_coordinates = collections.defaultdict(set)
    for point in parsed_interpolation_points:
        for coord, value in point.items():
            seen_coordinates[coord].add(value)
            if coord in free_coordinates:
                free_coordinates.remove(coord)

    for point in parsed_interpolation_points:
        for coord, values in seen_coordinates.items():
            if coord not in point:
                if len(values) != 1:
                    msg = f"Ambiguous interpolation waypoint broadcast at dimension {coord}"
                    raise ValueError(
                        msg,
                    )
                point[coord] = next(iter(values))

    if not axis_name:
        try:
            axis_name = determine_axis_type(seen_coordinates.keys())
        except KeyError:
            axis_name = "inter"

        if axis_name == ("angle", "inter"):
            warnings.warn(
                "Interpolating along axes with different dimensions "
                "will not include Jacobian correction factor.",
                stacklevel=2,
            )

    converted_coordinates = None
    converted_dims = [*free_coordinates, axis_name]

    path_segments = list(pairwise(parsed_interpolation_points))

    def required_sampling_density(waypoint_a: Mapping, waypoint_b: Mapping) -> float:
        ks = waypoint_a.keys()
        dist = _element_distance(waypoint_a, waypoint_b)
        delta = np.array([waypoint_a[k] - waypoint_b[k] for k in ks])
        delta_idx = [
            abs(d / (arr.coords[k][1] - arr.coords[k][0])) for d, k in zip(delta, ks, strict=True)
        ]
        return dist / np.max(delta_idx)

    # Approximate how many points we should use
    segment_lengths = [_element_distance(*segment) for segment in path_segments]
    path_length = sum(segment_lengths)

    gamma_offset = 0  # offset the gamma point to a k coordinate of 0 if possible
    if "G" in interpolation_points and shift_gamma:
        gamma_offset = sum(segment_lengths[0 : interpolation_points.index("G")])

    if not resolution:
        if n_points is None:
            resolution = np.min([required_sampling_density(*segment) for segment in path_segments])
        else:
            resolution = path_length / n_points

    def converter_for_coordinate_name(name: str) -> Callable[..., NDArray[np.float_]]:
        def raw_interpolator(*coordinates: NDArray[np.float_]) -> NDArray[np.float_]:
            return coordinates[free_coordinates.index(name)]

        if name in free_coordinates:
            return raw_interpolator

        # Conversion involves the interpolated coordinates
        def interpolated_coordinate_to_raw(*coordinates: NDArray[np.float_]) -> NDArray[np.float_]:
            # Coordinate order is [*free_coordinates, interpolated]
            interpolated = coordinates[len(free_coordinates)] + gamma_offset

            # Start with empty array that we will mask writes onto
            # We need to go with a masking approach rather than a concatenation based one because
            # the coordinates come from np.meshgrid
            dest_coordinate = np.zeros(shape=interpolated.shape)

            start = 0.0
            for i, length in enumerate(segment_lengths):
                end = start + float(length)
                normalized = (interpolated - start) / length
                seg_start, seg_end = path_segments[i]
                dim_start, dim_end = seg_start[name], seg_end[name]
                mask = np.logical_and(normalized >= 0, normalized < 1)
                dest_coordinate[mask] = (
                    dim_start * (1 - normalized[mask]) + dim_end * normalized[mask]
                )
                start = end
            return dest_coordinate

        return interpolated_coordinate_to_raw

    converted_coordinates = {d: arr.coords[d].values for d in free_coordinates}

    if n_points is None:
        n_points = int(path_length / resolution)

    # Adjust this coordinate under special circumstances
    converted_coordinates[axis_name] = (
        np.linspace(0, path_length, int(path_length / resolution)) - gamma_offset
    )

    converted_ds = convert_coordinates(
        arr,
        converted_coordinates,
        {
            "dims": converted_dims,
            "transforms": dict(
                zip(
                    arr.dims,
                    [converter_for_coordinate_name(str(d)) for d in arr.dims],
                    strict=True,
                ),
            ),
        },
        as_dataset=True,
    )

    if (
        axis_name in arr.dims and len(parsed_interpolation_points) == 2  # noqa: PLR2004
    ) and parsed_interpolation_points[1][axis_name] < parsed_interpolation_points[0][axis_name]:
        # swap the sign on this axis as a convenience to the caller
        converted_ds.coords[axis_name].data = -converted_ds.coords[axis_name].data

    if "id" in converted_ds.attrs:
        del converted_ds.attrs["id"]
        provenance_context: PROVENANCE = {
            "what": "Slice along path",
            "by": "slice_along_path",
            "parsed_interpolation_points": parsed_interpolation_points,
            "interpolation_points": interpolation_points,
        }

        provenance(converted_ds, arr, provenance_context)

    return converted_ds


@update_provenance("Automatically k-space converted")
def convert_to_kspace(
    arr: xr.DataArray,
    bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    resolution: dict[MOMENTUM, float] | None = None,
    calibration: DetectorCalibration | None = None,
    coords: dict[MOMENTUM, NDArray[np.float_]] | None = None,
    *,
    allow_chunks: bool = False,
    **kwargs: NDArray[np.float_],
) -> xr.DataArray:
    """Converts volumetric the data to momentum space ("backwards"). Typically what you want.

    Works in general by regridding the data into the new coordinate space and then
    interpolating back into the original data.

    For forward conversion, see sibling methods. Forward conversion works by
    totally unchanged by the conversion (if we do not apply a Jacobian correction), but the
    converting the coordinates, rather than by interpolating the data. As a result, the data will be
    coordinates will no longer have equal spacing.

    This is only really useful for zero and one dimensional data because for two dimensional data,
    the coordinates must become two dimensional in order to fully specify every data point
    (this is true in generality, in 3D the coordinates must become 3D as well).

    The only exception to this is if the extra axes do not need to be k-space converted. As is the
    case where one of the dimensions is `cycle` or `delay`, for instance.

    You can request a particular resolution for the new data with the `resolution=` parameter,
    or a specific set of bounds with the `bounds=`

    Examples:
        Convert a 2D cut with automatically inferred range and resolution.

        >>> convert_to_kspace(arpes.io.load_example_data())  # doctest: +SKIP
        xr.DataArray(...)

        Convert a 3D map with a specified momentum window

        >>> convert_to_kspace(  # doctest: +SKIP
                fermi_surface_map,
                kx=np.linspace(-1, 1, 200),
                ky=np.linspace(-1, 1, 350),
            )
        xr.DataArray(...)

    Args:
        arr (xr.DataArray): ARPES data
        bounds (dict[MOMENTUM, tuple[float, float]], optional):
            The key is the axis name. The value is the bounds. Defaults to {}.
            If not set this arg, set coords.
        resolution (dict[Momentum, float], optional): dict for the energy/angular resolution.
        calibration (DetectorCalibration, optional): DetectorCalibration object. Defaults to None.
        coords (dict[Momentum, Iterable[float], optional): Coordinate of k-space. Defaults to {}.
        allow_chunks (bool): [description]. Defaults to False.
        **kwargs: treated as coords.

    Raises:
        NotImplementedError: [description]
        AnalysisError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if coords is None:
        coords = {}
    if bounds is None:
        bounds = {}
    coords.update(**kwargs)
    assert isinstance(coords, dict)
    if isinstance(arr, xr.Dataset):
        msg = "Remember to use a DataArray not a Dataset, "
        msg += "attempting to extract spectrum and copy attributes."
        warnings.warn(
            msg,
            stacklevel=2,
        )
        attrs = arr.attrs.copy()
        arr = normalize_to_spectrum(arr)
        arr.attrs.update(attrs)
    assert isinstance(arr, xr.DataArray)

    if arr.S.angle_unit.startswith("Deg") or arr.S.angle_unit.startswith("deg"):
        arr.S.swap_angle_unit()

    # Chunking logic
    if allow_chunks and ("eV" in arr.dims) and len(arr.eV) > 50:  # noqa: PLR2004
        return _chunk_convert(
            arr=arr,
            bounds=bounds,
            resolution=resolution,
            calibration=calibration,
            coords=coords,
            **kwargs,
        )
    momentum_incompatibles: list[str] = [
        str(d) for d in arr.dims if not is_dimension_convertible_to_mementum(str(d))
    ]
    momentum_compatibles: list[str] = [
        str(d) for d in arr.dims if is_dimension_convertible_to_mementum(str(d))
    ]

    # Energy gets put at the front as a standardization
    if "eV" in momentum_incompatibles:
        momentum_incompatibles.remove("eV")

    momentum_compatibles.sort()

    # temporarily reassign coordinates for dimensions we will not
    # convert to "index-like" dimensions
    restore_index_like_coordinates: dict[str, NDArray[np.float_]] = {
        r: arr.coords[r].values for r in momentum_incompatibles
    }
    new_index_like_coordinates = {
        r: np.arange(len(arr.coords[r].values)) for r in momentum_incompatibles
    }
    arr = arr.assign_coords(new_index_like_coordinates)

    if not momentum_compatibles:
        return arr  # no need to convert, might be XPS or similar

    converted_dims: list[str] = (
        (["eV"] if ("eV" in arr.dims) else [])
        + determine_momentum_axes_from_measurement_axes(momentum_compatibles)
        + momentum_incompatibles
    )
    convert_cls = {
        ("phi",): ConvertKp,
        ("beta", "phi"): ConvertKxKy,
        ("phi", "theta"): ConvertKxKy,
        ("phi", "psi"): ConvertKxKy,
        # ('chi', 'phi',): ConvertKxKy,
        ("hv", "phi"): ConvertKpKz,
    }.get(tuple(momentum_compatibles))
    assert convert_cls is not None, "Cannot select convert class"

    converter = convert_cls(arr, converted_dims, calibration=calibration)

    converted_coordinates: dict[str, NDArray[np.float_]] = converter.get_coordinates(
        resolution=resolution,
        bounds=bounds,
    )
    if not set(coords.keys()).issubset(converted_coordinates.keys()):
        extra = set(coords.keys()).difference(converted_coordinates.keys())
        msg = f"Unexpected passed coordinates: {extra}"
        raise ValueError(msg)
    converted_coordinates.update(**coords)  # type: ignore[misc]
    result = convert_coordinates(
        arr,
        converted_coordinates,
        {
            "dims": converted_dims,
            "transforms": dict(
                zip(arr.dims, [converter.conversion_for(dim) for dim in arr.dims], strict=True),
            ),
        },
    )
    assert isinstance(result, xr.DataArray)
    return result.assign_coords(restore_index_like_coordinates)


def convert_coordinates(
    arr: xr.DataArray,
    target_coordinates: dict[str, NDArray[np.float_]],
    coordinate_transform: dict[
        str,
        list[str] | dict[str, Callable[..., NDArray[np.float_]]],
    ],
    *,
    as_dataset: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Return Band structure data (converted to k-space).

    Args:
        arr(xr.DataArray): ARPES data
        target_coordinates:(dict[Hashable, NDArray[np.float_]]):  coorrdinate for ...
        coordinate_transform(dict[str, list[str] | Callable]): coordinat for ...
        as_dataset(bool): if True, return the data as the dataSet

    Returns:
        xr.DataArray | xr.Dataset
    """
    assert isinstance(arr, xr.DataArray)
    ordered_source_dimensions = arr.dims

    grid_interpolator = grid_interpolator_from_dataarray(
        arr.transpose(*ordered_source_dimensions),
        fill_value=float("nan"),
    )

    # Skip the Jacobian correction for now
    # Convert the raw coordinate axes to a set of gridded points
    logger.debug(f"meshgrid: {[len(target_coordinates[_]) for _ in coordinate_transform['dims']]}")
    meshed_coordinates = np.meshgrid(
        *[target_coordinates[dim] for dim in coordinate_transform["dims"]],
        indexing="ij",
    )
    meshed_coordinates = [meshed_coord.ravel() for meshed_coord in meshed_coordinates]

    if "eV" not in arr.dims:
        with contextlib.suppress(ValueError):
            meshed_coordinates = [arr.S.lookup_offset_coord("eV"), *meshed_coordinates]
    old_coord_names = [str(dim) for dim in arr.dims if dim not in target_coordinates]
    assert isinstance(coordinate_transform["transforms"], dict)
    transforms: dict[str, Callable[..., NDArray[np.float_]]] = coordinate_transform["transforms"]
    logger.debug(f"transforms is {transforms}")
    old_coordinate_transforms = [
        transforms[str(dim)] for dim in arr.dims if dim not in target_coordinates
    ]
    logger.debug(f"old_coordinate_transforms: {old_coordinate_transforms}")

    output_shape = [len(target_coordinates[d]) for d in coordinate_transform["dims"]]

    def compute_coordinate(transform: Callable[..., NDArray[np.float_]]) -> NDArray[np.float_]:
        logger.debug(f"transform function is {transform}")
        return np.reshape(
            transform(*meshed_coordinates),
            output_shape,
            order="C",
        )

    old_dimensions = [compute_coordinate(tr) for tr in old_coordinate_transforms]

    ordered_transformations = [transforms[str(dim)] for dim in arr.dims]
    transformed_coordinates = [tr(*meshed_coordinates) for tr in ordered_transformations]

    if not isinstance(grid_interpolator, Interpolator):
        converted_volume = grid_interpolator(np.array(transformed_coordinates).T)
    else:
        converted_volume = grid_interpolator(transformed_coordinates)

    # Wrap it all up
    def acceptable_coordinate(c: NDArray[np.float_] | xr.DataArray) -> bool:
        """[TODO:summary].

        Currently we do this to filter out coordinates
        that are functions of the old angular dimensions,
        we could forward convert these, but right now we do not

        Args:
            c: [TODO:description]

        Returns:
            [TODO:description]
        """
        try:
            return bool(set(c.dims).issubset(coordinate_transform["dims"]))
        except AttributeError:
            return True

    target_coordinates = {k: v for k, v in target_coordinates.items() if acceptable_coordinate(v)}
    data = xr.DataArray(
        np.reshape(
            converted_volume,
            [len(target_coordinates[d]) for d in coordinate_transform["dims"]],
            order="C",
        ),
        target_coordinates,
        coordinate_transform["dims"],
        attrs=arr.attrs,
    )
    old_mapped_coords = [
        xr.DataArray(values, target_coordinates, coordinate_transform["dims"], attrs=arr.attrs)
        for values in old_dimensions
    ]
    if as_dataset:
        variables = {"data": data}
        variables.update(
            dict(
                zip(
                    old_coord_names,
                    old_mapped_coords,
                    strict=True,
                ),
            ),
        )
        return xr.Dataset(variables, attrs=arr.attrs)

    return data


def _chunk_convert(
    arr: xr.DataArray,
    bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    resolution: dict[MOMENTUM, float] | None = None,
    calibration: DetectorCalibration | None = None,
    coords: dict[MOMENTUM, NDArray[np.float_]] | None = None,
    **kwargs: NDArray[np.float_],
) -> xr.DataArray:
    DESIRED_CHUNK_SIZE = 1000 * 1000 * 20
    TOO_LARGE_CHUNK_SIZE = 100
    n_chunks: np.int_ = np.prod(arr.shape) // DESIRED_CHUNK_SIZE
    if n_chunks == 0:
        warnings.warn(
            "Data size is sufficiently small, set allow_chunks=False",
            stacklevel=2,
        )
        n_chunks += 1

    if n_chunks > TOO_LARGE_CHUNK_SIZE:
        warnings.warn(
            "Input array is very large. Please consider resampling.",
            stacklevel=2,
        )
    chunk_thickness = np.max(len(arr.eV) // n_chunks, 1)
    logger.debug(f"Chunking along energy: {n_chunks}, thickness {chunk_thickness}")
    finished = []
    low_idx = 0
    high_idx = chunk_thickness
    while low_idx < len(arr.eV):
        chunk = arr.isel(eV=slice(low_idx, high_idx))
        if len(chunk.eV) == 1:
            chunk = chunk.squeeze("eV")
        kchunk = convert_to_kspace(
            chunk,
            bounds=bounds,
            resolution=resolution,
            calibration=calibration,
            coords=coords,
            allow_chunks=False,
            **kwargs,
        )
        if "eV" not in kchunk.dims:
            kchunk = kchunk.expand_dims("eV")
        assert isinstance(kchunk, xr.DataArray)
        finished.append(kchunk)
        low_idx = high_idx
        high_idx = min(len(arr.eV), high_idx + chunk_thickness)
    return xr.concat(finished, dim="eV")


def _extract_symmetry_point(
    name: str,
    arr: xr.DataArray,
    *,
    extend_to_edge: bool = False,
) -> dict:
    """[TODO:summary].

    Args:
        name (str): [TODO:description]
        arr (xr.DataArray): [TODO:description]
        extend_to_edge (bool): [TODO:description]

    Returns:
        [TODO:description]
    """
    raw_point: dict[Hashable, float] = arr.attrs["symmetry_points"][name]
    G = arr.attrs["symmetry_points"]["G"]

    if not extend_to_edge or name == "G":
        return raw_point

    # scale the point so that it reaches the edge of the dataset
    S = np.array([raw_point[d] for d in arr.dims if d in raw_point])
    G = np.array([G[d] for d in arr.dims if d in raw_point])

    scale_factor = np.inf
    for i, d in enumerate([d for d in arr.dims if d in raw_point]):
        dS = (S - G)[i]
        coord = arr.coords[d]

        if np.abs(dS) < 0.001:  # noqa: PLR2004
            continue

        if dS < 0:
            required_scale = (np.min(coord) - G[i]) / dS
            if required_scale < scale_factor:
                scale_factor = float(required_scale)
        else:
            required_scale = (np.max(coord) - G[i]) / dS
            if required_scale < scale_factor:
                scale_factor = float(required_scale)

    S = (S - G) * scale_factor + G
    return dict(zip([d for d in arr.dims if d in raw_point], S, strict=False))


def _element_distance(waypoint_a: Mapping, waypoint_b: Mapping) -> np.float_:
    delta = np.array([waypoint_a[k] - waypoint_b[k] for k in waypoint_a])
    return np.linalg.norm(delta)
