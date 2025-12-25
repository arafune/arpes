"""Contains routines used to do path selections and manipulations on a dataset."""

from __future__ import annotations

import collections
from itertools import pairwise
from typing import TYPE_CHECKING, cast

import numpy as np
import xarray as xr

from arpes.constants import TWO_DIMENSION
from arpes.provenance import update_provenance

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray

from collections.abc import Callable, Hashable, Mapping
from logging import DEBUG, INFO

from arpes.debug import setup_logger
from arpes.provenance import Provenance, provenance
from arpes.utilities.conversion.coordinates import convert_coordinates
from arpes.utilities.conversion.grids import determine_axis_type

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


__all__ = (
    "discretize_path",
    "select_along_path",
    "slice_along_path",
)


@update_provenance("Discretize Path")
def discretize_path(
    path: xr.Dataset,
    n_points: int = 0,
    scaling: float | xr.Dataset | dict[str, float] | None = None,
) -> xr.Dataset:
    """Discretizes a path into a set of points spaced along the path.

    Shares logic with slice_along_path

    Args:
        path: The path specification.
        n_points: The number of points to space along the path.
        scaling: A metric allowing calculating a distance from mixed coordinates. This
          is needed because we disperse points equidistantly along the path. Typically
          you can leave this unset.

    Returns:
        An xr.Dataset of the points along the path.
    """
    if scaling is None:
        scaling = 1
    elif isinstance(scaling, xr.Dataset):
        scaling = {str(k): scaling[k].item() for k in scaling.data_vars}
    else:
        assert isinstance(scaling, dict)

    order = list(path.data_vars)
    if isinstance(scaling, dict):
        scaling = np.array(float(scaling[d]) for d in order)

    assert isinstance(scaling, np.ndarray | float)

    def as_vec(ds: xr.Dataset) -> NDArray[np.float64]:
        return np.array([ds[k].item() for k in order])

    def distance(a: xr.Dataset, b: xr.Dataset) -> float:
        return float(np.linalg.norm((as_vec(a) - as_vec(b)) * scaling))

    length = 0
    for idx_low, idx_high in zip(path.index.values, path.index[1:].values, strict=False):
        coord_low, coord_high = path.sel(index=idx_low), path.sel(index=idx_high)
        length += distance(coord_low, coord_high)

    n_points = int(length / 0.03) if not n_points else max(n_points - 1, 1)

    points = []
    distances = np.linspace(0, n_points - 1, n_points) * (length / n_points)

    total_dist = 0
    for idx_low, idx_high in zip(path.index.values, path.index[1:].values, strict=False):
        coord_low, coord_high = path.sel(index=idx_low), path.sel(index=idx_high)

        current_dist = distance(coord_low, coord_high)
        current_points = distances[distances < total_dist + current_dist]
        current_points = (current_points - total_dist) / current_dist
        distances = distances[len(current_points) :]
        total_dist += current_dist

        points += list(
            np.outer(current_points, as_vec(coord_high) - as_vec(coord_low)) + as_vec(coord_low),
        )

    points.append(as_vec(path.sel(index=path.index.values[-1])))

    new_index = np.array(range(len(points)))

    def to_dataarray(name: str) -> xr.DataArray:
        index = order.index(name)
        data = [p[index] for p in points]

        return xr.DataArray(np.array(data), {"index": new_index}, ["index"])

    return xr.Dataset({k: to_dataarray(k) for k in order})


@update_provenance("Select from data along a path")
def select_along_path(
    path: xr.Dataset,
    data: xr.DataArray,
    radius: float = 0,
    n_points: int = 0,
    *,
    scaling: float | xr.Dataset | dict[str, float] | None = None,
    **kwargs: Incomplete,
) -> xr.DataArray:
    """Performs integration along a path.

    This functionally allows for performing a finite width
    cut (with finite width perpendicular to the local path direction) along some path,
    and integrating along this perpendicular selection. This allows for better statistics in
    oversampled data.

    Args:
        path: The path to select along.
        data: The data to select/interpolate from.
        radius: A number or dictionary of radii to use for the selection along different dimensions,
                if none is provided reasonable values will be chosen. Alternatively, you can pass
                radii via `{dim}_r` kwargs as well, i.e. 'eV_r' or 'kp_r'
        n_points: The number of points to interpolate along the path, by default we will infer a
                  reasonable number from the radius parameter, if provided or inferred
        scaling: A metric allowing calculating a distance from mixed coordinates.
                 Pass it to discretize_path as is.
        kwargs: kwargs pass to S.select_around

    Returns:
        The data selected along the path.
    """
    new_path = discretize_path(path, n_points, scaling)

    selections = []
    for coord in new_path.G.iter_coords("index"):
        view = new_path.sel(coord, method="nearest")
        selections.append(data.S.select_around(view, radius=radius, **kwargs))

    return xr.concat(selections, new_path.index)


def slice_along_path(  # noqa: PLR0913
    arr: xr.DataArray,
    interpolation_points: list[Hashable | dict[Hashable, float]],
    axis_name: str = "",
    resolution: float = 0,
    n_points: int = 0,
    *,
    extend_to_edge: bool = False,
) -> xr.Dataset:
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

    Args:
        arr: Source data
        interpolation_points( list[str | dict[str, float]]):
            Path vertices
        axis_name: Label for the interpolated axis. Under special
            circumstances a reasonable name will be chosen,
        resolution: Requested resolution along the interpolated axis.
        n_points: Thej number of desired points along the output path. This will be inferred
            approximately based on resolution if not provided.
        extend_to_edge: Controls whether or not to scale the vector S -
            G for symmetry point S so that you interpolate
    such as when the interpolation dimensions are kx and ky: in this case the interpolated
    dimension will be labeled kp. In mixed or ambiguous situations the axis will be labeled
    by the default value 'inter' to the edge of the available data

    Returns:
        xr.DataArray containing the interpolated data.
    """
    parsed_interpolation_points, free_coordinates, seen_coordinates = _parse_interpolation_points(
        interpolation_points,
        arr,
        extend_to_edge=extend_to_edge,
    )
    logger.debug(f"parsed_interpolation_points: {parsed_interpolation_points}")

    if not axis_name:
        try:
            axis_name = determine_axis_type(seen_coordinates.keys())
        except KeyError:
            axis_name = "inter"

    converted_dims = [*free_coordinates, axis_name]

    path_segments = list(pairwise(parsed_interpolation_points))

    # Approximate how many points we should use
    segment_lengths = [_element_distance(segment[0], segment[1]) for segment in path_segments]
    path_length = float(np.sum(segment_lengths))

    resolution = resolution or (
        np.min(
            [_required_sampling_density(arr, segment[0], segment[1]) for segment in path_segments],
        )
        if not n_points
        else path_length / n_points
    )

    def converter_for_coordinate_name(name: str) -> Callable[..., NDArray[np.float64]]:
        def raw_interpolator(*coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
            return coordinates[free_coordinates.index(name)]

        if name in free_coordinates:
            return raw_interpolator

        # Conversion involves the interpolated coordinates
        def interpolated_coordinate_to_raw(
            *coordinates: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            # Coordinate order is [*free_coordinates, interpolated]
            interpolated = coordinates[len(free_coordinates)]

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

    n_points = n_points or int(sum(segment_lengths) / resolution)

    # Adjust this coordinate under special circumstances
    converted_coordinates[axis_name] = np.linspace(
        0,
        sum(segment_lengths),
        int(sum(segment_lengths) / resolution),
    )

    converted_ds = convert_coordinates(
        arr,
        target_coordinates=converted_coordinates,
        coordinate_transform={
            "dims": converted_dims,
            "transforms": {str(d): converter_for_coordinate_name(str(d)) for d in arr.dims},
        },
        as_dataset=True,
    )
    assert isinstance(converted_ds, xr.Dataset)

    if (
        axis_name in arr.dims and len(parsed_interpolation_points) == TWO_DIMENSION
    ) and parsed_interpolation_points[1][axis_name] < parsed_interpolation_points[0][axis_name]:
        # swap the sign on this axis as a convenience to the caller
        converted_ds = converted_ds.assign_coords({axis_name: -1 * converted_ds.coords[axis_name]})

    if "id" in converted_ds.attrs:
        del converted_ds.attrs["id"]
        provenance_context: Provenance = cast(
            "Provenance",
            {
                "what": "Slice along path",
                "by": "slice_along_path",
                "parsed_interpolation_points": parsed_interpolation_points,
                "interpolation_points": interpolation_points,
            },
        )

        provenance(converted_ds, arr, provenance_context)

    return converted_ds


def _parse_interpolation_points(
    interpolation_points: list[Hashable | dict[Hashable, float]],
    arr: xr.DataArray,
    *,
    extend_to_edge: bool,
) -> tuple[
    list[dict[Hashable, float]],
    list[Hashable],
    collections.defaultdict[Hashable, set[float]],
]:
    parsed_interpolation_points: list[dict[Hashable, float]] = []
    for x in interpolation_points:
        if isinstance(x, Hashable):
            parsed_interpolation_points.append(
                _extract_symmetry_point(
                    x,
                    arr,
                    extend_to_edge=extend_to_edge,
                ),
            )
        else:
            parsed_interpolation_points.append(x)
    seen_coordinates = collections.defaultdict(set)
    free_coordinates = list(arr.dims)
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

    return parsed_interpolation_points, free_coordinates, seen_coordinates


def _extract_symmetry_point(
    name: Hashable,
    arr: xr.DataArray,
    *,
    extend_to_edge: bool = False,
) -> dict[Hashable, float]:
    """[TODO:summary].

    Args:
        name (str):  Name of the symmetry points, such as G, X, L.
        arr (xr.DataArray): ARPES data.
        extend_to_edge (bool): [TODO:description]

    Returns: dict(Hashable, float)
        Return dict object as the symmetry point
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
    return dict(zip([d for d in arr.dims if d in raw_point], S, strict=True))


def _element_distance(
    waypoint_a: Mapping[Hashable, float],
    waypoint_b: Mapping[Hashable, float],
) -> np.float64:
    delta: NDArray[np.float64] = np.array(
        [waypoint_a[k] - waypoint_b[k] for k in waypoint_a],
        dtype=np.float64,
    )
    return np.float64(np.linalg.norm(delta))


def _required_sampling_density(
    arr: xr.DataArray,
    waypoint_a: Mapping[Hashable, float],
    waypoint_b: Mapping[Hashable, float],
) -> float:
    ks = waypoint_a.keys()
    dist = _element_distance(waypoint_a, waypoint_b)
    delta = np.array([waypoint_a[k] - waypoint_b[k] for k in ks])
    delta_idx = [
        abs(d / (arr.coords[k][1] - arr.coords[k][0])) for d, k in zip(delta, ks, strict=True)
    ]
    return dist / np.max(delta_idx)
