"""Contains routines for converting directly from angle to momentum.

This cannot be done easily for volumetric data because otherwise we will
not end up with an even grid. As a result, we typically use utilities here
to look at the forward projection of a single point or collection of
points/coordinates under the angle -> momentum transform.

Additionally, we have exact inverses for the volumetric transforms which are
useful for aligning cuts which use those transforms.
See `convert_coordinate_forward`.
"""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Literal, TypeGuard, TypeVar, Unpack

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from arpes._typing import is_dict_kspacecoords
from arpes.analysis.filters import gaussian_filter_arr
from arpes.debug import setup_logger
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

from .bounds_calculations import (
    euler_to_kx,
    euler_to_ky,
    euler_to_kz,
    full_angles_to_k,
)
from .core import convert_to_kspace

if TYPE_CHECKING:
    from collections.abc import Hashable

    from arpes._typing import KspaceCoords, XrTypes

__all__ = (
    "convert_coordinate_forward",
    "convert_coordinates",
    "convert_coordinates_to_kspace_forward",
    "convert_through_angular_pair",
    "convert_through_angular_point",
)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

A = TypeVar("A", NDArray[np.float64], float)


def convert_coordinate_forward(
    data: xr.DataArray,
    coords: dict[Hashable, float],
    **k_coords: Unpack[KspaceCoords],
) -> dict[Hashable, float]:
    """Inverse/forward transform for the small angle volumetric k-conversion code.

    This differs from the other forward transforms here which are exact,
    up to correct assignment of offset constants.

    This makes this routine very practical for determining the location of cuts to be taken
    around a point or direction of interest in k-space. If you use the exact methods to determine
    the location of interest in k-space then in general there will be some misalignment because
    the small angle volumetric transform is not the inverse of the exact forward transforms.

    The way that we accomplish this is that the data is copied and a "test charge" is placed in the
    data which distinguished the location of interest in angle-space. The data is converted with
    the volumetric interpolation methods, and the location of the "test charge" is determined in
    k-space. With the approximate location in k determined, this process is repeated once more with
    a finer k-grid to determine more precisely the forward transform location.

    A nice property of this approach is that it is automatic because it determines the result
    numerically using the volumetric transform code. Any changes to the volumetric code will
    automatically reflect here. However, it comes with a few downsides:

    #. The "test charge" is placed in a cell in the original data. This means that the resolution is
       limited by the resolution of the dataset in angle-space. This could be circumvented by
       regridding the data to have a higher resolution.
    #. The procedure can only be performed for a single point at a time.
    #. The procedure is relatively expensive.

    Another approach would be to write down the exact small angle approximated transforms.

    Args:
        data (XrTypes): The data defining the coordinate offsets and experiment geometry.
            (should be DataArray, and S.spectrum_type is "cut" or "map".)
        coords (dict[str, float]): The coordinates of a **point** in *angle*-space to be converted.
        k_coords: Coordinate for k-axis

    Returns:
        The location of the desired coordinate in **momentum**.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert data.spectrum_type in {"cut", "map"}, 'spectrum type must be "cut" or "map"'
    if data.spectrum_type == "map":
        if "eV" in coords:  # TODO: correction is required for "cut" data
            coords = dict(coords)
            energy_coord = coords.pop("eV")
            data = data.sel(eV=energy_coord, method="nearest")
        elif "eV" in data.dims:
            warnings.warn(
                """You didn't specify an energy coordinate for the high symmetry point but the
                dataset you provided has an energy dimension. This will likely be very
                slow. Where possible, provide an energy coordinate
                """,
                stacklevel=2,
            )
        if not k_coords:
            k_coords = {
                "kx": np.linspace(-4, 4, 300),
                "ky": np.linspace(-4, 4, 300),
            }
    elif not k_coords:  # data.spectrum_type = map
        k_coords = {"kp": np.linspace(-4, 4, 300)}
    # Copying after taking a constant energy plane is much much cheaper
    data = data.copy(deep=True)

    data.loc[data.G.round_coordinates(coords)] = data.values.max() * 100000
    data = gaussian_filter_arr(data, default_size=3)
    kdata = convert_to_kspace(data, coords=k_coords)
    near_target: dict[Hashable, float] = kdata.G.argmax_coords()
    if "eV" in near_target and data.spectrum_type == "cut":
        del near_target["eV"]
    coords_around_target = {k: np.linspace(v - 0.08, v + 0.08, 100) for k, v in near_target.items()}
    if is_dict_kspacecoords(coords_around_target):
        kdata_close = convert_to_kspace(
            data,
            coords=coords_around_target,
        )
    else:
        msg = "Incorrect coordinate."
        raise RuntimeError(msg)

    # inconsistently, the energy coordinate is sometimes returned here
    # so we remove it just in case
    coords = kdata_close.G.argmax_coords()
    if "eV" in coords:
        del coords["eV"]
    return coords


def convert_through_angular_pair(  # noqa: PLR0913
    data: xr.DataArray,
    first_point: dict[Hashable, float],
    second_point: dict[Hashable, float],
    cut_specification: dict[str, NDArray[np.float64]],
    transverse_specification: dict[str, NDArray[np.float64]],
    *,
    relative_coords: bool = True,
    **k_coords: NDArray[np.float64],
) -> xr.DataArray:
    """Converts the lower dimensional ARPES cut passing through `first_point` and `second_point`.

    This is a sibling method to `convert_through_angular_point`. A point and a `chi` angle
    fix a plane in momentum, as do two points. This method implements the two points
    equivalent whereas `convert_through_angular_point` fixes the latter.

    A difference between this function and `convert_through_angular_point` is that
    the momentum range requested by `cut_specification` is considered a margin outside
    the interval defined by `first_point` and `second_point`. I.e.
    np.linspace(0, 0, 400) would produce a 400 point interpolation between
    `first_point` and `second_point` whereas passing `np.linspace(-1, 0, 400)` would provide
    a left margin past `first_point` of 1 inverse angstrom.

    This endpoint relative behavior can be disabled with the `relative_coords` flag.

    Args:
        data: The angle space data to be converted.
        first_point: The angle space coordinates of the first point the cut should
          pass through.
        second_point: The angle space coordinates of the second point the cut should
          pass through. This point will have larger momentum coordinate in the output
          data than `first_point`, i.e. it will appear on the "right side" of the data
          if the data is plotted as a cut.
        cut_specification: A dictionary specifying the momentum varying axes
          on the output data. Interpreted as defining boundaries relative to
          the endpoints.
        transverse_specification: A dictionary specifying the transverse (summed)
          axis on the momentum converted data.
        relative_coords: Whether to give `cut_specification` relative to the momentum
          converted location specified in `coords`
        k_coords: Passed as hints through to `convert_coordinate_forward`.

    Returns:
        The momentum cut passing first through `first_point` and then through `second_point`.
    """
    assert data.spectrum_type == "map"
    k_first_point = convert_coordinate_forward(data, first_point, **k_coords)
    k_second_point = convert_coordinate_forward(data, second_point, **k_coords)

    k_dims = set(k_first_point.keys())
    if k_dims != {"kx", "ky"}:
        msg = f"Two point {k_dims} momentum conversion is not supported yet."
        raise NotImplementedError(msg)

    assert k_dims == set(cut_specification.keys()).union(transverse_specification.keys())
    assert "ky" in transverse_specification  # You must use ky as the transverse coordinate for now
    assert len(cut_specification) == 1

    offset_ang = np.arctan2(
        k_second_point["ky"] - k_first_point["ky"],
        k_second_point["kx"] - k_first_point["kx"],
    )
    logger.debug(f"Determined offset angle {-offset_ang}")

    with data.S.with_rotation_offset(-offset_ang):
        logger.debug("Finding first momentum coordinate.")
        k_first_point = convert_coordinate_forward(data, first_point, **k_coords)
        logger.debug("Finding second momentum coordinate.")
        k_second_point = convert_coordinate_forward(data, second_point, **k_coords)

        # adjust output coordinate ranges
        transverse_specification = {
            k: v + k_first_point[k] for k, v in transverse_specification.items()
        }

        # here, we assume we were passed an array for simplicities sake
        # we could also allow a slice in the future
        parallel_axis = next(iter(cut_specification.values()))
        parallel_dim = next(iter(cut_specification.keys()))
        if relative_coords:
            delta = parallel_axis[1] - parallel_axis[0]
            left_margin, right_margin = parallel_axis[0], parallel_axis[-1] + delta

            left_point = k_first_point["kx"] + left_margin
            right_point = k_second_point["kx"] + right_margin
            parallel_axis = np.linspace(left_point, right_point, len(parallel_axis))

        # perform the conversion
        logger.debug("Performing final momentum conversion.")
        logger.debug(f"key of transverse_specification : {list(transverse_specification.keys())}")
        transverse_specification_dims = list(transverse_specification.keys())
        transverse_specification.update(kx=parallel_axis)
        if is_dict_kspacecoords(transverse_specification):
            converted_data = convert_to_kspace(
                data,
                coords=transverse_specification,
            ).mean(transverse_specification_dims)
        else:
            msg = "Incorrect transverse_specification"
            raise RuntimeError(msg)
        logger.debug("Annotating the requested point momentum values.")
        return converted_data.assign_attrs(
            {
                "first_point_kx": k_first_point[parallel_dim],
                "second_point_kx": k_second_point[parallel_dim],
                "offset_angle": -offset_ang,
            },
        )


def convert_through_angular_point(
    data: xr.DataArray,
    coords: dict[Hashable, float],
    cut_specification: dict[str, NDArray[np.float64]],
    transverse_specification: dict[str, NDArray[np.float64]],
    *,
    relative_coords: bool = True,
    **k_coords: NDArray[np.float64],
) -> xr.DataArray:
    """Converts the lower dimensional ARPES cut passing through given angular `coords`.

    The fixed momentum axis is given by `cut_specification` in coordinates relative to
    the momentum converted copy of `coords`. Absolute coordinates can be enabled
    with the `relative_coords` flag.


    Args:
        data: The angle space data to be converted.
        coords: The angle space coordinates of the point the cut should pass through.
        cut_specification: A dictionary specifying the momentum varying axes
          on the output data.
        transverse_specification: A dictionary specifying the transverse (summed)
          axis on the momentum converted data.
        relative_coords: Whether to give `cut_specification` relative to the momentum
          converted location specified in `coords`
        k_coords: Passed as hints through to `convert_coordinate_forward`.

    Returns:
        A momentum cut passing through the point `coords`.
    """
    location_in_kspace = convert_coordinate_forward(
        data,
        coords,
        **k_coords,
    )
    all_momentum_dims = set(location_in_kspace.keys())
    assert all_momentum_dims == set(cut_specification.keys()).union(transverse_specification.keys())

    # adjust output coordinate ranges
    transverse_specification = {
        k: v + location_in_kspace[k] for k, v in transverse_specification.items()
    }
    if relative_coords:
        cut_specification = {k: v + location_in_kspace[k] for k, v in cut_specification.items()}
    transverse_specification.update(cut_specification)
    tarnsverse_dimensions = [
        dim for dim in transverse_specification if dim not in cut_specification
    ]
    # perform the conversion
    if is_dict_kspacecoords(transverse_specification) and is_dict_kspacecoords(cut_specification):
        converted_data = convert_to_kspace(
            data,
            coords=transverse_specification,
        ).mean(tarnsverse_dimensions, keep_attrs=True)
    else:
        msg = "Incorrect transverse_specification/cut_specification"
        raise RuntimeError(msg)

    for k, v in location_in_kspace.items():
        converted_data.attrs[f"highsymm_{k}"] = v

    return converted_data


@update_provenance("Forward convert coordinates")
def convert_coordinates(
    arr: XrTypes,
    *,
    collapse_parallel: bool = False,
) -> xr.Dataset:
    """Converts coordinates forward in momentum."""

    def unwrap_coord(coord: xr.DataArray | float) -> NDArray[np.float64] | float:
        if isinstance(coord, xr.DataArray):
            return coord.values
        return coord

    coord_names: set[str] = {"phi", "psi", "alpha", "theta", "beta", "chi", "hv", "eV"}
    raw_coords: dict[str, NDArray[np.float64] | float] = {
        k: unwrap_coord(arr.S.lookup_offset_coord(k)) for k in coord_names
    }
    raw_angles = {k: v for k, v in raw_coords.items() if k not in {"eV", "hv"}}

    parallel_collapsible: bool = (
        len([k for k in raw_angles if isinstance(raw_angles[k], np.ndarray)]) > 1
    )

    sort_by = ["eV", "hv", "phi", "psi", "alpha", "theta", "beta", "chi"]
    old_dims = sorted(
        [str(k) for k in arr.dims if k in coord_names],
        key=sort_by.index,
    )

    will_collapse = parallel_collapsible and collapse_parallel

    def expand_to(
        cname: str,
        c: NDArray[np.float64] | float,
    ) -> NDArray[np.float64] | float:
        if isinstance(c, float):
            return c
        assert isinstance(c, np.ndarray)
        index_list: list[None | slice] = [np.newaxis] * len(old_dims)
        assert old_dims.index(cname) is not None
        index_list[old_dims.index(cname)] = slice(None, None)
        return c[tuple(index_list)]

    # build the full kinetic energy array over relevant dimensions
    if arr.S.energy_notation == "Binding":
        kinetic_energy = (
            expand_to("eV", raw_coords["eV"])
            + expand_to("hv", raw_coords["hv"])
            - arr.S.analyzer_work_function
        )
    elif arr.S.energy_notation == "Final":
        kinetic_energy = expand_to("eV", raw_coords["eV"]) - arr.S.analyzer_work_function
    else:
        warnings.warn(
            "Energy notation is not specified. Assume the Binding energy notation",
            stacklevel=2,
        )
        kinetic_energy = (
            expand_to("eV", raw_coords["eV"])
            + expand_to("hv", raw_coords["hv"])
            - arr.S.analyzer_work_function
        )

    kx, ky, kz = full_angles_to_k(
        kinetic_energy,
        inner_potential=arr.S.inner_potential,
        **{k: expand_to(k, v) for k, v in raw_angles.items()},
    )

    if will_collapse:
        if np.sum(kx**2) > np.sum(ky**2):
            sign = kx / np.sqrt(kx**2 + 1e-8)
        else:
            sign = ky / np.sqrt(ky**2 + 1e-8)

        kp = sign * np.sqrt(kx**2 + ky**2)
        data_vars = {"kp": (old_dims, np.squeeze(kp)), "kz": (old_dims, np.squeeze(kz))}
    else:
        data_vars = {
            "kx": (old_dims, np.squeeze(kx)),
            "ky": (old_dims, np.squeeze(ky)),
            "kz": (old_dims, np.squeeze(kx)),
        }

    return xr.Dataset(data_vars, coords=arr.indexes)


def _is_dims_match_coordinate_convert(
    angles: tuple[str, ...],
) -> TypeGuard[
    tuple[Literal["phi"]]
    | tuple[Literal["theta"]]
    | tuple[Literal["beta"]]
    | tuple[Literal["phi"], Literal["theta"]]
    | tuple[Literal["beta"], Literal["phi"]]
    | tuple[Literal["hv"], Literal["phi"]]
    | tuple[Literal["hv"]]
    | tuple[Literal["beta"], Literal["hv"], Literal["phi"]]
    | tuple[Literal["hv"], Literal["phi"], Literal["theta"]]
    | tuple[Literal["hv"], Literal["phi"], Literal["psi"]]
    | tuple[Literal["chi"], Literal["hv"], Literal["phi"]]
]:
    return angles in {
        ("phi",),
        ("theta",),
        ("beta",),
        ("phi", "theta"),
        ("beta", "phi"),
        ("hv", "phi"),
        ("hv",),
        ("beta", "hv", "phi"),
        ("hv", "phi", "theta"),
        ("hv", "phi", "psi"),
        ("chi", "hv", "phi"),
    }


@update_provenance("Forward convert coordinates to momentum")
def convert_coordinates_to_kspace_forward(arr: XrTypes) -> xr.Dataset:
    """Forward converts all the individual coordinates of the data array.

    Args:
        arr: [TODO:description]
    """
    arr = arr.copy(deep=True)

    skip = {"eV", "cycle", "delay", "T"}
    all_indexes = {k: v for k, v in arr.indexes.items() if k not in skip}
    momentum_compatibles: list[str] = sorted(all_indexes.keys())
    if not momentum_compatibles:
        msg = "Cannot convert because no momentum compatible coordinate."
        raise RuntimeError(msg)

    tupled_momentum_compatibles = tuple(momentum_compatibles)
    dest_coords = []
    if _is_dims_match_coordinate_convert(tupled_momentum_compatibles):
        dest_coords = {
            ("phi",): ["kp", "kz"],
            ("theta",): ["kp", "kz"],
            ("beta",): ["kp", "kz"],
            ("phi", "theta"): ["kx", "ky", "kz"],
            ("beta", "phi"): ["kx", "ky", "kz"],
            ("hv", "phi"): ["kx", "ky", "kz"],
            ("hv",): ["kp", "kz"],
            ("beta", "hv", "phi"): ["kx", "ky", "kz"],
            ("hv", "phi", "theta"): ["kx", "ky", "kz"],
            ("hv", "phi", "psi"): ["kx", "ky", "kz"],
            ("chi", "hv", "phi"): ["kx", "ky", "kz"],
        }.get(tupled_momentum_compatibles, [])
    full_old_dims: list[str] = [*momentum_compatibles, "eV"]
    projection_vectors: NDArray[np.float64] = np.ndarray(
        shape=tuple(len(arr.coords[d]) for d in full_old_dims),
        dtype=object,
    )

    raw_coords = {
        "phi": arr.coords["phi"].values - arr.S.phi_offset,
        "beta": (0 if arr.coords["beta"] is None else arr.coords["beta"].values)
        - arr.S.beta_offset,
        "theta": (0 if arr.coords["theta"] is None else arr.coords["theta"].values)
        - arr.S.theta_offset,
        "hv": arr.coords["hv"],
    }
    raw_coords = {
        k: _broadcast_by_dim_location(
            data=v,
            target_shape=projection_vectors.shape,
            dim_location=full_old_dims.index(k) if k in full_old_dims else None,
        )
        for k, v in raw_coords.items()
    }

    # fill in the vectors
    binding_energy = _broadcast_by_dim_location(
        data=arr.coords["eV"] - arr.S.analyzer_work_function,
        target_shape=projection_vectors.shape,
        dim_location=full_old_dims.index("eV") if "eV" in full_old_dims else None,
    )
    photon_energy = _broadcast_by_dim_location(
        data=arr.coords["hv"],
        target_shape=projection_vectors.shape,
        dim_location=full_old_dims.index("hv") if "hv" in full_old_dims else None,
    )
    if arr.S.energy_notation == "Binding":
        kinetic_energy = binding_energy + photon_energy
    elif arr.S.energy_notation == "Final":
        kinetic_energy = binding_energy
    else:
        warnings.warn(
            "Energy notation is not specified. Assume the Binding energy notation",
            stacklevel=2,
        )
        kinetic_energy = binding_energy + photon_energy

    inner_potential = arr.S.inner_potential

    raw_translated = {
        "kx": euler_to_kx(
            kinetic_energy,
            phi=raw_coords["phi"],
            beta=raw_coords["beta"],
            theta=0,
            slit_is_vertical=arr.S.is_slit_vertical,
        ),
        "ky": euler_to_ky(
            kinetic_energy,
            phi=raw_coords["phi"],
            beta=raw_coords["beta"],
            theta=0,
            slit_is_vertical=arr.S.is_slit_vertical,
        ),
        "kz": euler_to_kz(
            kinetic_energy,
            phi=raw_coords["phi"],
            beta=raw_coords["beta"],
            theta=0,
            slit_is_vertical=arr.S.is_slit_vertical,
            inner_potential=inner_potential,
        ),
    }
    if "kp" in dest_coords:
        if np.sum(raw_translated["kx"] ** 2) > np.sum(raw_translated["ky"] ** 2):
            sign = raw_translated["kx"] / np.sqrt(raw_translated["kx"] ** 2 + 1e-8)
        else:
            sign = raw_translated["ky"] / np.sqrt(raw_translated["ky"] ** 2 + 1e-8)

        raw_translated["kp"] = np.sqrt(raw_translated["kx"] ** 2 + raw_translated["ky"] ** 2) * sign
    data_vars = {}
    for dest_coord in dest_coords:
        data_vars[dest_coord] = (full_old_dims, np.squeeze(raw_translated[dest_coord]))
    return xr.Dataset(data_vars, coords=arr.indexes)


# these are a little special, depending on the scan type we might not have a phi coordinate
# that aspect of this is broken for now, but we need not worry
def _broadcast_by_dim_location(
    data: xr.DataArray,
    target_shape: tuple[int, ...],
    dim_location: int | None = None,
) -> NDArray[np.float64]:
    if isinstance(data, xr.DataArray) and not data.dims:
        data = data.item()
    if isinstance(
        data,
        int | float,
    ):
        return np.ones(target_shape) * data
    # else we are dealing with an actual array
    the_slice: list[None | slice] = [None] * len(target_shape)
    assert dim_location is not None
    the_slice[dim_location] = slice(None, None, None)
    return np.asarray(data)[the_slice]

    # some notes on angle conversion:
    # BL4 conventions
    # angle conventions are standard:
    # phi = analyzer acceptance
    # polar = perpendicular scan angle
    # theta = parallel to analyzer slit rotation angle

    # [ 1  0          0          ]   [  cos(polar) 0 sin(polar) ]   [ 0          ]
    # [ 0  cos(theta) sin(theta) ] * [  0          1 0          ] * [ k sin(phi) ]
    # [ 0 -sin(theta) cos(theta) ]   [ -sin(polar) 0 cos(polar) ]   [ k cos(phi) ]
    #
    # =
    #
    # [ 1  0          0          ]     [ sin(polar) * cos(phi) ]
    # [ 0  cos(theta) sin(theta) ] * k [ sin(phi) ]
    # [ 0 -sin(theta) cos(theta) ]     [ cos(polar) * cos(phi) ]
    #
    # =
    #
    # k ( sin(polar) * cos(phi),
    #
    # main chamber conventions, with no analyzer rotation
    # (referred to as alpha angle in the Igor code)
    # angle conventions are standard:
    # phi = analyzer acceptance
    # polar = perpendicular scan angle
    # theta = parallel to analyzer slit rotation angle

    # [ 1 0 0                    ]     [ sin(phi + theta) ]
    # [ 0 cos(polar) sin(polar)  ] * k [ 0                ]
    # [ 0 -sin(polar) cos(polar) ]     [ cos(phi + theta) ]
    #
    # =
    #
    # k (sin(phi + theta), cos(phi + theta) * sin(polar), cos(phi + theta) cos(polar), )
    #

    # for now we are setting the theta angle to zero, this only has an effect for
    # vertical slit analyzers, and then only when the tilt angle is very large
