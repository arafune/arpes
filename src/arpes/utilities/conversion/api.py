"""Public API for momentum-space (k-space) conversion of ARPES data.

This module defines the user-facing entry points for converting ARPES
data from angle/energy space into momentum space. The functions provided
here are stable, documented, and intended to be imported and used directly
by analysis scripts and notebooks.

Design principles:
    - This module exposes *high-level orchestration functions* only.
    - Implementation details (coordinate transforms, interpolation,
      grid construction) are delegated to lower-level modules.
    - Functions here operate on `xarray.DataArray` objects and return
      new objects without modifying inputs in place.

Scope:
    - Backward (interpolating) k-space conversion via `convert_to_kspace`
    - Chunk-aware conversion for large energy stacks
    - Provenance tracking for reproducibility

Non-goals:
    - This module does not implement low-level coordinate transforms.
    - This module does not define interpolation kernels or numerical solvers.
    - Forward (coordinate-only) conversion routines live elsewhere.

Typical usage:
    >>> from arpes.utilities.conversion.api import convert_to_kspace
    >>> kdata = convert_to_kspace(data, kx=..., ky=...)

Module structure:
    - convert_to_kspace:
        Primary public entry point for k-space conversion.
    - _chunk_convert:
        Internal helper for chunk-wise conversion (not part of the public API).

See Also:
    - arpes.utilities.conversion.coordinates:
        Low-level coordinate transformation utilities.
    - arpes.utilities.conversion.core:
        Internal conversion machinery and interpolation logic.
"""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Unpack

import numpy as np
import xarray as xr

from arpes.debug import setup_logger
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum
from arpes.xarray_extensions.accessor.spectrum_type import AngleUnit

from .coordinates import convert_coordinates
from .core import _is_dims_match_coordinate_convert
from .grids import (
    determine_momentum_axes_from_measurement_axes,
    is_dimension_convertible_to_momentum,
)
from .kx_ky_conversion import ConvertKp, ConvertKxKy
from .kz_conversion import ConvertKpKz

if TYPE_CHECKING:
    from collections.abc import Hashable

    from numpy.typing import NDArray

    from arpes._typing.attrs_property import KspaceCoords
    from arpes._typing.base import MOMENTUM
    from arpes.utilities.conversion.base import CoordinateConverter
    from arpes.utilities.conversion.calibration import DetectorCalibration


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@update_provenance("Automatically k-space converted")
def convert_to_kspace(  # noqa: PLR0913
    arr: xr.DataArray,
    *,
    bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    resolution: dict[MOMENTUM, float] | None = None,
    calibration: DetectorCalibration | None = None,
    coords: KspaceCoords | None = None,
    allow_chunks: bool = False,
    **kwargs: Unpack[KspaceCoords],
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
        coords (KspaceCoords, optional): Coordinate of k-space. Defaults to {}.
        allow_chunks (bool): [description]. Defaults to False.
        **kwargs: treated as coords.

    Raises:
        NotImplementedError: [description]
        AnalysisError: [description]
        ValueError: [description]

    Returns:
        xr.DataArray: Converted ARPES (k-space) data.
    """
    coords = {} if coords is None else coords
    assert coords is not None
    coords.update(kwargs)

    bounds = bounds or {}
    arr = arr if isinstance(arr, xr.DataArray) else normalize_to_spectrum(arr)
    assert isinstance(arr, xr.DataArray)
    if arr.S.angle_unit is AngleUnit.DEG:
        arr = arr.S.switched_angle_unit()
    logger.debug(f"bounds (covnert_to_kspace): {bounds}")
    logger.debug(f"keys in coords (convert_to_kspace): {coords.keys()}")
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
        str(d)
        for d in arr.dims
        if not is_dimension_convertible_to_momentum(str(d)) and str(d) != "eV"
    ]
    momentum_compatibles: list[str] = sorted(  # Literal["phi", "theta", "beta", "chi", "psi", "hv"]
        [str(d) for d in arr.dims if is_dimension_convertible_to_momentum(str(d))],
    )

    if not momentum_compatibles:
        return arr  # no need to convert, might be XPS or similar

    converted_dims: list[str] = (
        (["eV"] if ("eV" in arr.dims) else [])
        + determine_momentum_axes_from_measurement_axes(
            momentum_compatibles,
        )  # axis_names: list[Literal["phi", "beta", "psi", "theta", "hv"]],
        + momentum_incompatibles
    )

    tupled_momentum_compatibles = tuple(momentum_compatibles)
    convert_cls: type[ConvertKp | ConvertKxKy | ConvertKpKz] | None = None
    if _is_dims_match_coordinate_convert(tupled_momentum_compatibles):
        convert_cls = {
            ("phi",): ConvertKp,
            ("beta", "phi"): ConvertKxKy,
            ("phi", "theta"): ConvertKxKy,
            ("phi", "psi"): ConvertKxKy,
            # ('chi', 'phi',): ConvertKxKy,
            ("hv", "phi"): ConvertKpKz,
        }.get(tupled_momentum_compatibles)
    assert convert_cls is not None, "Cannot select convert class"

    converter: CoordinateConverter = convert_cls(
        arr=arr,
        dim_order=converted_dims,
        calibration=calibration,
    )

    converted_coordinates: dict[Hashable, NDArray[np.floating]] = converter.get_coordinates(
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
        target_coordinates=converted_coordinates,
        coordinate_transform={
            "dims": converted_dims,
            "transforms": {str(dim): converter.conversion_for(dim) for dim in arr.dims},
        },
    )
    assert isinstance(result, xr.DataArray)
    return result


def _chunk_convert(
    arr: xr.DataArray,
    bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    resolution: dict[MOMENTUM, float] | None = None,
    calibration: DetectorCalibration | None = None,
    coords: KspaceCoords | None = None,
    **kwargs: Unpack[KspaceCoords],
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
            chunk = chunk.squeeze(dim="eV")
        kchunk = convert_to_kspace(
            arr=chunk,
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
