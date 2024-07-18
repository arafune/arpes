"""Syntax suger for combination of ARPES data."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger

import xarray as xr

from arpes.provenance import Provenance, provenance_multiple_parents

__all__ = ("concat_along_phi",)

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


def concat_along_phi(
    arr_a: xr.DataArray,
    arr_b: xr.DataArray,
    occupation_ratio: float | None = None,
    enhance_a: float = 1.0,
) -> xr.DataArray:
    """Montage two arpes data.

    Args:
        arr_a (xr.DataArray): one ARPES data
        arr_b (xr.DataArray): another ARPES data
        occupation_ratio(float | None): Identify the seam of "phi" axis.
        enhance_a: (float): The enhancement factor for arr_a to correct the intensity.

    Returns:
        Concatenated ARPES array
    """
    assert isinstance(arr_a, xr.DataArray)
    assert isinstance(arr_b, xr.DataArray)
    if occupation_ratio is not None:
        assert 0 <= occupation_ratio <= 1, "occupation_ratio should be between 0 and 1 (or None)."
    id_arr_a = arr_a.attrs["id"]
    id_arr_b = arr_b.attrs["id"]
    arr_a = arr_a.G.with_values(
        arr_a.values * enhance_a,
        keep_attrs=True,
    )
    id_add = _combine_id(id_arr_a, id_arr_b)
    if occupation_ratio is None:
        concat_array = xr.concat(
            [arr_a, arr_b],
            dim="phi",
            coords="minimal",
            combine_attrs="drop_conflicts",
        ).sortby("phi")
    else:
        if arr_a.coords["phi"].values.min() < arr_b.coords["phi"].values.min():
            left_arr, right_arr = arr_a, arr_b
        elif arr_a.coords["phi"].values.min() > arr_b.coords["phi"].values.min():
            left_arr, right_arr = arr_b, arr_a
        else:
            msg = "Cannot combine them, because the coordinate of arr_a and arr_b seems to be same."
            raise RuntimeError(
                msg,
            )
        assert (
            left_arr.coords["phi"].values.max() < right_arr.coords["phi"].values.max()
        ), 'Cannot combine them. Try "occupation_ration=None"'
        seam_phi = (
            left_arr.coords["phi"].values.max() - right_arr.coords["phi"].values.min()
        ) * occupation_ratio + right_arr.coords["phi"].values.min()
        concat_array = xr.concat(
            [
                left_arr.sel(phi=slice(None, seam_phi)),
                right_arr.sel(phi=slice(seam_phi, None)),
            ],
            dim="phi",
            coords="minimal",
            combine_attrs="drop_conflicts",
        ).sortby("phi")
    concat_array.attrs["id"] = id_add
    provenance_contents: Provenance = {
        "what": "concat_along_phi",
        "parant_id": (id_arr_a, id_arr_b),
        "occupation_ratio": occupation_ratio,
    }
    if enhance_a != 1.0:
        provenance_contents["enhance_a"] = enhance_a

    provenance_multiple_parents(
        concat_array,
        [arr_a, arr_b],
        record=provenance_contents,
        keep_parent_ref=True,
    )
    return concat_array


def _combine_id(id_a: int | tuple[int, ...], id_b: int | tuple[int]) -> tuple[int, ...]:
    if isinstance(id_a, int) and isinstance(id_b, int):
        return tuple(sorted([id_a, id_b]))
    if isinstance(id_a, tuple) and isinstance(id_b, int):
        return tuple(sorted([*list(id_a), id_b]))
    if isinstance(id_b, tuple) and isinstance(id_a, int):
        return tuple(sorted([*list(id_b), id_a]))
    if isinstance(id_a, tuple) and isinstance(id_b, tuple):
        return tuple(sorted(list(id_b) + list(id_a)))
    msg = "id_a and id_b must be int or tuple"
    raise RuntimeError(msg)
