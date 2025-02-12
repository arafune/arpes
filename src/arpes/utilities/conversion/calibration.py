"""Preliminary detector window corrections."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import scipy.interpolate
import xarray as xr

if TYPE_CHECKING:
    from numpy._typing import NDArray

__all__ = ("DetectorCalibration",)


def build_edge_from_list(points: list[dict[str, float]]) -> xr.Dataset:
    """Converts from a list of edge waypoints to a common representation as a DataSet."""
    dimensions = set(itertools.chain(*[p.keys() for p in points]))
    arrays = {}
    for dim in dimensions:
        values = [p[dim] for p in points]
        arrays[dim] = xr.DataArray(values, coords={dim: values}, dims=[dim])
    return xr.Dataset(arrays)


class DetectorCalibration:
    """A detector calibration model allowing for correcting the trapezoidal windowing."""

    _left_edge: xr.Dataset
    _right_edge: xr.Dataset

    def __init__(self, left: list[dict[str, float]], right: list[dict[str, float]]) -> None:
        """Build the edges for the calibration from a path for the left and right sides."""
        assert set(left[0].keys()) == {
            "phi",
            "eV",
        }
        self._left_edge = build_edge_from_list(left)
        self._right_edge = build_edge_from_list(right)
        # if the edges were passed incorrectly then do it ourselves
        if self._left_edge.phi.mean() > self._right_edge.phi.mean():
            self._left_edge, self._right_edge = self._right_edge, self._left_edge

    def __repr__(self) -> str:
        """Representation showing detailed attributes on edge locations."""
        rep = "<DetectorCalibration>\n\n"
        rep += "Left Edge\n"
        rep += str(self._left_edge)
        rep += "\n\nRightEdge\n"
        rep += str(self._right_edge)
        return rep

    def correct_detector_angle(
        self,
        eV: NDArray[np.float64],  # noqa: N803
        phi: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Applies a calibration to the detector `phi` angle."""
        left, right = (
            np.interp(x=0, xp=self._left_edge.eV.values, fp=self._left_edge.phi.values),
            np.interp(x=0, xp=self._right_edge.eV.values, fp=self._right_edge.phi.values),
        )
        xs = np.concatenate([self._left_edge.eV.values, self._right_edge.eV.values])
        ys = np.concatenate([self._left_edge.phi.values, self._right_edge.phi.values])
        zs = np.concatenate(
            [self._left_edge.eV.values * 0 + left, self._right_edge.eV.values * 0 + right],
        )
        return scipy.interpolate.griddata(np.stack([xs, zs], axis=1), ys, (eV, phi))
