"""Unit test for align.py."""

import xarray as xr

from arpes.analysis.align import align, align2d, align1d


def test_align2d_by_self_correlation(dataarray_cut: xr.DataArray):
    """Unit test for align2d."""
    assert align(dataarray_cut, dataarray_cut, subpixel=False) == (0, 0)


def test_align2d_by_self_correlation_subpixel(dataarray_cut: xr.DataArray):
    align2d_values = align2d(dataarray_cut, dataarray_cut, subpixel=True)
    assert align2d_values[0] < 0.0002
    assert align2d_values[1] < 0.0002


def test_align1d_by_self_correlation(dataarray_cut: xr.DataArray):
    a_spectrum = dataarray_cut.sel(phi=0, method="nearest")
    assert align(a_spectrum, a_spectrum, subpixel=False) == 0


def test_align1d_by_self_correlation_subpixel(dataarray_cut: xr.DataArray):
    a_spectrum = dataarray_cut.sel(phi=0, method="nearest")
    assert align1d(a_spectrum, a_spectrum, subpixel=True) < 0.002
