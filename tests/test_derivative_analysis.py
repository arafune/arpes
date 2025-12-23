"""Test for derivative procedure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import arpes.xarray_extensions  # noqa: F401
from arpes.analysis import (
    curvature1d,
    curvature2d,
    dn_along_axis,
    gaussian_filter_arr,
    minimum_gradient,
)

if TYPE_CHECKING:
    import xarray as xr
    from _typeshed import Incomplete


def test_dataarray_derivatives(sandbox_configuration: Incomplete) -> None:
    """Test for derivativation of xarray.

    Nick ran into an issue where he could not call dn_along_axis with a smooth function that
    expected a DataArray, but this is supposed to be supported. Issue was translation between
    np.ndarray and xr.DataArray internal to dn_along_axis.

    :param sandbox_configuration:
    :return:
    """

    def wrapped_filter(arr: xr.DataArray) -> xr.DataArray:
        return gaussian_filter_arr(arr, {"eV": 0.05, "phi": np.pi / 180})

    data = sandbox_configuration.load("basic/main_chamber_cut_0.fits").spectrum
    assert not data.S.is_differentiated
    d2_data = dn_along_axis(data, "eV", wrapped_filter, order=2)

    # some random sample
    assert [pytest.approx(v, 1e-3) for v in (d2_data.values[50:55, 60:62].ravel())] == [
        46225.01571650838,
        -5.820766091346741e-11,
        3.2014213502407074e-09,
        46225.01571650361,
        46225.01571650687,
        -1.6298145055770874e-09,
        1.6298145055770874e-09,
        46225.0157165036,
        92450.03143301269,
        46225.01571650588,
    ]
    assert d2_data.S.is_differentiated


class TestCurvature:
    """Test class for curvature analysis."""

    def test_curvature1d(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for curvature1d."""
        curvature1d_ = curvature1d(
            arr=gaussian_filter_arr(arr=dataarray_cut2, sigma={"eV": 0.01}, iteration_n=5),
            dim="eV",
            alpha=0.1,
        )
        assert curvature1d_.S.is_differentiated
        np.testing.assert_allclose(
            curvature1d_.S.fat_sel(phi=0).values[:10],
            np.array(
                [
                    7.67748744e-06,
                    1.15478339e-05,
                    1.54454883e-05,
                    1.53771677e-05,
                    1.49487128e-05,
                    1.39339501e-05,
                    1.21205324e-05,
                    9.34471491e-06,
                    5.52249250e-06,
                    6.73175422e-07,
                ],
            ),
        )

    def test_curvature2d(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for curvature2d."""
        curvature2d_ = curvature2d(
            gaussian_filter_arr(
                arr=dataarray_cut2,
                sigma={"eV": 0.01, "phi": 0.5},
                iteration_n=5,
            ),
            dims=("phi", "eV"),
            alpha=0.1,
        )
        assert curvature2d_.S.is_differentiated
        np.testing.assert_allclose(
            curvature2d_.S.fat_sel(phi=0).values[:10],
            np.array(
                [
                    0.00733302,
                    0.00256763,
                    -0.00067419,
                    0.002817,
                    0.00709423,
                    0.01187314,
                    0.01684305,
                    0.02169049,
                    0.02612197,
                    0.02988408,
                ],
            ),
            rtol=1e-5,
        )

    def test_minimum_gradient(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for minimum_gradient."""
        minimum_gradient_ = minimum_gradient(
            gaussian_filter_arr(arr=dataarray_cut2, sigma={"eV": 0.01, "phi": 1.0}, iteration_n=3),
        )
        assert minimum_gradient_.S.is_differentiated
        np.testing.assert_allclose(
            minimum_gradient_.S.fat_sel(phi=0).values[:10],
            np.array(
                [
                    112.05062133,
                    90.80459471,
                    89.49275982,
                    87.99979717,
                    86.68187383,
                    85.80900488,
                    85.53816062,
                    85.91798979,
                    86.90138614,
                    88.35749916,
                ],
            ),
            rtol=1e-5,
        )
