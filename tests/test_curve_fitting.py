"""Unit test for curve fitting."""

import sys

import numpy as np
import xarray as xr

from arpes.analysis import rebin
from arpes.fits import AffineBroadenedFD, LorentzianModel, broadcast_model
from arpes.fits.utilities import parse_model

RTOL = 1e-6 if sys.platform == "darwin" else 1e-2
TOLERANCE = 2e-3


def test_parse_model() -> None:
    """Test parse_model."""
    assert parse_model(AffineBroadenedFD) == AffineBroadenedFD
    assert parse_model((AffineBroadenedFD, LorentzianModel)) == (AffineBroadenedFD, LorentzianModel)
    assert parse_model([AffineBroadenedFD, LorentzianModel]) == [AffineBroadenedFD, LorentzianModel]
    assert parse_model("AffineBroadenedFD + LorentzianModel") == [
        AffineBroadenedFD,
        "+",
        LorentzianModel,
    ]


def test_broadcast_fitting(dataarray_cut: xr.DataArray) -> None:
    """Test broadcast fitting."""
    near_ef = dataarray_cut.isel(phi=slice(80, 120)).sel(eV=slice(-0.2, 0.1))
    near_ef_rebin = rebin(near_ef, phi=5)

    fit_results = broadcast_model(
        [AffineBroadenedFD],
        near_ef_rebin,
        "phi",
        progress=False,
    )
    a_band_data = fit_results.results.F.bands["a_"]
    np.testing.assert_allclose(
        a_band_data.center.values,
        np.array(
            [
                -0.00456955,
                -0.00217572,
                -0.00268341,
                -0.02043154,
                -0.00331786,
                -0.00397203,
                -0.00390515,
                -0.0198554,
            ],
        ),
        rtol=RTOL,
    )
    np.testing.assert_almost_equal(
        actual=a_band_data.sigma,
        desired=np.array((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)),
    )

    np.testing.assert_almost_equal(
        actual=a_band_data.amplitude,
        desired=np.array((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)),
    )
    assert np.abs(fit_results.results.F.p("a_center").mean().item() + 0.00761) < TOLERANCE

    fit_results = broadcast_model(
        [AffineBroadenedFD],
        near_ef_rebin,
        "phi",
        parallelize=True,
        progress=True,
    )
    assert fit_results.results.F.parameter_names == {
        "a_const_bkg",
        "a_conv_width",
        "a_center",
        "a_width",
        "a_lin_bkg",
        "a_offset",
    }

    assert fit_results.results.F.band_names == {"a_"}
    assert fit_results.F.fit_dimensions == ["eV"]
    np.testing.assert_allclose(
        fit_results.results.F.mean_square_error().values,
        np.array(
            [
                1558314.8960161,
                1511851.0381156,
                1458591.605262,
                457007.0366379,
                1279915.8356443,
                1319729.3599192,
                1229990.7381464,
                403335.4076894,
            ],
        ),
        rtol=RTOL,
    )

    params_ = fit_results.results.F.param_as_dataset("a_conv_width")
    np.testing.assert_allclose(
        params_["value"].values,
        np.array(
            [
                2.8879836,
                0.7276058,
                3.0653718,
                0.1415692,
                3.2048873,
                3.3533791,
                3.0634597,
                0.1502903,
            ],
        ),
        rtol=RTOL,
    )
