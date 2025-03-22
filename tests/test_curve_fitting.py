"""Unit test for curve fitting."""

import numpy as np
import xarray as xr
from lmfit.models import ConstantModel

from arpes.analysis import rebin
from arpes.fits import AffineBroadenedFD

RTOL = 5e-2  # 5 %
TOLERANCE = 1e-2


def test_broadcast_fitting(dataarray_cut: xr.DataArray) -> None:
    """Test broadcast fitting."""
    near_ef = dataarray_cut.isel(phi=slice(80, 120)).sel(eV=slice(-0.2, 0.1))
    near_ef_rebin = rebin(near_ef, phi=5)
    model = AffineBroadenedFD(prefix="a_") + ConstantModel()

    params = AffineBroadenedFD(prefix="a_").guess(
        near_ef_rebin.isel(phi=4).values,
        near_ef_rebin.coords["eV"].values,
    )
    fit_results = near_ef_rebin.S.modelfit("eV", model, params=params)
    a_band_data = fit_results.modelfit_results.F.bands["a_"]
    np.testing.assert_allclose(
        a_band_data.center.values,
        np.array(
            [
                -0.15701477,
                -0.19971782,
                -0.11516844,
                -0.21612229,
                -0.06146412,
                -0.17533675,
                -0.1897528,
                -0.31079581,
            ],
        ),
        rtol=RTOL,  # TODO: [RA]  Consider why this value strongly depends on the platform.
    )
    np.testing.assert_allclose(
        actual=a_band_data.sigma.values,
        desired=np.array(
            [
                0.05145544,
                0.05377813,
                0.04796474,
                0.04098757,
                0.02604283,
                0.03924345,
                0.04505744,
                0.04447656,
            ],
        ),
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        actual=a_band_data.amplitude,
        desired=np.array((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)),
    )
    assert (
        np.abs(fit_results.modelfit_results.F.p("a_center").mean().item() + 0.17817159792226375)
        < TOLERANCE
    )


#    fit_results = broadcast_model(
#        [AffineBroadenedFD],
#        near_ef_rebin,
#        "phi",
#        parallelize=True,
#        prefixes=("a_",),
#        params={
#            "a_center": {"value": 0.0, "vary": True, "min": -0.1},
#            "a_width": {"value": 0.1},
#            "a_lin_slope": {"value": 20000, "max": 30000, "min": 10000},
#        },
#    )
#    assert fit_results.results.F.parameter_names == {
#        "a_const_bkg",
#        "a_conv_width",
#        "a_center",
#        "a_width",
#        "a_lin_slope",
#        "a_offset",
#    }
#
#    assert fit_results.results.F.band_names == {"a_"}
#    np.testing.assert_allclose(
#        fit_results.results.F.mean_square_error().values,
#        np.array(
#            [
#                13769.07645062,
#                15927.48553538,
#                13030.76068365,
#                11847.62221047,
#                9828.60331845,
#                16164.81630826,
#                12844.01730529,
#                11887.61788115,
#            ],
#        ),
#        rtol=RTOL,
#    )
#
#    params_ = fit_results.results.F.param_as_dataset("a_conv_width")
#    np.testing.assert_allclose(
#        params_["value"].values,
#        np.array(
#            [
#                0.02348614,
#                0.02656286,
#                0.02529021,
#                0.01889499,
#                0.01715083,
#                0.01831382,
#                0.02238334,
#                0.02063916,
#            ],
#        ),
#        rtol=RTOL,
#    )
