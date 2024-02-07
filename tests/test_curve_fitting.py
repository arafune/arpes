"""Unit test for curve fitting."""

import numpy as np

from arpes.analysis import rebin
from arpes.fits import AffineBroadenedFD, broadcast_model
from arpes.io import example_data

TOLERANCE = 1e-4


def test_broadcast_fitting() -> None:
    """Test broadcast fitting."""
    cut = example_data.cut.spectrum
    near_ef = cut.isel(phi=slice(80, 120)).sel(eV=slice(-0.2, 0.1))
    near_ef = rebin(near_ef, phi=5)

    fit_results = broadcast_model([AffineBroadenedFD], near_ef, "phi", progress=False)

    assert np.abs(fit_results.F.p("a_fd_center").values.mean() + 0.00506558) < TOLERANCE

    fit_results = broadcast_model(
        [AffineBroadenedFD],
        near_ef,
        "phi",
        parallelize=True,
        progress=True,
    )
    assert fit_results.results.F.parameter_names == {
        "a_const_bkg",
        "a_conv_width",
        "a_fd_center",
        "a_fd_width",
        "a_lin_bkg",
        "a_offset",
    }
    assert fit_results.results.F.band_names == {"a_fd_"}
    assert fit_results.F.broadcast_dimensions == ["phi"]
    assert fit_results.F.fit_dimensions == ["eV"]
