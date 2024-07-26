"""Unit test for plotting/holoviews.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arpes.analysis import rebin
from arpes.fits import AffineBroadenedFD, broadcast_model
from arpes.plotting import concat_along_phi_ui, fit_inspection, profile_view
from holoviews import DynamicMap
from holoviews.core.layout import AdjointLayout, Layout

if TYPE_CHECKING:
    import xarray as xr


class TestProfileView:
    """Class for profile_view function."""

    def test_basic_profile_view(self, dataarray_cut2: xr.DataArray) -> None:
        img = profile_view(dataarray_cut2)
        assert isinstance(img, AdjointLayout)

    def test_concat_along_phi_ui(self, mote2_1: xr.Dataset, mote2_2: xr.Dataset) -> None:
        img = concat_along_phi_ui(
            mote2_1.spectrum.S.corrected_angle_by("beta"),
            mote2_2.spectrum.S.corrected_angle_by("beta"),
            width=500,
        )
        assert isinstance(img, DynamicMap)

    def test_fit_inspection(self, dataarray_cut: xr.DataArray) -> None:
        near_ef = dataarray_cut.isel(phi=slice(80, 120)).sel(eV=slice(-0.2, 0.1))
        near_ef_rebin = rebin(near_ef, phi=5)

        fit_results = broadcast_model(
            [AffineBroadenedFD],
            near_ef_rebin,
            "phi",
            prefixes=("a_",),
            params={
                "a_center": {"value": 0.0, "vary": True, "min": -0.1},
                "a_width": {"value": 0.1},
                "a_lin_bkg": {"value": 20000, "max": 30000, "min": 10000},
            },
            progress=False,
        )
        img = fit_inspection(fit_results)
        assert isinstance(img, Layout)
