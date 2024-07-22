"""Unit test for plotting/holoviews.py."""

import xarray as xr
from holoviews.core.layout import AdjointLayout

from arpes.plotting import profile_view


class TestProfileView:
    """Class for profile_view function."""

    def test_basic_profile_view(self, dataarray_cut2: xr.DataArray) -> None:
        img = profile_view(dataarray_cut2)
        assert isinstance(img, AdjointLayout)
