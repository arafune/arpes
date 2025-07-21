import numpy as np
import pytest
import xarray as xr
from holoviews.core.layout import AdjointLayout
from holoviews.streams import PointerX, PointerY

from arpes.plotting.ui.profile import ProfileApp, profile_view
import arpes.xarray_extensions  # noqa: ANN001


@pytest.fixture
def sample_data():
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 5, 50)
    data = np.exp(-(np.outer(y - 2.5, x - 5) ** 2))  # Gaussian
    da = xr.DataArray(data, coords={"y": y, "x": x}, dims=("y", "x"))
    return da


def test_profile_app_instantiation(sample_data):
    app = ProfileApp(sample_data)
    assert isinstance(app.output_pane.object, AdjointLayout)
    assert "Coordinates:" in app.coord_display()


def test_show_coords_formatting(sample_data):
    app = ProfileApp(sample_data)
    coord_text = app._show_coords(1.23456, 7.89123)
    assert coord_text == "Coordinates: (1.23e+00, 7.89e+00)"


def test_profile_view_default_image(sample_data):
    layout = profile_view(sample_data)
    assert isinstance(layout, AdjointLayout)


def test_profile_view_quadmesh(sample_data):
    layout = profile_view(sample_data, use_quadmesh=True)
    assert isinstance(layout, AdjointLayout)


def test_profile_view_with_custom_pointer(sample_data):
    posx = PointerX(x=5)
    posy = PointerY(y=2.5)
    layout = profile_view(sample_data, posx=posx, posy=posy)
    assert isinstance(layout, AdjointLayout)
