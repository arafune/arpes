import holoviews as hv
import numpy as np
import pytest
import xarray as xr

from arpes.plotting.ui.fit import fit_inspection


@pytest.fixture
def simple_dataset():
    x = np.linspace(0, 10, 11)
    y = np.linspace(-1, 1, 21)
    X, Y = np.meshgrid(x, y, indexing="ij")
    data = np.sin(X) * np.cos(Y)

    ds = xr.Dataset(
        {
            "modelfit_data": (("k", "eV"), data),
            "modelfit_best_fit": (("k", "eV"), data * 0.9),
        },
        coords={"k": x, "eV": y},
    )
    return ds


@pytest.mark.parametrize("use_quadmesh", [True, False])
@pytest.mark.parametrize("log", [True, False])
def test_fit_inspection_layout_structure(simple_dataset, use_quadmesh, log):
    layout = fit_inspection(
        simple_dataset,
        spectral_name="",
        use_quadmesh=use_quadmesh,
        log=log,
        width=400,
        height=300,
        cmap="viridis",
        profile_view_height=200,
    )

    assert isinstance(layout, hv.Layout)
    assert len(layout) == 2

    left_panel = layout[0]
    assert isinstance(left_panel, hv.AdjointLayout)

    main_overlay = left_panel.main
    assert isinstance(main_overlay, hv.DynamicMap)
    # assert any(isinstance(el, (hv.Image, hv.QuadMesh)) for el in main_overlay)

    right_overlay = left_panel.right
    assert isinstance(right_overlay, hv.DynamicMap)
    assert all(isinstance(el, hv.DynamicMap) for el in right_overlay)

    right_panel = layout[1]
    assert isinstance(right_panel, hv.DynamicMap)


def test_fit_inspection_with_spectral_prefix(simple_dataset):
    ds = simple_dataset.rename_vars(
        {
            "modelfit_data": "spectrum_modelfit_data",
            "modelfit_best_fit": "spectrummodelfit_best_fit",
        },
    )
    layout = fit_inspection(
        ds,
        spectral_name="spectrum",
        log=False,
        width=300,
        height=200,
        cmap="gray",
        profile_view_height=100,
    )

    assert isinstance(layout, hv.Layout)
    assert len(layout) == 2
    assert isinstance(layout[0], hv.AdjointLayout)
