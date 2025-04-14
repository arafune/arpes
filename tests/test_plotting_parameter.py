import xarray as xr
from matplotlib.axes import Axes

from arpes.plotting.parameter import plot_parameter


def test_plot_parameter(fitresult_fermi_edge_correction: xr.Dataset):
    param_name = "center"
    fit_results = fitresult_fermi_edge_correction.modelfit_results
    ax = plot_parameter(
        fit_data=fit_results,
        param_name=param_name,
        shift=1.0,
        x_shift=0.5,
        two_sigma=True,
        color="red",
    )

    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "phi"
    assert ax.get_ylabel() == "center"

    # Check if the plot contains data
    assert len(ax.lines) > 0
    assert ax.lines[0].get_color() == "red"


def test_F_plot_param(fitresult_fermi_edge_correction: xr.Dataset):
    fit_results = fitresult_fermi_edge_correction.modelfit_results
    ax = fit_results.F.plot_param("center")
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "phi"
    assert ax.get_ylabel() == "center"

    # Check if the plot contains data
    assert len(ax.lines) > 0
