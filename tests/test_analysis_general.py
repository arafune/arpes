"""Unit test for analysis.general.py."""

import numpy as np
import xarray as xr
from arpes.analysis.general import (
    normalize_by_fermi_distribution,
)


def test_normalize_by_fermi_distribution(dataarray_map: xr.DataArray) -> None:
    cut = dataarray_map.sum("theta", keep_attrs=True).sel(
        eV=slice(-0.2, 0.2),
        phi=slice(-0.25, 0.3),
    )
    cut_at_0 = cut.sel(phi=0, method="nearest")
    np.testing.assert_allclose(
        normalize_by_fermi_distribution(cut_at_0)[:12],
        np.array(
            [
                207347.85665894,
                207305.51187134,
                205025.95785522,
                203731.65188599,
                204150.95864868,
                205776.95861816,
                203399.02005005,
                202427.34777832,
                203042.97930908,
                203049.06216431,
                200666.57147217,
                197436.81713867,
            ],
        ),
    )


def test_normalize_by_fermi_distribution_total_broadening(dataarray_map: xr.DataArray) -> None:
    cut = dataarray_map.sum("theta", keep_attrs=True).sel(
        eV=slice(-0.2, 0.2),
        phi=slice(-0.25, 0.3),
    )
    cut_at_0 = cut.sel(phi=0, method="nearest")
    np.testing.assert_allclose(
        normalize_by_fermi_distribution(cut_at_0, total_broadening=0.03)[:12],
        np.array(
            [
                207707.21226,
                207826.179214,
                205772.205478,
                204806.277626,
                205711.498189,
                208056.485503,
                206664.307747,
                207136.756698,
                209888.552459,
                212969.879364,
                214875.002101,
                217696.089503,
            ],
        ),
    )
