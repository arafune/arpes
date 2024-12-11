"""Unit test for tarpes.py."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.special import erf

from arpes.analysis import tarpes


def temporal_from_rate(
    t: float | NDArray[np.float64],
    g: float,
    sigma: float,
    k_ex: float,
    t0: float = 0,
) -> float | NDArray[np.float64]:
    """Temporal profile.

    From a rate equation, which is used in (for example)
    10.1021/acs.nanolett.3c03251
    """
    return (
        (g / 2)
        * np.exp(k_ex * t0 + sigma**2 * (k_ex**2) / 2)
        * np.exp(-k_ex * t)
        * (
            erf((t - t0 + (sigma**2) * k_ex) / (sigma * np.sqrt(2)))
            + erf((t0 + (sigma**2) * k_ex) / (sigma * np.sqrt(2)))
        )
    )


n_data = 150
pixel = 20
position = np.linspace(100, 103, n_data)
delaytime = np.linspace(-100e-15, 2500e-15, n_data)
rng = np.random.default_rng(42)
noise = rng.normal(loc=0, scale=0.01, size=n_data)
tempo_intensity = (
    temporal_from_rate(
        t=delaytime,
        g=1,
        sigma=50e-15,
        k_ex=2e12,
        t0=0.2e-12,
    )
    + noise
    + 0.02
)

mock_tarpes = [
    xr.DataArray(
        data=rng.integers(100, size=pixel * pixel).reshape(pixel, pixel) * tempo_intensity[i],
        dims=["phi", "eV"],
        coords={
            "phi": np.linspace(np.deg2rad(-10), np.deg2rad(10), pixel),
            "eV": np.linspace(5, 6, pixel),
        },
        attrs={"position": position[i], "id": int(i + 1)},
    )
    for i in range(n_data)
]


def test_find_t_for_max_intensity():
    """Test for find_t_for_max_intensity."""
    tarpes_dataarray = tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=100.31,
    )
    assert tarpes.find_t_for_max_intensity(tarpes_dataarray) == 1021.2881894590657
    tarpes_dataarray = tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=100.31,
        convert_position_to_time=False,
    )
    assert tarpes.find_t_for_max_intensity(tarpes_dataarray) == 0.15308724832215148
