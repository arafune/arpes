"""Build a mock for tarpes data."""

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.special import erf

__all__ = ("build_mock_tarpes", "temporal_from_rate")


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


def build_mock_tarpes(n_data: int = 150, pixel: int = 20) -> list[xr.DataArray]:
    """Build a tarpes mock.

    Args:
        n_data (int): size of temporal data
        pixel (int): size of each ARPES mock data (Square shape is assumed).

    Returns:
        list[xr.DataArray]
    """
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

    return [
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
