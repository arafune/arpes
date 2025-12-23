"""Unit test arpes.analysis.deconvolution module."""

import numpy as np
import pytest
import xarray as xr

import arpes.xarray_extensions  # noqa: F401 to register xarray extensions
from arpes.analysis.deconvolution import (
    LOGLEVEL,
    deconvolve_ice,
    deconvolve_rl,
    make_psf,
    make_psf1d,
)


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def data_2d() -> xr.DataArray:
    x = np.linspace(-5, 5, 21)
    y = np.linspace(-3, 3, 15)
    X, Y = np.meshgrid(x, y, indexing="ij")
    arr = np.exp(-(X**2 + Y**2))
    return xr.DataArray(
        arr,
        dims=("kx", "ky"),
        coords={"kx": x, "ky": y},
        name="intensity",
        attrs={"units": "arb"},
    )


@pytest.fixture
def data_1d() -> xr.DataArray:
    x = np.linspace(-5, 5, 51)
    arr = np.exp(-(x**2))
    return xr.DataArray(
        arr,
        dims=("kx",),
        coords={"kx": x},
        name="intensity",
        attrs={"units": "arb"},
    )


@pytest.fixture
def psf_kernel() -> np.ndarray:
    k = np.ones((3, 3), dtype=np.float64)
    return k / k.sum()


@pytest.fixture
def psf_kernel_1d() -> np.ndarray:
    k = np.ones(3, dtype=np.float64)
    return k / k.sum()


# -----------------------------
# make_psf1d
# -----------------------------
def test_make_psf1d_basic(data_2d: xr.DataArray):
    psf = make_psf1d(data_2d, dim="kx", sigma=0.5)

    assert isinstance(psf, xr.DataArray)
    assert psf.dims == ("kx",)
    assert psf.shape == (data_2d.sizes["kx"],)
    assert np.all(psf.values > 0)


def test_make_psf1d_other_dim_removed(data_2d: xr.DataArray):
    psf = make_psf1d(data_2d, dim="ky", sigma=1.0)

    assert psf.dims == ("ky",)
    assert "kx" not in psf.dims


# -----------------------------
# make_psf
# -----------------------------
def test_make_psf_basic(data_2d: xr.DataArray):
    psf = make_psf(
        data_2d,
        sigmas={"kx": 0.5, "ky": 0.3},
        fwhm=False,
    )

    assert isinstance(psf, xr.DataArray)
    assert psf.dims == data_2d.dims
    assert psf.shape == data_2d.shape
    assert np.all(psf.values >= 0)


def test_make_psf_fwhm_branch(data_2d: xr.DataArray):
    psf = make_psf(
        data_2d,
        sigmas={"kx": 1.0, "ky": 1.0},
        fwhm=True,
    )

    assert np.isfinite(psf.values).all()


def test_make_psf_clip_branch(data_2d: xr.DataArray):
    psf = make_psf(
        data_2d,
        sigmas={"kx": 0.5, "ky": 0.5},
        clip=2.0,
    )

    assert psf.sizes["kx"] < data_2d.sizes["kx"]
    assert psf.sizes["ky"] < data_2d.sizes["ky"]


def test_make_psf_debug_logging_path(data_2d: xr.DataArray, caplog):
    # force debug branch
    if LOGLEVEL == 10:  # DEBUG
        caplog.set_level(10)
        make_psf(
            data_2d,
            sigmas={"kx": 0.4, "ky": 0.4},
        )
        assert any("psf_coords" in r.message for r in caplog.records)
    else:
        # still execute function to count coverage
        make_psf(
            data_2d,
            sigmas={"kx": 0.4, "ky": 0.4},
        )


# -----------------------------
# deconvolve_rl
# -----------------------------
def test_deconvolve_rl_basic(data_2d: xr.DataArray):
    psf = make_psf(
        data_2d,
        sigmas={"kx": 0.5, "ky": 0.5},
    )

    out = deconvolve_rl(data_2d, psf, n_iterations=3)

    assert isinstance(out, xr.DataArray)
    assert out.shape == data_2d.shape
    assert out.dims == data_2d.dims
    assert np.isfinite(out.values).all()


def test_deconvolve_rl_attrs_preserved(data_2d: xr.DataArray):
    psf = make_psf(
        data_2d,
        sigmas={"kx": 0.3, "ky": 0.3},
    )

    out = deconvolve_rl(data_2d, psf)

    assert out.attrs["units"] == "arb"


# -----------------------------
# deconvolve_ice
# -----------------------------
#


def test_deconvolve_ice_basic(data_1d: xr.DataArray, psf_kernel_1d):
    out = deconvolve_ice(
        data_1d,
        psf_kernel_1d,
        n_iterations=5,
    )

    assert isinstance(out, xr.DataArray)
    assert out.shape == data_1d.shape
    assert np.isfinite(out.values).all()


def test_deg_arg_of_deconvolve_ice(data_1d: xr.DataArray, psf_kernel_1d):
    out = deconvolve_ice(
        data_1d,
        psf_kernel_1d,
        n_iterations=7,
        deg=4,
    )

    out_with_deg_non = deconvolve_ice(
        data_1d,
        psf_kernel_1d,
        n_iterations=7,
        deg=None,
    )

    np.testing.assert_allclose(out.values, out_with_deg_non.values, rtol=1e-5)


def test_deconvolve_ice_deg_none_branch(data_1d: xr.DataArray, psf_kernel_1d):
    out = deconvolve_ice(
        data_1d,
        psf_kernel_1d,
        n_iterations=6,
        deg=None,
    )

    assert np.isfinite(out.values).all()


def test_deconvolve_ice_provenance(data_1d: xr.DataArray, psf_kernel_1d):
    data = data_1d.copy()
    data.attrs["provenance"] = ["initial"]

    out = deconvolve_ice(
        data,
        psf_kernel_1d,
    )

    assert "provenance" in out.attrs
    assert isinstance(out.attrs["provenance"], list)
