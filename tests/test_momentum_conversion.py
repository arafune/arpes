"""Unit test for k-conversion."""
import numpy as np
import pytest
import xarray as xr

import arpes.xarray_extensions  # pylint: disable=unused-import, redefined-outer-name  # noqa: F401
from arpes.io import example_data
from arpes.utilities.conversion import convert_to_kspace
from arpes.utilities.conversion.forward import convert_through_angular_point


def load_energy_corrected() -> xr.DataArray:
    """Loading map data (example_data.map)."""
    return example_data.map.spectrum


def test_cut_momentum_conversion() -> None:
    """Validates that the core APIs are functioning."""
    kdata = convert_to_kspace(example_data.cut.spectrum, kp=np.linspace(-0.12, 0.12, 600))
    selected = kdata.values.ravel()[[0, 200, 800, 1500, 2800, 20000, 40000, 72000]]
    assert np.nan_to_num(selected).tolist() == [
        pytest.approx(c)
        for c in [
            0.0,
            329.93641856427644,
            282.81464843603896,
            258.6560679332663,
            201.5580256163084,
            142.11410841363914,
            293.0273837097613,
            0.0,
        ]
    ]


def test_cut_momentum_conversion_ranges() -> None:
    """Validates that the user can select momentum ranges."""
    data = example_data.cut.spectrum
    kdata = convert_to_kspace(data, kp=np.linspace(-0.12, 0.12, 80))

    expected_values = """
    237, 202, 174, 157, 169, 173, 173, 177, 165, 171, 159, 160, 154,
    155, 153, 143, 139, 139, 138, 125, 119, 122, 117, 117, 113, 138,
    144, 148, 149, 133, 137, 151, 143, 144, 135, 147, 136, 133, 145,
    139, 136, 138, 143, 133, 131, 139, 145, 139, 144, 133, 143, 155,
    151, 150, 157, 151, 147, 121, 131, 128, 128, 137, 138, 136, 151,
    151, 151, 154, 165, 165, 159, 172, 168, 167, 168, 163, 171, 169,
    160, 150
    """.replace(
        ",",
        "",
    ).split()
    assert kdata.argmax(dim="eV").values.tolist() == [int(m) for m in expected_values]


def test_fermi_surface_conversion() -> None:
    """Validates that the kx-ky conversion code is behaving."""
    data = load_energy_corrected().S.fermi_surface

    kdata = convert_to_kspace(
        data,
        coords={"kx": np.linspace(-2.5, 1.5, 400), "ky": np.linspace(-2, 2, 400)},
    )

    kx_max = kdata.idxmax(dim="ky").max().item()
    ky_max = kdata.idxmax(dim="kx").max().item()

    assert ky_max == pytest.approx(0.4373433583959896)
    assert kx_max == pytest.approx(-0.02506265664160412)
    assert kdata.mean().item() == pytest.approx(613.50919747114)
    assert kdata.fillna(0).mean().item() == pytest.approx(415.330388958026)


def test_convert_angular_point_and_angle() -> None:
    """Validates that we correctly convert through high symmetry points."""
    test_point = {
        "phi": -0.13,
        "theta": -0.1,
        "eV": 0.0,
    }
    data = load_energy_corrected()

    kdata = convert_through_angular_point(
        data,
        test_point,
        {"ky": np.linspace(-1, 1, 400)},
        {"kx": np.linspace(-0.02, 0.02, 10)},
    )

    max_values = [
        4145.34763451004,
        4356.4657536204595,
        4539.154281973837,
        4767.057860841638,
        4966.353725499497,
        5134.052860826806,
        5387.027147279516,
        5562.648608259634,
        5967.406263151459,
        6499.694353077622,
        6864.222962311083,
        7108.975499752097,
        7788.282551108498,
        8157.747570521674,
        8525.72537495676,
        8510.367719895963,
        8256.260480417679,
        7781.137336181722,
        7150.166364522493,
        6760.781766440197,
        6377.054129242761,
        6071.60591854745,
        5919.467723608378,
        5624.884046350201,
        3076.564947442262,
        117.3817138160862,
    ]
    assert kdata.sel(ky=slice(-0.7, 0)).isel(eV=slice(None, -20, 5)).max("ky").values.tolist() == [
        pytest.approx(c) for c in max_values
    ]
