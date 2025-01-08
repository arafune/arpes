"""unittest for plotting.basic.py."""

from unittest.mock import patch

import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from arpes.plotting.basic import make_overview


# ダミーデータを作成するための補助関数
def create_dummy_data(id_value: str) -> xr.DataArray:
    data: xr.DataArray = xr.DataArray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dims=("eV", "x"),
    )
    data.attrs["id"] = id_value  # type: ignore[arg-type]
    return data


def test_make_overview_valid_input() -> None:
    data_all: list[xr.DataArray] = [create_dummy_data(f"ID{i}") for i in range(6)]

    fig, ax = make_overview(data_all, ncols=3)

    assert isinstance(fig, Figure)
    assert len(ax) == len(data_all)
    assert len(ax) == 6


@pytest.mark.skip
def test_make_overview_invalid_input() -> None:
    # data_allがSequenceではない場合
    with pytest.raises(AssertionError):
        make_overview(None)  # type: ignore[arg-type]

    with pytest.raises(AssertionError):
        make_overview({})  # type: ignore[arg-type]


@pytest.mark.skip
def test_make_overview_proper_plotting() -> None:
    data_all: list[xr.DataArray] = [create_dummy_data(f"ID{i}") for i in range(1)]

    # モックでmatplotlibのax.plotの呼び出しを確認
    with (
        patch.object(Axes, "text") as mock_text,
        patch.object(Axes, "pcolormesh") as mock_pcolormesh,
    ):
        make_overview(data_all, ncols=1)
        mock_pcolormesh.assert_called_once()  # pcolormeshが呼び出されていることを確認
        mock_text.assert_called_once()  # textが呼び出されていることを確認


@pytest.mark.skip
@pytest.mark.parametrize(
    ("data_all", "expected_ncols", "expected_nrows"),
    [
        ([create_dummy_data(f"ID{i}") for i in range(4)], 2, 2),  # 4つのデータ
        ([create_dummy_data(f"ID{i}") for i in range(5)], 2, 3),  # 5つのデータ
        ([create_dummy_data(f"ID{i}") for i in range(6)], 3, 2),  # 6つのデータ
    ],
)
def test_make_overview_layout(
    data_all: list[xr.DataArray],
    expected_ncols: int,
    expected_nrows: int,
) -> None:
    fig, ax = make_overview(data_all, ncols=expected_ncols)

    # レイアウトの検証
    nrows: int = len(ax) // expected_ncols
    if len(ax) % expected_ncols:
        nrows += 1
    assert len(ax) == len(data_all)
    assert len(ax) // expected_ncols == expected_nrows
