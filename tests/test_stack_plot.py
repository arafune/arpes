"""Unit test for statck_dispersion_plot."""

import numpy as np
import pytest
import xarray as xr
from arpes.plotting import stack_plot


class TestHelperFunction:
    """Test class for helper function in stack_plot."""

    def test_y_shifted(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for helperfuncion, _y_shifted."""
        an_iterator = dataarray_cut2.G.iter_coords("phi")
        first_coord = next(an_iterator)
        ys0 = stack_plot._y_shifted(
            offset_correction="zero",
            coord_value=first_coord["phi"],
            marginal=dataarray_cut2.sel(first_coord),
            scale_parameters=(1, 10, False),
        )
        np.testing.assert_allclose(
            ys0[:5],
            np.array(
                [
                    -9.82091280e-02,
                    -1.56042628e-01,
                    -1.45321028e-01,
                    -1.69142328e-01,
                    -1.16088128e-01,
                ],
            ),
        )
        ys1 = stack_plot._y_shifted(
            offset_correction="constant",
            coord_value=first_coord["phi"],
            marginal=dataarray_cut2.sel(first_coord),
            scale_parameters=(1, 10, False),
        )
        np.testing.assert_allclose(
            ys1[:5],
            np.array([-0.21780313, -0.27563663, -0.26491503, -0.28873633, -0.23568213]),
        )

    def test__rebinning(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for helperfuncsion, _rebinning."""
        rebinning = stack_plot._rebinning(dataarray_cut2, stack_axis="phi", max_stacks=10)
        assert rebinning[1] == "phi"
        assert rebinning[2] == "eV"
        np.testing.assert_allclose(
            rebinning[0].values[0][:10],
            np.array(
                [
                    36.2613544,
                    43.2389617,
                    40.6978181,
                    37.3331042,
                    42.6655625,
                    44.6055878,
                    37.40546,
                    43.4389916,
                    40.0674895,
                    36.31281796,
                ],
            ),
        )

        not_rebinning = stack_plot._rebinning(dataarray_cut2, stack_axis="phi", max_stacks=830)
        np.testing.assert_allclose(not_rebinning[0].values, dataarray_cut2.values)

    def test__scale_factor(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for helperfuncsion, _scale_factor."""
        scale_factor = stack_plot._scale_factor(dataarray_cut2, "phi")
        np.testing.assert_allclose(scale_factor, 0.19045560735480058)
        np.testing.assert_allclose(
            stack_plot._scale_factor(dataarray_cut2, "phi", offset_correction="constant"),
            desired=0.19896425702750428,
        )
        np.testing.assert_allclose(
            stack_plot._scale_factor(dataarray_cut2, "phi", offset_correction="constant_right"),
            desired=0.19896425702750428,
        )

        np.testing.assert_allclose(
            stack_plot._scale_factor(dataarray_cut2, "phi", offset_correction=None),
            desired=0.19184807723402728,
        )


class TestStackDispersionPlot:
    """Test class for stack_dispersion_plot."""

    def test_statck_dispersion_plot_1(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for stack_dispersion_plot.

        Todo:
            Add more assert.
        """
        _, ax = stack_plot.stack_dispersion_plot(
            data=dataarray_cut2,
            max_stacks=20,
            scale_factor=20,
            title="2PPE Xe/Au(111)",
            linewidth=0.3,
            color="plasma",
            shift=0.0,
            mode="hide_line",
        )
        assert ax.get_title() == "2PPE Xe/Au(111)"

    def test_statck_dispersion_plot_2(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for stack_dispersion_plot.

        with fill_between_mode.

        Todo:
            Add more assert.
        """
        _, ax = stack_plot.stack_dispersion_plot(
            data=dataarray_cut2,
            max_stacks=10,
            linewidth=0.3,
            color="plasma",
            mode="fill_between",
            offset_correction="zero",
            shift=0.0,
        )
        assert len(ax.collections) == 10
        paths = ax.collections
        np.testing.assert_allclose(
            paths[0].get_paths()[0].vertices[:3],
            np.array([[9.0, 0.19602282], [9.0, 0.19632079], [9.002, 0.19634603]]),
        )
        np.testing.assert_allclose(
            actual=paths[-1].get_paths()[0].vertices[:3],
            desired=np.array([[9.0, -0.19602282], [9.0, -0.19578132], [9.002, -0.19573485]]),
        )
        xmin, xmax = ax.get_xlim()
        assert xmin == 8.95
        assert xmax == 10.05
        ymin, ymax = ax.get_ylim()
        assert ymin == -0.2157909190282624
        assert ymax == 0.21910736470983783


class TestFlatStackPlot:
    """Test class for flat_stack_plot."""

    def test_flat_stack_plot(self, dataarray_cut2: xr.DataArray) -> None:
        """Testing for flat_stack_plot.

        Todo:
            Add more assert.

        """
        _, ax = stack_plot.flat_stack_plot(
            dataarray_cut2,
            max_stacks=10,
            title="2PPE Xe/Au(111)",
            linewidth=0.5,
            color="plasma",
            label="label test",
            mode="line",
            figsize=(7, 5),
        )

        lines = ax.lines
        assert len(lines) == 10

    def test_arg_flat_stack_plot_should_be_2_dimensional(
        self,
        dataarray_cut2: xr.DataArray,
    ) -> None:
        """Test for checck if the data is 2D in flat_stack_plot."""
        with pytest.raises(IndexError):
            _, ax = stack_plot.flat_stack_plot(data=dataarray_cut2.sum("phi"))
