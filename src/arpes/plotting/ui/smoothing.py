"""Provides a Holoviews & Panel based implementation of ARPES data manipulation tools.

This module defines interactive visualization tools based on Holoviews for use in ARPES data

All visualizations are designed to work with `xarray.DataArray` or `xarray.Dataset` and are
rendered via the `bokeh` backend of Holoviews.

Dependencies:
    - holoviews
    - panel
    - numpy
    - xarray
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Unpack

import panel as pn
import holoviews as hv
import numpy as np
from holoviews import AdjointLayout, DynamicMap, Image, QuadMesh

from arpes.debug import setup_logger

from ._helper import default_plot_kwargs, fix_xarray_to_fit_with_holoview, get_image_options


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

hv.extension("bokeh", logo=False)
pn.extension()


class SmoothingApp:
    def __init__(self, data: xrDataArray, output_var_name: str = "smoothed") -> None:
        self.data = data
        self.output_var_name = output_var_name
        self.outpu = data.copy()

        self.smoothing_func = {
            "None": (self.none_smoothing, {}),
            "Gaussian": (
                self.gaussian_smoothing,
                {
                    "sigma": pn.widgets.FloatSlider(
                        name="Sigma",
                        start=0.1,
                        end=10.0,
                        step=0.1,
                        value=1.0,
                    ),
                    "iteration": pn.widgets.IntSlider(
                        value=1,
                        start=1,
                        end=10,
                        step=1,
                        name="Iterations",
                    ),
                },
            ),
            "Savitzky-Golay": (
                self.savitzky_golay_smoothing,
                {
                    "window_length": pn.widgets.IntSlider(
                        value=5,
                        start=1,
                        step=2,
                        end=20,
                        name="widdow_length",
                    ),
                    "polyorder": pn.widgets.IntSlider(
                        value=2,
                        start=1,
                        end=20,
                        step=1,
                        name="polyorder",
                    ),
                },
            ),
            "Uniform": (
                self.uniform_smoothing,
                {
                    "size": pn.widgets.IntSlider(value=3, start=1, end=20),
                    "iteration": pn.widgets.IntSlider(
                        value=1,
                        start=1,
                        end=10,
                        step=1,
                        name="Iterations",
                    ),
                },
            ),
        }
