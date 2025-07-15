"""
Unit tests for the gradient fill plotting functions in `decoration.py`.

This test suite verifies the behavior of `h_gradient_fill` and `v_gradient_fill`
functions which produce horizontal and vertical gradient-filled regions in a matplotlib Axes.

All logical branches and visual side effects (such as `imshow` and `fill_between` calls)
are covered using `unittest.mock.patch` and `pytest.raises`.

Test coverage includes:
- Gradient fill with and without solid regions.
- Explicit and implicit axes (`ax` vs. `plt.gca()`).
- Alpha blending and step parameter behavior.
- Invalid parameter ranges (e.g., x1 >= x2) triggering assertions.

Mocks are used to isolate matplotlib calls and verify that expected plotting methods
are invoked the correct number of times.

Requires:
    pytest
    pytest-cov
    matplotlib

Example:
    $ pytest --cov=decoration tests/test_plotting_decoration.py
"""

from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
from matplotlib.image import AxesImage

from arpes.plotting.decoration import h_gradient_fill, v_gradient_fill


def test_h_gradient_fill_no_solid_with_patch():
    fig, ax = plt.subplots()
    with (
        patch.object(ax, "imshow") as mock_imshow,
        patch.object(ax, "fill_betweenx") as mock_fill_betweenx,
    ):
        mock_imshow.return_value = AxesImage(ax)
        im = h_gradient_fill(0.0, 1.0, None, fill_color="blue", ax=ax, alpha=0.5)

        mock_imshow.assert_called_once()
        mock_fill_betweenx.assert_not_called()
        assert isinstance(im, AxesImage)


def test_h_gradient_fill_with_solid_right():
    fig, ax = plt.subplots()
    with (
        patch.object(ax, "imshow") as mock_imshow,
        patch.object(ax, "fill_betweenx") as mock_fill_betweenx,
    ):
        mock_imshow.return_value = AxesImage(ax)
        im = h_gradient_fill(1.0, 2.0, 3.0, fill_color="red", ax=ax, alpha=1.0, step="post")

        mock_imshow.assert_called_once()
        mock_fill_betweenx.assert_called_once()
        assert isinstance(im, AxesImage)


def test_h_gradient_fill_with_solid_left():
    fig, ax = plt.subplots()
    with (
        patch.object(ax, "imshow") as mock_imshow,
        patch.object(ax, "fill_betweenx") as mock_fill_betweenx,
    ):
        mock_imshow.return_value = AxesImage(ax)
        im = h_gradient_fill(1.0, 2.0, 0.5, fill_color="green", ax=ax, alpha=0.8, step="mid")

        mock_imshow.assert_called_once()
        mock_fill_betweenx.assert_called_once()
        assert isinstance(im, AxesImage)


def test_v_gradient_fill_no_solid():
    fig, ax = plt.subplots()
    with (
        patch.object(ax, "imshow") as mock_imshow,
        patch.object(ax, "fill_between") as mock_fill_between,
    ):
        mock_imshow.return_value = AxesImage(ax)
        im = v_gradient_fill(0.0, 1.0, None, fill_color="purple", ax=ax, alpha=0.5)

        mock_imshow.assert_called_once()
        mock_fill_between.assert_not_called()
        assert isinstance(im, AxesImage)


def test_v_gradient_fill_with_solid_top():
    fig, ax = plt.subplots()
    with (
        patch.object(ax, "imshow") as mock_imshow,
        patch.object(ax, "fill_between") as mock_fill_between,
    ):
        mock_imshow.return_value = AxesImage(ax)
        im = v_gradient_fill(0.0, 2.0, 3.0, fill_color="orange", ax=ax, alpha=0.7, step="pre")

        mock_imshow.assert_called_once()
        mock_fill_between.assert_called_once()
        assert isinstance(im, AxesImage)


def test_v_gradient_fill_with_solid_bottom():
    fig, ax = plt.subplots()
    with (
        patch.object(ax, "imshow") as mock_imshow,
        patch.object(ax, "fill_between") as mock_fill_between,
    ):
        mock_imshow.return_value = AxesImage(ax)
        im = v_gradient_fill(1.0, 4.0, 0.0, fill_color="black", ax=ax, alpha=1.0)

        mock_imshow.assert_called_once()
        mock_fill_between.assert_called_once()
        assert isinstance(im, AxesImage)


def test_h_gradient_fill_with_default_ax(monkeypatch):
    fig, ax = plt.subplots()

    # monkeypatch plt.gca() to return our test ax
    monkeypatch.setattr(plt, "gca", lambda: ax)

    with (
        patch.object(ax, "imshow") as mock_imshow,
        patch.object(ax, "fill_betweenx") as mock_fill_betweenx,
    ):
        mock_imshow.return_value = AxesImage(ax)
        im = h_gradient_fill(0.0, 2.0, None, fill_color="blue", alpha=0.5)

        mock_imshow.assert_called_once()
        mock_fill_betweenx.assert_not_called()
        assert isinstance(im, AxesImage)


def test_v_gradient_fill_with_default_ax(monkeypatch):
    fig, ax = plt.subplots()

    monkeypatch.setattr(plt, "gca", lambda: ax)

    with (
        patch.object(ax, "imshow") as mock_imshow,
        patch.object(ax, "fill_between") as mock_fill_between,
    ):
        mock_imshow.return_value = AxesImage(ax)
        im = v_gradient_fill(0.0, 2.0, None, fill_color="gray", alpha=1.0)

        mock_imshow.assert_called_once()
        mock_fill_between.assert_not_called()
        assert isinstance(im, AxesImage)


def test_h_gradient_fill_invalid_range():
    fig, ax = plt.subplots()

    with pytest.raises(AssertionError):
        h_gradient_fill(3.0, 1.0, None, fill_color="blue", ax=ax)


def test_v_gradient_fill_invalid_range():
    fig, ax = plt.subplots()

    with pytest.raises(AssertionError):
        v_gradient_fill(5.0, 2.0, None, fill_color="red", ax=ax)
