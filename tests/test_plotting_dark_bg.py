"""Unit test for dark background mode."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

from arpes.plotting.dark_bg import (
    DEFAULT_DARK_MODE,
    apply_dark_to_ax,
    apply_dark_to_colorbar,
    apply_dark_to_figure,
    dark_background,
    get_dark_mode_params,
)
from arpes.plotting.utils import get_colorbars


def test_dark_background_contextmanager():
    original_color = plt.rcParams["axes.edgecolor"]
    overrides = {"axes.edgecolor": "red"}  # RcParamKey compliant override
    with dark_background(overrides):
        # Inside the context, the rcParam should be overridden
        assert plt.rcParams["axes.edgecolor"] == "red"
    # After exiting the context, rcParam should be restored
    assert plt.rcParams["axes.edgecolor"] == original_color


def test_get_dark_mode_params_copy():
    overrides = {"axes.edgecolor": "blue"}
    params = get_dark_mode_params(overrides)
    # The returned dict should reflect the overrides
    assert params["axes.edgecolor"] == "blue"
    # DEFAULT_DARK_MODE should remain unchanged

    assert DEFAULT_DARK_MODE["axes.edgecolor"] == "white"


def test_apply_dark_to_colorbar_outline_invisible():
    fig, ax = plt.subplots()
    im = ax.imshow([[1, 2], [3, 4]])
    cbar = fig.colorbar(im)

    cbar.outline.set_visible(False)

    apply_dark_to_colorbar(cbar)

    assert cbar.outline.get_visible() is False


def test_apply_dark_to_colorbar_with_xlabel():
    fig, ax = plt.subplots()
    im = ax.imshow([[1, 2], [3, 4]])
    cbar = fig.colorbar(im)

    # ← これが必要
    cbar.ax.set_xlabel("Intensity")

    apply_dark_to_colorbar(cbar)

    assert cbar.ax.xaxis.label.get_color() == "white"


def test_get_dark_mode_params_transparent_true():
    params = get_dark_mode_params(transparent=True)

    assert params["axes.facecolor"] == "none"
    assert params["figure.facecolor"] == "none"
    assert params["savefig.facecolor"] == "none"


def test_get_dark_mode_params_transparent_false():
    params = get_dark_mode_params(transparent=False)

    assert params["axes.facecolor"] == "black"
    assert params["figure.facecolor"] == "black"
    assert params["savefig.facecolor"] == "black"


def _make_fig_with_colorbar():
    """Utility: Return Figure + Axes + Colorbar."""
    fig, ax = plt.subplots()
    data = np.random.rand(5, 5)
    im = ax.imshow(data)
    cbar = fig.colorbar(im)
    return fig, ax, cbar


def test_rcparams_restored_after_context():
    """The rcParams are correctly restored inside and outside the context."""
    key = "axes.edgecolor"
    original = plt.rcParams[key]

    with dark_background():
        assert plt.rcParams[key] == DEFAULT_DARK_MODE[key]

    assert plt.rcParams[key] == original


def test_rcparams_override():
    """DEFAULT_DARK_MODE is overridden by overrides."""
    key = "axes.edgecolor"
    with dark_background({key: "red"}):
        assert plt.rcParams[key] == "red"


def test_figure_none_uses_gcf():
    """If fig=None, plt.gcf() is implicitly used and apply_dark is called."""
    fig = plt.figure()
    plt.figure(fig.number)  # make it current

    ax = fig.add_subplot()
    im = ax.imshow([[1, 2], [3, 4]])
    _ = fig.colorbar(im)

    with dark_background():
        pass

    assert ax.get_facecolor()[3] == 0  # RGBA  A=0 → 'none'
    for spine in ax.spines.values():
        assert spine.get_edgecolor() == (1.0, 1.0, 1.0, 1.0)

    cbars = get_colorbars(fig)
    assert cbars
    cb = cbars[0]
    assert cb.outline.get_edgecolor() == (1.0, 1.0, 1.0, 1.0)


def test_dark_applies_to_axes_and_colorbar():
    """Axes and Colorbar are correctly darkened."""
    fig, ax, _ = _make_fig_with_colorbar()

    with dark_background():
        pass

    assert ax.get_facecolor()[3] == 0
    for spine in ax.spines.values():
        assert spine.get_edgecolor() == (1.0, 1.0, 1.0, 1.0)

    xtick_color = ax.xaxis.get_ticklabels()[0].get_color()
    assert xtick_color == "white"

    cbars = get_colorbars(fig)
    assert len(cbars) == 1
    cb = cbars[0]

    assert cb.outline.get_edgecolor() == (1.0, 1.0, 1.0, 1.0)

    tick = cb.ax.get_yticklabels()[0]
    assert tick.get_color() == "white"


def test_dark_inside_context_new_figure():
    """Figures created inside the context are also darkened."""
    with dark_background():
        fig, ax, _ = _make_fig_with_colorbar()

    assert ax.get_facecolor()[3] == 0

    for spine in ax.spines.values():
        assert spine.get_edgecolor() == (1.0, 1.0, 1.0, 1.0)

    ticks = ax.xaxis.get_ticklabels()
    if ticks:
        assert ticks[0].get_color() == "white"

    # Colorbar
    cbars = get_colorbars(fig)
    assert cbars
    cb = cbars[0]
    assert cb.outline.get_edgecolor() == (1.0, 1.0, 1.0, 1.0)


def test_rcparams_applied_inside_context():
    """The rcParams should change inside the context and be restored after exit."""
    original = plt.rcParams["axes.edgecolor"]

    with dark_background():
        assert plt.rcParams["axes.edgecolor"] == "white"

    assert plt.rcParams["axes.edgecolor"] == original


def test_apply_dark_to_ax_changes_colors():
    """apply_dark_to_ax should modify spine and text colors."""
    _, ax = plt.subplots()
    ax.set_title("Test Title")

    apply_dark_to_ax(ax)

    for spine in ax.spines.values():
        assert spine.get_edgecolor() == (1.0, 1.0, 1.0, 1.0)

    assert ax.xaxis.label.get_color() == "white"
    assert ax.yaxis.label.get_color() == "white"


def test_apply_dark_to_ax_transparent_true():
    _, ax = plt.subplots()

    apply_dark_to_ax(ax, transparent=True)

    assert ax.get_facecolor()[3] == 0  # fully transparent
    for spine in ax.spines.values():
        assert spine.get_edgecolor() == (1, 1, 1, 1)


def test_apply_dark_to_ax_transparent_false():
    _, ax = plt.subplots()

    apply_dark_to_ax(ax, transparent=False)

    assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)  # black


def test_apply_dark_to_colorbar_changes_colors():
    """apply_dark_to_colorbar should update outline, ticks, and labels."""
    fig, ax = plt.subplots()
    im = ax.imshow([[1, 2], [3, 4]])
    cbar = fig.colorbar(im)

    cbar.ax.set_ylabel("Intensity")
    apply_dark_to_colorbar(cbar)

    assert cbar.outline.get_edgecolor() == (1, 1, 1, 1)
    assert cbar.ax.yaxis.label.get_color() == "white"


def _make_colorbar():
    fig, ax = plt.subplots()
    sm = ScalarMappable(
        norm=mpl.colors.Normalize(0, 1),
        cmap="viridis",
    )
    cbar = fig.colorbar(sm, ax=ax)
    return fig, cbar


def test_apply_dark_to_colorbar_transparent_true():
    _, cbar = _make_colorbar()

    apply_dark_to_colorbar(cbar, transparent=True)

    assert cbar.ax.get_facecolor()[3] == 0  # transparent
    assert cbar.outline.get_edgecolor() == (1, 1, 1, 1)


def test_apply_dark_to_colorbar_transparent_false():
    _, cbar = _make_colorbar()

    apply_dark_to_colorbar(cbar, transparent=False)

    assert cbar.ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    assert cbar.outline.get_facecolor()[:3] == (0.0, 0.0, 0.0)


def test_dark_background_styles_axes_and_colorbars():
    """dark_background should apply styling after the context exits."""
    fig, ax = plt.subplots()
    im = ax.imshow([[1, 2], [3, 4]])
    cbar = fig.colorbar(im)

    with dark_background(fig=fig):
        pass

    # Axes
    assert ax.spines["bottom"].get_edgecolor() == (1, 1, 1, 1)
    assert ax.xaxis.label.get_color() == "white"

    # Colorbar
    assert isinstance(cbar, Colorbar)
    assert cbar.outline.get_edgecolor() == (1, 1, 1, 1)


def test_dark_background_transparent_facecolor():
    """Figure and Axes should have transparent facecolors."""
    fig, ax = plt.subplots()
    with dark_background(fig=fig):
        pass

    # Figure facecolor should be transparent (alpha=0)
    fc = fig.get_facecolor()
    assert fc[3] == 0  # RGBA alpha channel

    # Axes facecolor should be transparent (alpha=0)
    ax_fc = ax.get_facecolor()
    assert ax_fc[3] == 0


def test_apply_dark_to_figure_transparent_true():
    fig, ax = plt.subplots()
    apply_dark_to_figure(fig, transparent=True)

    assert fig.patch.get_facecolor()[3] == 0


def test_apply_dark_to_figure_transparent_false():
    fig, ax = plt.subplots()
    apply_dark_to_figure(fig, transparent=False)

    assert fig.patch.get_facecolor()[:3] == (0.0, 0.0, 0.0)


def test_dark_background_context_transparent_false():
    fig, ax = plt.subplots()

    with dark_background(fig=fig, transparent=False):
        pass

    # applied AFTER exiting the context
    assert fig.patch.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)


def test_dark_background_context_transparent_true():
    fig, ax = plt.subplots()

    with dark_background(fig=fig, transparent=True):
        pass

    assert fig.patch.get_facecolor()[3] == 0
    assert ax.get_facecolor()[3] == 0
